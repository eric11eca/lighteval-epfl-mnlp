import os
import logging
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS

from lighteval.utils.utils import EnvConfig
from lighteval.models.abstract_model import ModelInfo
from lighteval.models.utils import _get_dtype, _simplify_name, _get_model_sha
from lighteval.utils.imports import is_accelerate_available
from lighteval.models.transformers.transformers_model import TransformersModelConfig, TransformersModel


logger = logging.getLogger(__name__)

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate.utils import calculate_maximum_sizes, convert_bytes, get_max_memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

distance_strategy_mapping = {
    "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
    "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
    "dot_product": DistanceStrategy.DOT_PRODUCT,
    "jaccard": DistanceStrategy.JACCARD,
    "cosine": DistanceStrategy.COSINE
}


@dataclass
class EmbeddingModelConfig(TransformersModelConfig):
    """Configuration for the embedding model."""

    similarity_fn: str = "cosine"
    top_k: int = 5
    docs_name_or_path: str = "lighteval/knowledge_base"

    def __post_init__(self):
        self.revision = "main"

        return super().__post_init__()

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision="main")


class EmbeddingModel(TransformersModel):
    def __init__(
        self,
        env_config: EnvConfig,
        config: EmbeddingModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self.accelerator = config.accelerator

        self._config = config.init_configs(env_config)
        self._max_length = self._init_max_length(config.max_length)

        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config, env_config)

        self.model = self._create_auto_model(config, env_config)
        self._device = config.accelerator.device if config.accelerator is not None else "cpu"

        torch.set_grad_enabled(False)

        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = config.get_model_sha()
        self.precision = _get_dtype(config.dtype, config=self._config)

        self.model_info = ModelInfo(
            model_name=self.model_name,
            model_sha=self.model_sha,
            model_dtype=self.precision,
            model_size=-1,
        )

        self.docs_name_or_path = config.docs_name_or_path
        self.top_k = config.top_k
        self.similarity_fn = config.similarity_fn
        self.vector_db = self._build_knowledge_base()

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def init_model_parallel(self, model_parallel: bool | None = None) -> Tuple[bool, Optional[dict], Optional[str]]:
        """Compute all the parameters related to model_parallel"""
        if not is_accelerate_available():
            return False, None, None

        self.num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.num_machines = torch.cuda.device_count() // self.num_local_processes
        if self.num_machines == 0:
            logger.info("We are not in a distributed setting. Setting model_parallel to False.")
            model_parallel = False

        if model_parallel is None:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            model_parallel = bool(self.num_local_processes < len(max_memory_all_gpus))
            logger.info(
                f"Setting model parallel to {model_parallel} since "
                f"the number of local processes is {self.num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )
        if model_parallel is True:
            max_memory_all_gpus = get_max_memory()  # A dict of the max memory for all the gpus
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_mem_this_process = {
                k: v
                for k, v in max_memory_all_gpus.items()
                if k % self.num_local_processes == (self.accelerator.process_index % self.num_local_processes)
            }
            device_map = "auto"
            logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_mem_this_process} and device map to {device_map}"
            )
        else:
            max_mem_this_process = None
            device_map = None
            logger.info(
                f"Model parallel was set to False, max memory set to {max_mem_this_process} and device map to {device_map}"
            )
        return model_parallel, max_mem_this_process, device_map

    def _create_auto_model(
        self, config: TransformersModelConfig, env_config: EnvConfig
    ) -> HuggingFaceEmbeddings:
        """
        Creates an instance of the pretrained HF embedding model through the SentenceTransformer loader.
        Requires the pkg `sentence-transformers` to be installed.

        Args:
            config (TransformersModelConfig): The configuration for the model.
            env_config (EnvConfig): The environment configuration.

        Returns:
            HuggingFaceEmbeddings: The created auto model instance for embedding.
        """
        config.model_parallel, max_memory, device_map = self.init_model_parallel(config.model_parallel)
        torch_dtype = _get_dtype(config.dtype, self._config)
        model_kwargs = {
            'device': 'cuda',
            "model_kwargs":{
                'torch_dtype': torch_dtype,
                'max_memory': max_memory,
            }}
        encode_kwargs = {'normalize_embeddings': True if config.similarity_fn == "cosine" else False}
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.pretrained,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            multi_process=True,
            cache_folder=env_config.cache_dir,
            # token=env_config.token,
            # trust_remote_code=config.trust_remote_code,
            # revision=(config.revision + (f"/{config.subfolder}" if config.subfolder else "")),
        )

        return embedding_model

    def _create_auto_tokenizer(
        self, config: TransformersModelConfig, env_config: EnvConfig
    ) -> transformers.PreTrainedTokenizer:
        return self._create_auto_tokenizer_with_name(
            model_name=config.pretrained,
            revision=config.revision,
            env_config=env_config,
            tokenizer_name=config.tokenizer,
            subfolder=config.subfolder,
            trust_remote_code=config.trust_remote_code,
        )

    def _create_auto_tokenizer_with_name(
        self,
        model_name: str,
        revision: str,
        env_config: EnvConfig,
        tokenizer_name: str = None,
        subfolder: str = None,
        trust_remote_code: bool = False,
    ) -> transformers.PreTrainedTokenizer:
        """
        Create a Hugging Face AutoTokenizer for language model.

        Args:
            pretrained (str): The identifier of the pretrained model to load.
            revision (str): The specific model version to load.
            subfolder (str): The subfolder within the model repository.
            tokenizer (str, optional): The identifier of the tokenizer to load. If not provided, the default tokenizer for the pretrained model will be used.
            cache_dir (str, optional): The directory to cache the downloaded models and tokens. Defaults to "/scratch".
            trust_remote_code (bool, optional): Whether to trust remote code execution during tokenization. Defaults to False.

        Returns:
            transformers.PreTrainedTokenizer: The created tokenizer.

        Raises:
            RecursionError: If an error occurs during tokenization, a fallback tokenizer with "<unk>" token will be created.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name if tokenizer_name is None else tokenizer_name,
                revision=revision + (f"/{subfolder}" if subfolder is not None else ""),
                cache_dir=env_config.cache_dir,
                token=env_config.token,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                truncation_side="left",
            )
        except RecursionError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name if tokenizer_name is None else tokenizer_name,
                revision=revision + (f"/{subfolder}" if subfolder is not None else ""),
                cache_dir=env_config.cache_dir,
                token=env_config.token,
                trust_remote_code=trust_remote_code,
                unk_token="<unk>",
                padding_side="left",
                truncation_side="left",
            )
        except FileNotFoundError:
            logger.warning(
                "Problem when lodading the tokenizer in the cache - discarding the provided cache path value."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name if tokenizer_name is None else tokenizer_name,
                revision=revision + (f"/{subfolder}" if subfolder is not None else ""),
                token=env_config.token,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                truncation_side="left",
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self._max_length
        logger.info("Tokenizer truncation and padding size set to the left side.")

        return tokenizer

    def _init_max_length(self, max_length) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.

        Args:
            max_length (Optional[int]): The maximum length of the input sequence. If not provided, it will be determined
                based on the model's configuration or tokenizer's model_max_length attribute.

        Returns:
            int: Max length to use depending on the available args and config
        """
        if max_length is not None:
            return int(max_length)
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")

        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)

        # Default max sequence length setting for when no `max_length` is provided
        # or no max length config setting is found in the model or tokenizer.
        return 512

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device


    def _split_documents(
        self,
        chunk_size: int,
        knowledge_base: List[LangchainDocument]
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

    def _build_knowledge_base(self) -> FAISS:
        """
        Build a knowledge base from the given documents.
        """
        ds = load_dataset(self.docs_name_or_path, split="train")
        knowledge_base = [
            LangchainDocument(
                page_content=doc["text"],
                metadata={"source": doc["source"]}) for doc in tqdm(ds)
        ]

        docs_processed = self._split_documents(
            self._max_length, knowledge_base)

        logger.info(f"Building FAISS knowledge base from {len(docs_processed)} documents.")
        vector_db = FAISS.from_documents(
            docs_processed,
            self.model,
            distance_strategy=distance_strategy_mapping[self.similarity_fn],
        )
        return vector_db

    def run_model(
        self,
        query: str,
        k: int = 5
    ) -> List[LangchainDocument]:
        """
        Retrieve documents from the knowledge base based on the query.
        """
        # query_vector = self.model.embed_query(query)
        retrieved_docs = self.vector_db.similarity_search(query=query, k=self.top_k)
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]

        context = "\nRelavent Documents:\n"
        context += "".join([
            f"Document {str(i)}:::\n" + doc
            for i, doc in enumerate(retrieved_docs_text)
        ])

        return context + "\n\n" + query
