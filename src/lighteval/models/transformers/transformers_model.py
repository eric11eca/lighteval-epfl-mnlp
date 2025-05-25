# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import logging
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    PretrainedConfig,
)
from transformers.generation.utils import GenerateOutput, GenerationConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from lighteval.data import (
    GenerativeTaskDataset,
    LoglikelihoodDataset,
    LoglikelihoodSingleTokenDataset,
    DPODataCollatorWithPadding
)
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import (
    Batch,
    RewardModelingResponse,
    GenerativeMultiturnResponse,
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.models.utils import _get_dtype, _get_model_sha, _simplify_name, batched
from lighteval.tasks.requests import (
    GreedyUntilMultiTurnRequest,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
)
from lighteval.utils.imports import (
    NO_AUTOGPTQ_ERROR_MSG,
    NO_BNB_ERROR_MSG,
    is_accelerate_available,
    is_autogptq_available,
    is_bnb_available,
)
from lighteval.utils.parallelism import find_executable_batch_size
from lighteval.utils.utils import EnvConfig, as_list, boolstring_to_bool


logger = logging.getLogger(__name__)


if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate.utils import calculate_maximum_sizes, convert_bytes, get_max_memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


@dataclass
class TransformersModelConfig:
    """
    Base configuration class for models.

    Attributes:
        pretrained (str):
            HuggingFace Hub model ID name or the path to a pre-trained
            model to load. This is effectively the `pretrained_model_name_or_path`
            argument of `from_pretrained` in the HuggingFace `transformers` API.
        accelerator (Accelerator): accelerator to use for model training.
        tokenizer (Optional[str]): HuggingFace Hub tokenizer ID that will be
            used for tokenization.
        multichoice_continuations_start_space (Optional[bool]): Whether to add a
            space at the start of each continuation in multichoice generation.
            For example, context: "What is the capital of France?" and choices: "Paris", "London".
            Will be tokenized as: "What is the capital of France? Paris" and "What is the capital of France? London".
            True adds a space, False strips a space, None does nothing
        pairwise_tokenization (bool): Whether to tokenize the context and continuation as separately or together.
        subfolder (Optional[str]): The subfolder within the model repository.
        revision (str): The revision of the model.
        batch_size (int): The batch size for model training.
        max_gen_toks (Optional[int]): The maximum number of tokens to generate.
        max_length (Optional[int]): The maximum length of the generated output.
        add_special_tokens (bool, optional, defaults to True): Whether to add special tokens to the input sequences.
           If `None`, the default value will be set to `True` for seq2seq models (e.g. T5) and
            `False` for causal models.
        model_parallel (bool, optional, defaults to False):
            True/False: force to use or not the `accelerate` library to load a large
            model across multiple devices.
            Default: None which corresponds to comparing the number of processes with
                the number of GPUs. If it's smaller => model-parallelism, else not.
        dtype (Union[str, torch.dtype], optional, defaults to None):):
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model's weights.
        device (Union[int, str]): device to use for model training.
        quantization_config (Optional[BitsAndBytesConfig]): quantization
            configuration for the model, manually provided to load a normally floating point
            model at a quantized precision. Needed for 4-bit and 8-bit precision.
        trust_remote_code (bool): Whether to trust remote code during model
            loading.
        generation_parameters (GenerationParameters): Range of parameters which will affect the generation.
        generation_config (GenerationConfig): GenerationConfig object (only passed during manual creation)

    Methods:
        __post_init__(): Performs post-initialization checks on the configuration.
        _init_configs(model_name, env_config): Initializes the model configuration.
        init_configs(env_config): Initializes the model configuration using the environment configuration.
        get_model_sha(): Retrieves the SHA of the model.

    """

    pretrained: str
    accelerator: "Accelerator" = None
    tokenizer: Optional[str] = None
    multichoice_continuations_start_space: Optional[bool] = None
    pairwise_tokenization: bool = False
    subfolder: Optional[str] = None
    revision: str = "main"
    batch_size: int = -1
    ref_free_norm: Optional[str] = "none"
    max_gen_toks: Optional[int] = 256
    max_length: Optional[int] = None
    add_special_tokens: bool = True
    model_parallel: Optional[bool] = None
    dtype: Optional[Union[str, torch.dtype]] = None
    device: Union[int, str] = "cuda"
    quantization_config: Optional[BitsAndBytesConfig] = None
    trust_remote_code: bool = False
    use_chat_template: bool = False
    compile: bool = False
    generation_parameters: GenerationParameters = None
    generation_config: GenerationConfig = None

    def __post_init__(self):
        # Making sure this parameter is a boolean
        self.multichoice_continuations_start_space = boolstring_to_bool(self.multichoice_continuations_start_space)

        if self.multichoice_continuations_start_space is not None:
            if self.multichoice_continuations_start_space:
                logger.info(
                    "You set `multichoice_continuations_start_space` to true. This will force multichoice continuations to use a starting space"
                )
            else:
                logger.info(
                    "You set `multichoice_continuations_start_space` to false. This will remove a leading space from multichoice continuations, if present."
                )

        self.model_parallel = boolstring_to_bool(self.model_parallel)
        self.compile = boolstring_to_bool(self.compile)

        if self.quantization_config is not None and not is_bnb_available():
            raise ImportError(NO_BNB_ERROR_MSG)

        if not isinstance(self.pretrained, str):
            raise ValueError("Pretrained model name must be passed as string.")
        if not isinstance(self.device, str):
            raise ValueError("Current device must be passed as string.")

        if self.generation_config and self.generation_parameters:
            raise ValueError(
                "Can't use both generation_config and generation_parameters argument. Pass the generation parameters to your generation config object"
            )

        if not self.generation_parameters and not self.generation_config:
            self.generation_parameters = GenerationParameters()

    def _init_configs(self, model_name: str, env_config: EnvConfig) -> PretrainedConfig:
        revision = self.revision
        if self.subfolder:
            revision = f"{self.revision}/{self.subfolder}"
        auto_config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
            cache_dir=env_config.cache_dir,
            token=env_config.token
        )

        # Gathering the model's automatic quantization config, if available
        try:
            model_auto_quantization_config = auto_config.quantization_config
            logger.info("An automatic quantization config was found in the model's config. Using it to load the model")
        except (AttributeError, KeyError):
            model_auto_quantization_config = None

        if model_auto_quantization_config is not None:
            if self.quantization_config is not None:
                # We don't load models quantized by default with a different user provided conf
                logger.warning("You manually requested quantization on a model already quantized! Using the model's automatic quantization config instead.")
                self.quantization_config = None

            # We add the quantization to the model params we store
            if model_auto_quantization_config["quant_method"] == "gptq":
                if not is_autogptq_available():
                    raise ImportError(NO_AUTOGPTQ_ERROR_MSG)
                auto_config.quantization_config["use_exllama"] = None
                self.quantization_config = GPTQConfig(**auto_config.quantization_config, disable_exllama=True)
            elif model_auto_quantization_config["quant_method"] == "bitsandbytes":
                if not is_bnb_available():
                    raise ImportError(NO_BNB_ERROR_MSG)
                self.quantization_config = BitsAndBytesConfig(**auto_config.quantization_config)

        return auto_config

    def init_configs(self, env_config: EnvConfig) -> PretrainedConfig:
        return self._init_configs(self.pretrained, env_config=env_config)

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.pretrained, revision=self.revision)


@dataclass
class BaseModelConfig(TransformersModelConfig):
    def __post_init__(self):
        super().__post_init__()

        warnings.warn(
            "BaseModelConfig is deprecated and will be removed. Use TransformersModelConfig instead",
            FutureWarning,
        )


class TransformersModel(LightevalModel):
    def __init__(
        self,
        env_config: EnvConfig,
        config: TransformersModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config.init_configs(env_config)
        self.accelerator = config.accelerator
        self._max_length = self._init_max_length(config.max_length)
        self.use_chat_template = config.use_chat_template

        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config, env_config)

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config, env_config)
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        # We are in DP (and launch the script with `accelerate launch`)
        if not config.model_parallel and not isinstance(config.quantization_config, BitsAndBytesConfig):
            logger.info(f"Using Data Parallelism, putting model on device {self._device}")
            self.model = self.model.to(self._device)
        if config.compile:
            try:
                logger.info("Compiling the model")
                self.model.model.compile()
            except AttributeError as e:
                logger.warning("Could not compile the model because: ", e)

        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = config.get_model_sha()

        self.precision = _get_dtype(config.dtype, config=self._config)
        if config.generation_config is None:
            self.generation_parameters = config.generation_parameters
            self.generation_config_dict = self.generation_parameters.to_transformers_dict()
        else:
            self.generation_config_dict = config.generation_config.to_dict()

        if is_accelerate_available():
            model_size, _ = calculate_maximum_sizes(self.model)
            model_size = convert_bytes(model_size)
        else:
            model_size = -1
        self.model_info = ModelInfo(
            model_name=self.model_name,
            model_sha=self.model_sha,
            model_dtype=self.precision,
            model_size=model_size,
        )

        self.label_pad_token_id = -100
        self.padding_value = self._tokenizer.pad_token_id
        self.truncation_mode = "keep_end"
        self.max_prompt_length = 128
        self.ref_free_norm = config.ref_free_norm

        self.pairwise_tokenization = config.pairwise_tokenization

    @classmethod
    def from_model(
        cls,
        model: Union[AutoModelForCausalLM, LightevalModel],
        env_config: EnvConfig,
        accelerator: "Accelerator" = None,
        tokenizer_name: str = None,  # custom tokenizer
        trust_remote_code: bool = False,
        use_chat_template: bool = False,
        add_special_tokens: bool = True,
        pairwise_tokenization: bool = False,
        multichoice_continuations_start_space: bool = None,
    ):
        # Slightly hackish way to test if the model is a AutoModelForCausalLM, since the instances don't
        # derive from this class explicitely
        assert isinstance(model, LightevalModel) or type(model).__name__ in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()

        if isinstance(model, LightevalModel):
            return model

        # Instanciate the object without using __init__
        self = cls.__new__(cls)
        self._config = model.config
        self._max_length = self._init_max_length(max_length=model.config.max_length)
        self._tokenizer = self._create_auto_tokenizer_with_name(
            model_name=model.name_or_path,
            revision=model.config._commit_hash,
            env_config=env_config,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
        )
        self.model_name = _simplify_name(model.name_or_path)
        self.model_sha = model.config._commit_hash

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

        self.accelerator = accelerator
        if accelerator is not None:
            self._device = accelerator.device
            self.model = self.accelerator.prepare(self.model.to(accelerator.device))
        else:
            self._device = "cpu"

        self.use_chat_template = use_chat_template
        self._add_special_tokens = add_special_tokens if add_special_tokens is not None else False
        self.pairwise_tokenization = pairwise_tokenization
        self.multichoice_continuations_start_space = multichoice_continuations_start_space

        self.precision = _get_dtype(model.dtype, config=self._config)

        if is_accelerate_available():
            model_size, _ = calculate_maximum_sizes(self.model)
            model_size = convert_bytes(model_size)
        else:
            model_size = -1
        self.model_info = ModelInfo(
            model_name=self.model_name,
            model_sha=self.model_sha,
            model_dtype=self.precision,
            model_size=model_size,
        )

        return self

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
            device_map = "auto" if self.accelerator else None
            logger.info(
                f"Model parallel was set to False, max memory set to {max_mem_this_process} and device map to {device_map}"
            )
        return model_parallel, max_mem_this_process, device_map

    def _create_auto_model(
        self, config: TransformersModelConfig, env_config: EnvConfig
    ) -> transformers.PreTrainedModel:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        config.model_parallel, max_memory, device_map = self.init_model_parallel(config.model_parallel)
        torch_dtype = _get_dtype(config.dtype, self._config)

        logger.info(
            f"max_memory {max_memory}, device_map {device_map}, torch_dtype {torch_dtype}"
        )

        pretrained_config = AutoConfig.from_pretrained(
            config.pretrained,
            revision=(config.revision + (f"/{config.subfolder}" if config.subfolder else "")),
            trust_remote_code=config.trust_remote_code,
            cache_dir=env_config.cache_dir,
            token=env_config.token,
        )

        kwargs = {}
        if "quantization_config" not in pretrained_config.to_dict():
            kwargs["quantization_config"] = config.quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained,
            revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            max_memory=max_memory,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=env_config.cache_dir,
            offload_folder=env_config.cache_dir,
            token=env_config.token,
            **kwargs,
        )

        def get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()

            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb

        model_size = get_model_size(model)

        if config.quantization_config is not None and "quant_method" in config.quantization_config.to_dict():
            logger.info(
                f"Model loaded with quantization method {config.quantization_config.quant_method} has model size {model_size:.2f} MB"
            )
        else:
            logger.info(f"Model loaded has model size {model_size:.2f} MB")

        return model

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
                "Problem when loading the tokenizer in the cache - discarding the provided cache path value."
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
        return 2048

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    @property
    def disable_tqdm(self) -> bool:
        disable_tqdm = False
        if self.accelerator:
            disable_tqdm = bool(not self.accelerator.is_main_process)
        return disable_tqdm

    def _check_continuations_start_space(self, continuation: str) -> str:
        """Some models tokenizer want a space at the beginning and other not. We update this if needed here.
        multichoice_continuations_start_space can be:
        - True (add a space if these isn't one)
        - False (remove a space if there is one)
        - None (Don't touch - default)
        Todo: find a way to add this back WITHOUT breaking compatibility with the harness
        """
        if self.multichoice_continuations_start_space is not None:
            if self.multichoice_continuations_start_space and continuation[0] != " ":
                continuation = " " + continuation
            if not self.multichoice_continuations_start_space and continuation[0] == " ":
                continuation = continuation.lstrip()
        return continuation

    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).logits

    def _get_batch_size(self, max_input_length: int, override_bs: int = 0, starting_batch_size: int = 512) -> int:
        if override_bs > 0:
            return override_bs
        logger.info(f"Detecting largest batch size with max_input_length={max_input_length}")

        @find_executable_batch_size(
            starting_batch_size=starting_batch_size
        )  # if OOM, then halves batch_size and tries again
        def forward_batch(batch_size):
            test_batch = torch.ones(
                (batch_size + int(0.1 * batch_size), max_input_length), device=self.device
            ).long()  # We add 10% for marging :)
            F.log_softmax(self._model_call(test_batch).float(), dim=-1).cpu()
            return batch_size

        batch_size = forward_batch()
        logger.info(f"Determined largest batch size: {batch_size}")
        return batch_size

    def greedy_until_multi_turn(  # noqa: C901
        self, requests: list[GreedyUntilMultiTurnRequest], override_bs: Optional[int] = None
    ) -> GenerativeMultiturnResponse:
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)["input_ids"]

        results = []

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=1)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch)

        if self.accelerator:
            dataloader = self.accelerator.prepare(dataloader)

        logger.warning("Running greedy multi turn generation, the batch size is set to 1 for this task.")

        for request_batch in tqdm(
            dataloader, desc="Greedy Multi Turn generation", position=1, leave=False, disable=self.disable_tqdm
        ):
            request = request_batch[0]
            # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
            if self.use_chat_template:
                stop_tokens = []
            else:
                stop_tokens = request.stop_sequence
            max_generated_tokens = request.generation_size
            context = request.context[0]
            max_context_size_allowed = self._max_length - max_generated_tokens

            model_inputs = self.tokenizer(
                context,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_context_size_allowed,
                add_special_tokens=self.add_special_tokens,
            ).to(self.device)

            stopping_criteria = transformers.StoppingCriteriaList(
                [
                    *[
                        MultiTokenEOSCriteria(
                            sequence, self.tokenizer, input_ids_shape=model_inputs["input_ids"].shape
                        )
                        for sequence in stop_tokens
                    ],
                ]
            )

            generation_config = self.generation_config_dict.copy()
            generation_config.update(
                {
                    "max_new_tokens": max_generated_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id
                    else self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": False,
                }
            )

            model_outputs: GenerateOutput = self.model.generate(
                **model_inputs, stopping_criteria=stopping_criteria, **generation_config
            )
            model_outputs = model_outputs.sequences[0, model_inputs["input_ids"].size(1) :]

            # We manage stop tokens in an extra step in case they were incorrectly detected earlier
            # (which can happen for multitoken stop sequences)
            decoded_generation = self.tokenizer.decode(model_outputs)  # should we skip_special_tokens=True here?
            for term in stop_tokens:
                decoded_generation = decoded_generation.split(term)[0]
            model_generations = [model_outputs]

            input_tokens = [model_inputs["input_ids"]]

            for i, multi_turn_context in enumerate(request.context[1:]):
                multi_turn_context = multi_turn_context.format(model_response=decoded_generation)

                model_inputs = self.tokenizer(
                    multi_turn_context,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_context_size_allowed,
                    add_special_tokens=self.add_special_tokens,
                ).to(self.device)

                stopping_criteria = transformers.StoppingCriteriaList(
                    [
                        *[
                            MultiTokenEOSCriteria(
                                sequence, self.tokenizer, input_ids_shape=model_inputs["input_ids"].shape
                            )
                            for sequence in stop_tokens
                        ],
                    ]
                )

                generation_config = self.generation_config_dict.copy()
                generation_config.update(
                    {
                        "max_new_tokens": max_generated_tokens,
                        "pad_token_id": self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id
                        else self.tokenizer.eos_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "do_sample": False,
                    }
                )

                model_outputs: GenerateOutput = self.model.generate(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    stopping_criteria=stopping_criteria,
                    **generation_config,
                )
                model_outputs = model_outputs.sequences[0, model_inputs["input_ids"].size(1) :]
                model_generations.append(model_outputs)
                input_tokens.append(model_inputs["input_ids"])

                decoded_generation = self.tokenizer.decode(model_outputs, skip_special_tokens=True)
                for term in stop_tokens:
                    decoded_generation = decoded_generation.split(term)[0]

            if self.accelerator:
                padding_size = max(gen.shape[0] for gen in model_generations)
                for i, gen in enumerate(model_generations):
                    model_generations[i] = F.pad(
                        gen, (0, padding_size - gen.shape[0]), value=self.tokenizer.pad_token_id
                    )
                model_generations = torch.stack(model_generations, dim=0)
                model_generations, lengths = self.pad_and_gather(model_generations, drop_last_samples=False)

            model_answers = []
            for generation, _ in zip(model_generations, lengths):
                generation = generation.cpu().tolist()
                decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
                model_answers.append(decoded)

            for answers in batched(model_answers, len(request.context)):
                results.append(
                    GenerativeMultiturnResponse(
                        result=answers,
                        input_tokens=input_tokens,
                        generated_tokens=[],
                        truncated_tokens_count=0,
                        padded_tokens_count=0,
                    )
                )

        return results

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        results = []

        for split_start, split_end in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            if dataset[0].generation_size is None:
                # No constraints on the generation size: max length allowed is the max model context
                max_context_continuation_size_allowed = self._max_length
            else:
                # Longest context in the current split is the first item (since we sort reversed)
                longest_context_continuation_size_in_split = (
                    len(dataset[0].tokenized_context) + dataset[0].generation_size
                )
                max_context_continuation_size_allowed = min(
                    longest_context_continuation_size_in_split, self._max_length
                )
            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_context_continuation_size_allowed,
                starting_batch_size=starting_batch_size,
            )
            # For next iteration, since the batch will be smaller, we'll test a bigger batch size
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
                if self.use_chat_template:
                    stop_tokens = []
                else:
                    # NOTE: we are assuming all items in a batch behave similarly (same
                    # stop_tokens and max_tokens genrated) which is not necessarily
                    # the case! Because of that we only use batch size of 1
                    stop_tokens = batch[0].stop_sequence

                max_new_tokens = batch[0].generation_size
                returns_logits = batch[0].use_logits
                num_samples = batch[0].num_samples
                do_sample = batch[0].do_sample

                context = [c.context for c in batch]

                # See doc https://huggingface.co/docs/transformers/v4.38.2/en/pad_truncation#padding-and-truncation
                # Will do left truncation and padding, as defined when creating the tokenizer
                tokenized = self.tokenizer(
                    context,
                    truncation="longest_first",  # we truncate to the model max length if needed
                    padding="max_length",  # we pad to the longest sequence
                    return_tensors="pt",
                    max_length=max_context_continuation_size_allowed,  # we always allow minimum one token of generation
                    add_special_tokens=self.add_special_tokens,
                ).to(self.device)

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                context_size = tokenized["input_ids"].shape[1]
                if context_size > self._max_length:
                    logger.warning(
                        f"The context size of your batch ({context_size}) is bigger than the maximum context size allowed by the model ({self._max_length}) for a task in"
                        + str({i.task_name for i in batch})
                        + ". This is likely to lead to some errors."  # noqa C401
                    )
                    # There will be truncation of at least one sample, maximum generation size will be one
                    max_new_tokens = 1
                else:  # We can't allow generation of more than max_length
                    if max_new_tokens is None:  # If generation size is not set, we go all the way
                        max_new_tokens = self._max_length - context_size
                    else:
                        max_new_tokens = min(self._max_length - context_size, max_new_tokens)
                        if max_new_tokens < 1:
                            max_new_tokens = 1

                prepared_batch = Batch(
                    input_ids=tokenized["input_ids"],
                    input_lengths=[len(item == 1) for item in tokenized["attention_mask"]],
                    input_mask=tokenized["attention_mask"],
                    truncated=[max(len(c) - tokenized["input_ids"].shape[1], 0) for c in context],
                    padded=[sum(mask == 0) for mask in tokenized["attention_mask"]],
                )

                cur_reponses = self._generate(
                    batch=prepared_batch,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=returns_logits,
                    num_samples=num_samples,
                    do_sample=do_sample,
                )
                results.extend(cur_reponses)

        return dataset.get_original_order(results)

    def _generate(
        self,
        batch: Batch,
        max_new_tokens: int,
        stop_tokens: list[str],
        returns_logits: Optional[bool] = False,
        num_samples: Optional[int] = 1,
        do_sample: Optional[bool] = False,
    ) -> list[GenerativeResponse]:
        """Contains the actual logic of the generation.
        First computes the stop sequences, then generates the predictions, then converts the outputs to GenerativeResponse.
        """
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop_sequences=stop_tokens, batch=batch)
        batch_size, _ = batch.input_ids.shape

        generation_config = self.generation_config_dict.copy()
        generation_config.update(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            num_return_sequences=num_samples,
            output_logits=returns_logits,
            renormalize_logits=True,
        )

        # Compute model generation
        outputs: GenerateOutput = self.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.input_mask,
            stopping_criteria=stopping_criteria,
            **generation_config,
        )
        generations = outputs.sequences[:, batch.input_ids.size(1) :]
        generations = torch.reshape(generations, (batch_size, num_samples, -1))
        generations, len_gens = self.pad_and_gather(generations, num_samples=num_samples)
        batch.input_ids, len_ids = self.pad_and_gather(batch.input_ids)

        logits, len_logits = None, None
        if returns_logits:
            logits, len_logits = self.pad_and_gather(outputs.logits)
            logits = logits.cpu().numpy()

        # We gather remaining info
        batch.truncated = torch.tensor(batch.truncated, device=self.device)
        if self.accelerator:
            batch.truncated = self.accelerator.gather_for_metrics(batch.truncated)
        batch.padded = torch.tensor(batch.padded, device=self.device)
        if self.accelerator:
            batch.padded = self.accelerator.gather_for_metrics(batch.padded)

        # We convert to GenerativeResponse outputs
        all_responses = []
        for ix, (batched_generations, batched_input, trunc, padded) in enumerate(
            zip(generations, batch.input_ids, batch.truncated, batch.padded)
        ):
            result_generations = []
            decoded_generations = []
            # Ensure the generated responses do not contain the stop sequences.
            for generation in batched_generations:
                generation = generation[: len_gens[ix]]
                result_generations.append(generation)
                decoded_generation = self.tok_decode([generation])[0]

                for term in stop_tokens:
                    decoded_generation = decoded_generation.split(term)[0]

                decoded_generations.append(decoded_generation)

            cur_response = GenerativeResponse(
                result=decoded_generations,
                logits=logits[ix][: len_logits[ix]] if returns_logits else None,
                generated_tokens=result_generations,
                input_tokens=batched_input[: len_ids[ix]],
                truncated_tokens_count=trunc.cpu().item(),
                padded_tokens_count=padded.cpu().item(),
            )
            all_responses.append(cur_response)

        return all_responses

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (list[Tuple[str, dict]]): _description_

        Returns:
            list[Tuple[float, bool]]: _description_
        """
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )

        return self._loglikelihood_tokens(requests, override_bs=override_bs)

    def loglikelihood_rolling(
        self,
        requests: list[LoglikelihoodRollingRequest],
        override_bs=None,
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""

        for request in requests:  # tuple of one elem
            request.tokenized_context = [self.tokenizer.eos_token_id]  # Fake context
            request.tokenized_continuation = self.tok_encode(request.context)

        results = self._loglikelihood_tokens(
            requests,
            override_bs=override_bs,
            return_bool_score=False,
            rolling=True,
        )
        return results

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: int = -1,
        return_bool_score: bool = True,
        rolling: bool = False,
    ) -> list[LoglikelihoodResponse]:
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        res = []

        for split_start, split_end in tqdm(dataset.splits_start_end_iterator()):
            context_enc = dataset[0].tokenized_context
            continuation_enc = dataset[0].tokenized_continuation
            if rolling:  # we take all the sequence in rolling mode
                max_context_continuation_size_allowed = len(context_enc + continuation_enc)
            else:  # in normal mode, we left cut the context if needed
                max_context_continuation_size_allowed = len(
                    (context_enc + continuation_enc)[-(self._max_length + 1) :][:-1]
                )

            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_context_continuation_size_allowed,
                starting_batch_size=starting_batch_size,
            )
            starting_batch_size = batch_size * 2

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                prepared_batch = self.prepare_batch_logprob(
                    batch,
                    padding_length=max_context_continuation_size_allowed,
                    max_context=max_context_continuation_size_allowed,
                )

                # torch.cuda.reset_peak_memory_stats()

                model_output = self._model_call(prepared_batch.input_ids)
                logits = F.log_softmax(model_output, dim=-1)  # [batch, padding_length, vocab]

                logits_sum = []
                max_equals = []
                batch_cont_tokens = []
                for cur_request, cur_logits, inplen in zip(batch, logits, prepared_batch.input_lengths):
                    cont_toks = torch.tensor(cur_request.tokenized_continuation, dtype=torch.long, device=self.device)
                    contlen = cont_toks.shape[0]
                    # We only look at the continuation tokens
                    if contlen > inplen:
                        # Continuation is longer than the input size, we are in rolling mode (only continuation)
                        cur_logits = cur_logits.unsqueeze(0).to(self.device)  # [1, seq, vocab]
                        cont_toks = cont_toks[:inplen].unsqueeze(0).to(self.device)  # [1, seq]
                    else:
                        cur_logits = (
                            cur_logits[inplen - contlen : inplen].unsqueeze(0).to(self.device)
                        )  # [1, seq, voc]
                        cont_toks = cont_toks.unsqueeze(0).to(self.device)  # [1, seq]

                    # Check if per-token argmax is exactly equal to continuation
                    greedy_tokens = cur_logits.argmax(dim=-1).to(self.device)
                    # Sometimes the continuation is longer than allowed by the model, we only look at the first tokens
                    max_equal = (greedy_tokens == cont_toks).all().squeeze(0).to(self.device)

                    # Obtain log-probs at the corresponding continuation token indices
                    cur_logits = torch.gather(cur_logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    logits_sum.append(cur_logits.sum())
                    max_equals.append(max_equal)
                    batch_cont_tokens.append(cont_toks)

                # Sync all
                # Need reshaping before gather
                batched_inputs, len_inputs = self.pad_and_gather(prepared_batch.input_ids)
                max_cont_tokens_length = max(len(c[0]) for c in batch_cont_tokens)
                batch_cont_tokens = torch.cat(
                    [
                        F.pad(c, (0, max_cont_tokens_length - c.shape[1], 0, 0), value=self.tokenizer.pad_token_id)
                        for c in batch_cont_tokens
                    ],
                    dim=0,
                )
                batch_cont_tokens, len_tokens = self.pad_and_gather(batch_cont_tokens)
                # Can be gathered as such
                logits = torch.tensor(logits_sum, device=self.device)
                max_equal = torch.tensor(max_equals, device=self.device)
                batch_truncated = torch.tensor(prepared_batch.truncated, device=self.device)
                batch_padded = torch.tensor(prepared_batch.padded, device=self.device)
                if self.accelerator:
                    logits = self.accelerator.gather_for_metrics(logits)
                    max_equal = self.accelerator.gather_for_metrics(max_equal)
                    batch_truncated = self.accelerator.gather_for_metrics(batch_truncated)
                    batch_padded = self.accelerator.gather_for_metrics(batch_padded)

                for ix, (logit, cont_tokens, maxe, batched_input, trunc, padded) in enumerate(
                    zip(logits, batch_cont_tokens, max_equal, batched_inputs, batch_truncated, batch_padded)
                ):
                    answer = LoglikelihoodResponse(
                        # todo: we might want to store the logits unsummed
                        result=(float(logit.sum()), bool(maxe)) if return_bool_score else float(logit.sum()),
                        input_tokens=batched_input[: len_inputs[ix]].cpu().tolist(),
                        generated_tokens=cont_tokens[: len_tokens[ix]].cpu().tolist(),
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    res.append(answer)

                # print(f"Peak memory during inference: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

                # Clean up GPUs
                del model_output
                del logits
                del batched_inputs
                del batch_truncated
                del batch_padded

        return dataset.get_original_order(res)

    def prepare_batch_logprob(
        self, batch: list[Request], padding_length: int, max_context: Optional[int] = None, single_token: bool = False
    ):
        """Tokenize a batch of inputs and return also the length, truncations and padding.
        This step is done manually since we tokenize log probability inputs together with their continuation,
        to manage possible extra spaces added at the start by tokenizers, see tok_encode_pair.
        """
        if single_token:
            inputs = [request.tokenized_context for request in batch]
        else:
            inputs = [
                request.tokenized_context + request.tokenized_continuation[:-1] for request in batch
            ]  # The last token (an eos) doesn't need to be given to the model

        input_tokens = []
        attention_masks = []
        input_lengths = []
        truncated = []
        padded = []

        if max_context is None:
            logger.warning("max_context is None, using max_length")
            max_context = self._max_length

        # Each sample is concatenated and cut to length or padded to max_length
        for orig_tokens in inputs:
            truncated.append(max(len(orig_tokens) - max_context, 0))

            # Truncate from the left if needed to fit in the model's context
            tokens = torch.tensor((orig_tokens)[-max_context:], dtype=torch.long).to(self.device)
            sequence_len = tokens.shape[0]

            # We add padding, if needed
            padding_length = padding_length if padding_length is not None else sequence_len

            if padding_length - sequence_len < 0:
                logger.warning(f"Padding length {padding_length} is smaller than input length {sequence_len}")
                raise ValueError("Negative padding")

            padded.append(padding_length - sequence_len)
            # Right padding, since we ignore these logprobs in the end
            tokens = F.pad(tokens, (0, padding_length - sequence_len), value=self.tokenizer.pad_token_id)

            # We create the attention mask to ignore padding
            mask = tokens == self.tokenizer.pad_token_id
            attention_masks.append(mask)

            input_tokens.append(tokens.unsqueeze(0))  # [1, padding_length]
            input_lengths.append(sequence_len)

        batched_inputs = torch.cat(input_tokens, dim=0)  # [batch, padding_length]
        attention_masks = torch.cat(attention_masks, dim=0)

        return Batch(
            input_ids=batched_inputs,
            input_mask=attention_masks,
            input_lengths=input_lengths,
            truncated=truncated,
            padded=padded,
        )

    def pad_and_gather(
        self, output_tensor: torch.Tensor, drop_last_samples: bool = True, num_samples: int = None
    ) -> torch.Tensor:
        """
        Pads the `output_tensor` to the maximum length and gathers the lengths across processes.

        Args:
            output_tensor (torch.Tensor): The output tensor to be padded.
            drop_last_samples (bool, optional): Whether to drop the last samples during gathering.
            Last samples are dropped when the number of samples is not divisible by the number of processes.
                Defaults to True.

        Returns:
            torch.Tensor: The padded output tensor and the gathered length tensor.
        """
        # Create a tensor of size batch_size, [output_length] * batch_size, for each process
        # output_tensor can be of size: batch_size * num_samples * length_item or just batch_size * length_item
        length_tensor = torch.tensor([output_tensor.shape[-1]] * output_tensor.shape[0], device=self.device)
        if self.accelerator is not None:
            # Gather all the lengths, we end up with a tensor of size num_processes [output_length_1, output_length_2, ...]
            length_tensor = self.accelerator.gather(length_tensor)
        # We pad the output_tensor to the max length
        max_length = length_tensor.max().item()
        padding = (
            (0, max_length - output_tensor.shape[-1], 0, 0, 0, 0)
            if num_samples is not None
            else (0, max_length - output_tensor.shape[-1], 0, 0)
        )
        output_tensor = F.pad(output_tensor, padding, value=self.tokenizer.pad_token_id)
        if self.accelerator:
            if drop_last_samples:
                output_tensor = self.accelerator.gather_for_metrics(output_tensor)
            else:
                output_tensor = self.accelerator.gather(output_tensor)
        return output_tensor, length_tensor

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Args:
            requests (list[Tuple[str, dict]]): _description_

        Returns:
            list[Tuple[float, bool]]: _description_
        """
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
            else:
                request.tokenized_context = self.tok_encode(request.context)

            # Some models tokenizer want a space at the beginning and other not
            continuations = [self._check_continuations_start_space(c) for c in request.choices]

            # We must not accidentally prepend a continuation with a start of sentence token.
            continuations_enc = [self.tok_encode(c, add_special_tokens=False) for c in continuations]
            if any(len(c) > 1 for c in continuations_enc):
                raise ValueError(
                    f"Trying to do single token multiple choice but one choice has several tokens: {continuations_enc}. "
                    "If the additional pre-token is a space, try to set `multichoice_continuations_start_space=False` in the model parameters "
                )
            request.tokenized_continuation = continuations_enc

        return self._loglikelihood_single_token(requests, override_bs=override_bs)

    def _loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: int = -1
    ) -> list[LoglikelihoodSingleTokenResponse]:
        dataset = LoglikelihoodSingleTokenDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        starting_batch_size = STARTING_BATCH_SIZE
        res = []

        for split_start, split_end in tqdm(dataset.splits_start_end_iterator()):
            context_enc = dataset[0].tokenized_context
            max_context = len(context_enc[-self._max_length :])
            batch_size = self._get_batch_size(override_bs=override_bs, max_input_length=max_context)

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)
            if self.accelerator is not None:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm, position=1):
                prepared_batch = self.prepare_batch_logprob(
                    batch, padding_length=max_context, max_context=max_context, single_token=True
                )

                out = self._model_call(prepared_batch.input_ids)  # [batch, padding_length, vocab]
                out = F.log_softmax(out, dim=-1)  # we do a softmax over the options, no the vocab

                batch_probs = []
                batch_cont_tokens = []
                for cur_request, logits, inplen in zip(batch, out, prepared_batch.input_lengths):
                    # Get the last token
                    logits = logits[inplen - 1]  # [vocab]

                    cont_toks = torch.tensor(
                        cur_request.tokenized_continuation, dtype=torch.long, device=self.device
                    ).squeeze(-1)  # [num_choices]

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    probs = torch.gather(logits, dim=0, index=cont_toks)  # [num_choices]

                    # Answer: (log prob, is-exact-match)
                    # probs = torch.nn.functional.softmax(logits.float(), dim=0)  # [num_choices]
                    batch_probs.append(probs)
                    batch_cont_tokens.append(cont_toks)

                # Sync all
                # Need reshape before gather
                batched_inputs, len_inputs = self.pad_and_gather(prepared_batch.input_ids)
                # We sometimes have different tasks with a different number of choices.
                # Padding to -10000 makes sure that we won't reach index problems later as all log probs will be smaller than that
                batch_probs = pad_sequence(batch_probs, batch_first=True, padding_value=-65504.0)
                batch_probs, len_probs = self.pad_and_gather(batch_probs)
                batch_cont_tokens = pad_sequence(batch_cont_tokens, batch_first=True, padding_value=-65504.0)
                batch_cont_tokens, len_cont = self.pad_and_gather(batch_cont_tokens)

                # No reshape
                batch_truncated = torch.tensor(prepared_batch.truncated, device=self.device)
                batch_padded = torch.tensor(prepared_batch.padded, device=self.device)
                if self.accelerator:
                    batch_truncated = self.accelerator.gather_for_metrics(batch_truncated)
                    batch_padded = self.accelerator.gather_for_metrics(batch_padded)

                for ix, (probs, cont_tokens, batched_input, trunc, padded) in enumerate(
                    zip(batch_probs, batch_cont_tokens, batched_inputs, batch_truncated, batch_padded)
                ):
                    answer = LoglikelihoodSingleTokenResponse(
                        result=probs[: len_probs[ix]].detach().cpu().tolist(),
                        input_tokens=batched_input[: len_inputs[ix]].cpu().tolist(),
                        generated_tokens=cont_tokens[: len_cont[ix]].cpu().tolist(),
                        truncated_tokens_count=trunc.cpu().item(),
                        padded_tokens_count=padded.cpu().item(),
                    )
                    res.append(answer)

                # Clean up GPUs
                del out
                del batch_probs
                del batched_inputs
                del batch_truncated
                del batch_padded

        return dataset.get_original_order(res)

    # Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L531
    def pad_to_length(self, tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [
                    tensor,
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def _prepare_dialogue_from_tokenizer(
        self,
        example: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._tokenizer.chat_template is None:
            return example
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    messages.append({"role": "user", "content": p})
                else:
                    messages.append({"role": "assistant", "content": p})
            # assert that the last message before this is user
            assert messages[-1]["role"] == "user"

            # required for DPO code only, otherwise discarded
            temp_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            # end with chosen/rejected
            messages.append({"role": "assistant", "content": example["chosen"]})
            example["text_chosen"] = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages[-1] = {"role": "assistant", "content": example["rejected"]}
            example["text_rejected"] = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example["prompt"] = temp_prompt
        # single turn
        else:
            # needed for DPO
            messages = [
                {"role": "user", "content": example["prompt"]},
            ]
            temp_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            example["text_chosen"] = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            example["text_rejected"] = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example["prompt"] = temp_prompt
        return example

    def tokenize_row(self, request) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        # example = self._prepare_dialogue_from_tokenizer({
        #     "prompt": request.context,
        #     "chosen": request.chosen_continuation,
        #     "rejected": request.rejected_continuation,
        # })

        # prompt = example["prompt"]
        # chosen = example["chosen"]
        # rejected = example["rejected"]

        prompt = request.context
        chosen = request.chosen_continuation
        rejected = request.rejected_continuation

        # remove prompt tokens from chosen + rejected, otherwise they repeat
        # see issue 140 https://github.com/allenai/reward-bench/issues/140
        # should impact results only slightly thanks to per token dpo reward math!
        chosen = chosen.replace(prompt, "")
        rejected = rejected.replace(prompt, "")

        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337
        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
        rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self._max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self._max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self._max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
                which are tensors of shape (batch_size, sequence_length).
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = self.pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        self.pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        return concatenated_batch

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        norm_log_prob: bool = False,
        label_pad_token_id: int = -100
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of
                label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token.
                Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            norm_log_prob: If True, return the normalized log probability per (non-masked) token.
                Note, only one of average_log_prob and norm_log_prob can be True.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities
                of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        elif norm_log_prob:
            return -torch.norm((per_token_logps * loss_mask), p=2, dim=-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, batch: Dict[str, Union[List, torch.LongTensor]], ref_free: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        len_chosen = batch["chosen_labels"].shape[0]
        concatenated_batch = self.concatenated_inputs(
            batch,
            padding_value=self.tokenizer.pad_token_id,
            device=self.device,
        )

        all_logits = self.model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"]
        ).logits

        if self.ref_free_norm == "norm":
            average_log_prob = False
            norm_log_prob = True
        elif self.ref_free_norm == "avg":
            average_log_prob = True
            norm_log_prob = False
        elif self.ref_free_norm == "sum":
            average_log_prob = False
            norm_log_prob = False
        # handles when reference model exists
        elif self.ref_free_norm == "none":
            average_log_prob = False
            norm_log_prob = False
        else:
            raise ValueError(f"Unknown ref_free_norm: {self.ref_free_norm}! Must be one of [norm, avg, sum, none]")

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob,
            norm_log_prob=norm_log_prob,
            label_pad_token_id=self.label_pad_token_id
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def dpo_inference(self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None) -> list[RewardModelingResponse]:
        for request in requests:
            batch = self.tokenize_row(request)
            request.chosen_input_ids = batch["chosen_input_ids"]
            request.chosen_attention_mask = batch["chosen_attention_mask"]
            request.chosen_labels = batch["chosen_labels"]
            request.rejected_input_ids = batch["rejected_input_ids"]
            request.rejected_attention_mask = batch["rejected_attention_mask"]
            request.rejected_labels = batch["rejected_labels"]
            request.tokenized_context = batch["prompt_input_ids"]
            request.prompt_input_ids = batch["prompt_input_ids"]
            request.prompt_attention_mask = batch["prompt_attention_mask"]

        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        res = []

        for split_start, split_end in tqdm(dataset.splits_start_end_iterator()):
            logger.info(f"Evaluating split {split_start} to {split_end}")
            context_enc_chosen = dataset[0].chosen_input_ids
            context_enc_rejected = dataset[0].chosen_attention_mask

            max_context = len(context_enc_chosen) + len(context_enc_rejected)

            batch_size = self._get_batch_size(
                override_bs=override_bs,
                max_input_length=max_context,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=DPODataCollatorWithPadding(
                    pad_token_id=self._tokenizer.pad_token_id,
                    label_pad_token_id=self.label_pad_token_id
                ),
                drop_last=False, shuffle=False)
            if self.accelerator:
                dataloader = self.accelerator.prepare(dataloader)

            for batch in tqdm(dataloader, disable=self.disable_tqdm, position=1):
                # Get the reference model log probabilities
                ref_chosen_logps = batch.get("reference_chosen_logps", None)
                ref_rejected_logps = batch.get("reference_rejected_logps", None)

                # Get the log probabilities
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    _,  # policy_chosen_logits,
                    _,  # policy_rejected_logits,
                ) = self.concatenated_forward(
                    batch, ref_free=ref_chosen_logps is None)

                if self.ref_free_norm == "none":
                    if ref_chosen_logps is None:
                        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    else:
                        ref_chosen_logps = ref_chosen_logps.to(self.device)
                    if ref_rejected_logps is None:
                        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
                    else:
                        ref_rejected_logps = ref_rejected_logps.to(self.device)

                    # Compute the log ratios as rewards
                    rewards_chosen = policy_chosen_logps.detach() - ref_chosen_logps.detach()
                    rewards_rejected = policy_rejected_logps.detach() - ref_rejected_logps.detach()
                else:
                    # Compute the log ratios as rewards without reference model
                    rewards_chosen = policy_chosen_logps.detach()
                    rewards_rejected = policy_rejected_logps.detach()

                # convert to float for bfloat16 case
                scores_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                scores_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

                # Store the results
                for ix, (chosen_score, rejected_score) in enumerate(zip(
                    scores_chosen_batch, scores_rejected_batch)):
                    res.append(
                        RewardModelingResponse(
                            result=(chosen_score, rejected_score),
                            policy_chosen_logps=policy_chosen_logps[ix].float().cpu().numpy().tolist(),
                            policy_rejected_logps=policy_rejected_logps[ix].float().cpu().numpy().tolist()
                        )
                    )

                # Clean up GPUs
                del policy_chosen_logps
                del policy_rejected_logps
                del rewards_chosen
                del rewards_rejected

        return dataset.get_original_order(res)

class BaseModel(TransformersModel):
    def __post_init__(self):
        super().__post_init__()

        warnings.warn(
            "Careful, the BaseModel name is deprecated and will be removed, you should use TransformersModel instead!",
            FutureWarning,
        )

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch: Batch = None,
        input_ids_shape: Tuple[int, int] = None,
    ):
        if batch is not None:
            initial_decoder_input_length = batch.input_ids.shape[1]
            batch_size = batch.input_ids.shape[0]
        else:
            initial_decoder_input_length = input_ids_shape[1]
            batch_size = input_ids_shape[0]

        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: list[str],
    batch: Batch,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[MultiTokenEOSCriteria(sequence, tokenizer, batch) for sequence in stop_sequences],
        ]
    )
