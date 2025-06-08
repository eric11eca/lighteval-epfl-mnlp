## 🔍 FAISS Index Reuse for RAG Evaluation

This fork adds support for **reusing or saving FAISS indexes** during RAG evaluations in LightEval, dramatically improving speed and avoiding redundant recomputation.

### ✨ Why This Matters

By default, LightEval rebuilds the FAISS index **every run**, even if your dataset and embedding model are unchanged. With this update, you can:
- Load a previously computed FAISS index from disk
- Save the index the first time it's built
- Avoid recomputing embeddings and chunk processing unnecessarily

---

### 🔧 How to Use

To activate FAISS index reuse, modify your `rag_model.yaml` file as follows:

```yaml
rag_params:
    embedding_model: "thenlper/gte-small" 
    docs_name_or_path: "m-ric/huggingface_doc" 
    similarity_fn: cosine 
    top_k: 20
    num_chunks: 100000
    faiss_index_path: "./faiss_cache/my_index.index"
```

#### ✅ `faiss_index_path`
- **If this path exists**, the index will be loaded from disk using `FAISS.load_local(...)`.
- **If it doesn't exist**, the index will be built from `docs_name_or_path`, then saved to this path using `vector_db.save_local(...)`.

> Example:
```yaml
faiss_index_path: "indexes/wiki_stem_faiss_index"
```

This will create `wiki_stem_faiss_index/index.faiss` and `index.pkl`.

---

### 🛠️ What Was Changed

#### 1. `embed_model.py`
- Added a `faiss_index_path` parameter in `EmbeddingModelConfig`.
- Modified `_build_knowledge_base(...)` to check if the FAISS index should be loaded or built and saved.

#### 2. `main_accelerate.py`
- Extracts `faiss_index_path` from the `rag_params` config block and passes it to the model config.

---

### ✅ Benefits

- ⏱️ **Faster**: Skips re-embedding and chunking for static datasets
- 💾 **Cacheable**: Can be reused across multiple model evaluations
- 🧪 **Modular**: Keeps FAISS logic inside the embedding model where it belongs

---

### 💡 Tips

- You must ensure the embedding model and dataset used match the FAISS index — otherwise, results will be unreliable.
- The `faiss_index_path` directory will be created if it doesn't exist.

---

This addition makes RAG evaluation workflows faster and more reproducible.
