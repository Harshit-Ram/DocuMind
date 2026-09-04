"""Smoke-test the real functions extracted from app.py (no Streamlit runtime needed)."""
import ast

import faiss
import numpy as np

src = open("app.py", encoding="utf-8").read()
tree = ast.parse(src)
want = {"chunk_text", "build_faiss_index", "retrieve_relevant_chunks"}
code = "\n".join(
    ast.unparse(n)
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in want
)
g = {
    "faiss": faiss,
    "np": np,
    "numpy": np,
    "CHUNK_SIZE": 800,
    "CHUNK_OVERLAP": 150,
    "DEFAULT_TOP_K": 3,
}
exec(code, g)

chunks = g["chunk_text"]("Hello world. " * 200)
print("chunks:", len(chunks))
assert len(chunks) >= 2


class FakeEmb:
    def encode(self, x, **kwargs):
        rng = np.random.default_rng(0)
        vecs = rng.random((len(x), 8)).astype("float32")
        if kwargs.get("normalize_embeddings"):
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


idx = g["build_faiss_index"](chunks, FakeEmb())
print("index ntotal:", idx.ntotal)
assert idx.ntotal == len(chunks)

# Edge case that crashed the original: k larger than number of chunks
res = g["retrieve_relevant_chunks"]("hello", chunks, idx, FakeEmb(), k=99)
print("retrieved with k=99:", len(res))
assert len(res) == len(chunks), "k>n guard failed"

res2 = g["retrieve_relevant_chunks"]("hello", chunks, idx, FakeEmb(), k=2)
assert len(res2) == 2
print("RAG CORE LOGIC OK")
