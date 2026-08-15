from doc_agent.contracts import Chunk
from doc_agent.index import store
from doc_agent.index.chunk import script_profile
from doc_agent.index.embedding_models import encode_texts
from doc_agent.retrieval.retriever import Retriever


def test_hash_retrieval_returns_exact_drug(tmp_path, monkeypatch):
    cfg = {
        "paths": {"index_dir": str(tmp_path)},
        "embed": {"backend": "hash", "dim": 512, "model": "hash", "batch_size": 8},
        "retrieve": {"k": 1},
    }
    chunks = [
        Chunk(id="a", doc_id="a.jpg", text="Tab. Phoscon 210mg", page_ids=["a"]),
        Chunk(id="b", doc_id="b.jpg", text="Tab. Napa 500mg", page_ids=["b"]),
    ]
    store.build(chunks, encode_texts([chunk.text for chunk in chunks], cfg, "passage"), cfg)
    assert Retriever(cfg).retrieve("Phoscon 210mg", 1)[0].doc_id == "a.jpg"


def test_code_switch_script_profile():
    assert script_profile("Tab. ওষুধ") == {"latin", "bengali"}
