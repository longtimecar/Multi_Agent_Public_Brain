"""RAG knowledge library: parsing, chunking, ingest, hybrid search."""

from __future__ import annotations

import base64

import pytest

from membrain import rag


def test_chunk_text_respects_boundaries_and_overlap():
    text = "# 标题一\n" + ("甲" * 300) + "\n\n## 标题二\n" + ("乙" * 300)
    chunks = rag.chunk_text(text, size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(len(c) <= 280 for c in chunks)
    assert "甲" in chunks[0] and "乙" in chunks[-1]


def test_extract_text_rejects_unknown_type(tmp_path):
    path = tmp_path / "x.exe"
    path.write_bytes(b"binary")
    with pytest.raises(ValueError):
        rag.extract_text(path)


def test_ingest_and_search_roundtrip(tmp_path):
    home = tmp_path / "memory"

    class FakeEmbedder:
        enabled = True

        def _vec(self, text):
            import numpy as np

            base = 1.0 if "茶" in text or "tea" in text else 0.0
            return np.array([base, 1.0 - base, 0.5], dtype="float32")

        def embed(self, texts):
            import numpy as np

            return np.stack([self._vec(t) for t in texts])

        embed_queries = embed

    body = "用户偏好：喜欢喝茶，不喜欢咖啡。\n\n项目笔记：membrain 是个人长期记忆系统。"
    result = rag.ingest_text(home=home, body=body, category="notes", title="测试文档",
                             embedder=FakeEmbedder(), llm=None)
    assert result["chunks"] >= 1
    assert result["embedded"] is True

    docs = rag.list_documents(home)
    assert docs and docs[0]["title"] == "测试文档"

    hits = rag.search(home, "喜欢喝茶", embedder=FakeEmbedder())
    assert hits and hits[0]["score"] > 0
    assert "茶" in hits[0]["text"]

    doc = rag.get_document(home, result["id"])
    assert doc["chunk_list"]

    assert rag.delete_document(home, result["id"]) is True
    assert rag.list_documents(home) == []


def test_ingest_upload_base64(tmp_path):
    home = tmp_path / "memory"
    payload = base64.b64encode("第一段内容。\n第二段内容。".encode("utf-8")).decode("ascii")
    result = rag.ingest_upload(home=home, filename="note.txt", data_b64=payload,
                               category="docs", embedder=None, llm=None)
    assert result["chunks"] >= 1
    assert rag.list_documents(home)
