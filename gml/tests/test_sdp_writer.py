"""Tests for orchestration/sdp/writer.py — SDPWriter."""
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestration.memory_store import JsonlMemoryStore
from orchestration.retriever.base import Retriever
from orchestration.sam.aal_record import AALRecord, AALTuple
from orchestration.sdp.entity_index import EntityIndex
from orchestration.sdp.writer import AUTH_CHUNK, AUTH_TUPLE, SDPWriter


class _FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.ingested = []

    async def search(self, embedded):
        return []

    async def get_top_matches(self, embedded, k=50):
        return []

    async def ingest(self, items):
        self.ingested.extend(items)


@pytest.fixture
def writer_stack():
    fd, path = tempfile.mkstemp(prefix="locomo-writer-test-", suffix=".jsonl")
    os.close(fd)  # Windows refuses to unlink while the fd is open.
    tmp = Path(path)
    store = JsonlMemoryStore(tmp)
    retriever = _FakeRetriever()
    idx = EntityIndex()
    yield SDPWriter(store=store, retriever=retriever, entity_index=idx), store, retriever, idx
    tmp.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_empty_record_writes_nothing(writer_stack):
    writer, store, retriever, idx = writer_stack
    items = await writer.write(AALRecord())
    assert items == []
    assert store.load_all() == []
    assert retriever.ingested == []


@pytest.mark.asyncio
async def test_writes_tuple_with_correct_metadata(writer_stack):
    writer, store, retriever, idx = writer_stack
    rec = AALRecord(
        tuples=[AALTuple("Caroline", "research", "agencies", time="Sun")],
        timestamp=datetime(2024, 5, 7, tzinfo=timezone.utc),
        session_id=2,
    )
    items = await writer.write(rec)
    assert len(items) == 1
    item = items[0]
    assert item.source == "aal-tuple"
    assert item.authority_score == AUTH_TUPLE
    assert item.entity == "Caroline"
    assert item.attribute == "research"
    assert item.value == "agencies"
    # Persistence check
    assert len(store.load_all()) == 1
    # Retriever ingested
    assert len(retriever.ingested) == 1


@pytest.mark.asyncio
async def test_writes_chunk_with_correct_metadata(writer_stack):
    writer, store, retriever, idx = writer_stack
    rec = AALRecord(
        chunk_summary="Caroline went to the support group on Sunday May 7",
        chunk_user="raw user msg",
        chunk_assistant="raw assistant reply",
    )
    items = await writer.write(rec)
    assert len(items) == 1
    item = items[0]
    assert item.source == "aal-chunk"
    assert item.authority_score == AUTH_CHUNK
    assert item.entity is None
    assert "Sunday" in item.content


@pytest.mark.asyncio
async def test_writes_both_tuples_and_chunk(writer_stack):
    writer, store, retriever, idx = writer_stack
    rec = AALRecord(
        tuples=[AALTuple("A", "go", "B"), AALTuple("C", "see", "D")],
        chunk_summary="A summary",
    )
    items = await writer.write(rec)
    assert len(items) == 3  # 2 tuples + 1 chunk
    sources = sorted(i.source for i in items)
    assert sources == ["aal-chunk", "aal-tuple", "aal-tuple"]


@pytest.mark.asyncio
async def test_entity_index_updated(writer_stack):
    writer, store, retriever, idx = writer_stack
    rec = AALRecord(
        tuples=[AALTuple("Caroline", "go", "park")],
        entities=["caroline", "park"],
    )
    await writer.write(rec)
    # Both via tuple extraction and via record.entities, "caroline" lands in index
    assert idx.lookup_query("Caroline?") != set()


@pytest.mark.asyncio
async def test_retriever_failure_doesnt_block_persistence(writer_stack):
    """If the retriever raises, store still has the records."""
    writer, store, _ret, _idx = writer_stack

    class _BadRet(_FakeRetriever):
        async def ingest(self, items):
            raise RuntimeError("retriever crashed")

    writer.retriever = _BadRet()
    rec = AALRecord(chunk_summary="x")
    items = await writer.write(rec)
    assert len(items) == 1
    # Disk still has it
    assert len(store.load_all()) == 1


@pytest.mark.asyncio
async def test_entity_index_failure_doesnt_block(writer_stack):
    writer, store, retriever, _idx = writer_stack

    class _BadIdx(EntityIndex):
        def add_many(self, records):
            raise RuntimeError("idx crashed")

    writer.entity_index = _BadIdx()
    rec = AALRecord(tuples=[AALTuple("X", "go", "Y")])
    items = await writer.write(rec)
    assert len(items) == 1
    assert len(retriever.ingested) == 1
