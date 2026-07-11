"""Unit tests for the sandboxed filesystem MCP server."""

import sys
from pathlib import Path

import pytest

# examples/ isn't a package; import the module by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import filesystem_server as fsmod  # noqa: E402


@pytest.fixture
def fs(tmp_path):
    return fsmod.FileSystem(str(tmp_path))


def test_write_read_roundtrip(fs):
    assert "Wrote" in fs.write_file("notes/todo.txt", "buy milk")
    assert fs.read_file("notes/todo.txt") == "buy milk"


def test_list_and_search(fs):
    fs.write_file("a.txt", "1")
    fs.write_file("sub/b.log", "2")
    listing = fs.list_directory(".")
    assert "[FILE] a.txt" in listing and "[DIR]  sub/" in listing
    assert fs.search_files("*.log") == "sub/b.log"


def test_info_and_move_and_delete(fs):
    fs.write_file("x.txt", "hi")
    info = fs.get_file_info("x.txt")
    assert '"type": "file"' in info and '"size": 2' in info
    fs.move_file("x.txt", "y.txt")
    assert fs.read_file("y.txt") == "hi"
    fs.delete_file("y.txt")
    with pytest.raises(FileNotFoundError):
        fs.read_file("y.txt")


def test_sandbox_blocks_escape(fs):
    for bad in ["../escape.txt", "../../etc/passwd", "/etc/passwd"]:
        with pytest.raises(fsmod.SandboxError):
            fs.read_file(bad)


def test_tools_call_via_handle(fs):
    handle = fsmod.handle
    # successful write
    resp = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "write_file", "arguments": {"path": "f.txt", "content": "yo"}}}, fs)
    assert resp["result"]["isError"] is False
    # sandbox escape surfaces as isError, not a crash
    resp = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "read_file", "arguments": {"path": "../../etc/passwd"}}}, fs)
    assert resp["result"]["isError"] is True
    assert "SandboxError" in resp["result"]["content"][0]["text"]
