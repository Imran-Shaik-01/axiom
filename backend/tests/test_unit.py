import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import providers
from context_budget import trim_messages
from providers import fix_seam, strip_think
from tools import run_python
from tools_apps import pdf_tutor


class Msg:
    def __init__(self, role, content, images=None):
        self.role = role
        self.content = content
        self.images = images or []


def test_fix_seam_inserts_space():
    assert fix_seam("the", "elephant") == " elephant"


def test_fix_seam_keeps_space():
    assert fix_seam("the ", "elephant") == "elephant"


def test_fix_seam_keeps_punct():
    assert fix_seam("Hello,", " world") == " world"
    assert fix_seam("end.", "Next") == "Next"


def test_fix_seam_empty():
    assert fix_seam("", "Hi") == "Hi"
    assert fix_seam("Hi", "") == ""


def test_strip_think_complete():
    text = "<think>secret</think>\nHello"
    assert strip_think(text) == "Hello"


def test_strip_think_unfinished():
    text = "Hi\n<think>still thinking"
    assert "think" not in strip_think(text).lower()
    assert strip_think(text) == "Hi"


def test_strip_think_nested_noise():
    text = "<think>a</think>Answer<think>b</think>"
    assert strip_think(text) == "Answer"


def test_trim_keeps_newest():
    msgs = [Msg("user", "old " * 1000), Msg("assistant", "mid"), Msg("user", "newest question")]
    kept, dropped = trim_messages(msgs, budget=80)
    assert kept[-1].content == "newest question"
    assert dropped >= 1


def test_trim_starts_with_user():
    msgs = [Msg("user", "a" * 10), Msg("assistant", "b" * 10), Msg("user", "c" * 10)]
    kept, _ = trim_messages(msgs, budget=40)
    assert kept[0].role == "user"


def test_trim_clips_single_oversized_newest_message():
    newest = Msg("user", "question " * 20, images=["one", "two"])
    kept, dropped = trim_messages([newest], budget=20)
    assert dropped == 0
    assert kept[-1].content.startswith("question")
    assert len(kept[-1].content) + 2000 * len(kept[-1].images) <= 20


@pytest.mark.asyncio
async def test_python_tool_runs_bounded_snippets():
    assert await run_python("total = 0\nfor n in range(5):\n    total += n\nprint(total)") == "10"


@pytest.mark.asyncio
async def test_python_tool_rejects_imports_and_attribute_access():
    assert "unsupported statement: Import" in await run_python("import os")
    assert "unsupported syntax: Attribute" in await run_python("().__class__")


@pytest.mark.asyncio
async def test_python_tool_preserves_short_circuit_values_and_augmented_assignment():
    code = (
        'n = 0\n'
        'print(n != 0 and 10 / n)\n'
        'print("a" or "b")\n'
        'values = [1]\n'
        'values += [2]\n'
        'print(values)'
    )
    assert await run_python(code) == "False\na\n[1, 2]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression",
    [
        '"a" * 2000000000',
        '2000000000 * "a"',
        '"a" * 1000000000000000000',
        '1000000000000000000 * "a"',
    ],
)
async def test_python_tool_checks_string_multiplication_before_allocation(expression):
    result = await run_python(expression)
    assert "result exceeds the 10,000-item limit" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [[], ["<think>private reasoning</think>"]])
async def test_pdf_query_rewrite_falls_back_when_provider_output_is_empty(monkeypatch, tokens):
    async def fake_stream(*args, **kwargs):
        for token in tokens:
            yield "token", {"text": token}
        yield "done", {}

    monkeypatch.setattr(providers, "stream_with_failover", fake_stream)
    question = "What does the report conclude?"
    history = [pdf_tutor.HistoryMessage(role="user", content="Earlier question")]

    assert await pdf_tutor.rewrite_query(question, history) == question
