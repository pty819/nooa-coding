"""Tests for the nooa-memory backed learning system."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa import Agent
from nooa.unifiedllm import FakeLLMClient, LLMResponse
from nooa_memory import MemoryConfig, MemoryManager

from nooa_coding.learning import Lesson, LessonExtractor, LessonStore


@pytest.fixture()
def memory_manager(tmp_path: Path) -> MemoryManager:
    """Create a real MemoryManager with HashingEmbedder backed by tmp SQLite."""
    llm = FakeLLMClient(scripted_responses=[])

    class _TestAgent(Agent, llm=llm):
        pass

    agent = _TestAgent()
    config = MemoryConfig(enabled=True, path=str(tmp_path / "test_memory.sqlite"))
    return MemoryManager(agent, config=config)


@pytest.fixture()
def store(memory_manager: MemoryManager) -> LessonStore:
    return LessonStore(memory_manager)


class TestLessonStore:
    def test_add_and_recall(self, store: LessonStore):
        lesson = Lesson(
            category="bug",
            title="Always check return codes",
            content="Shell commands must check returncode before proceeding.",
            context="CI pipeline failed silently",
        )
        lesson_id = store.add(lesson)
        assert lesson_id != ""

        results = store.recall("shell return code check")
        assert len(results) >= 1
        assert any("return" in r.content.lower() for r in results)

    def test_recall_filters_to_lessons(self, store: LessonStore, memory_manager: MemoryManager):
        """Non-lesson memories should not appear in recall results."""
        from nooa_memory.schema import MemoryType

        memory_manager.remember(
            "User prefers dark mode", type=MemoryType.INFO, title="UI Pref"
        )
        store.add(Lesson(category="tool", title="Use ruff", content="Run ruff before commit."))

        results = store.recall("ruff lint")
        assert any("ruff" in r.content.lower() for r in results)

    def test_recall_by_category(self, store: LessonStore):
        store.add(Lesson(category="bug", title="Bug lesson", content="Check nulls."))
        store.add(Lesson(category="workflow", title="WF lesson", content="Use branches."))

        bugs = store.recall("check", category="bug")
        assert all(r.category == "bug" for r in bugs)

    def test_recent(self, store: LessonStore):
        store.add(Lesson(category="general", title="First", content="First lesson."))
        store.add(Lesson(category="general", title="Second", content="Second lesson."))

        recent = store.recent(limit=5)
        assert len(recent) == 2
        titles = {r.title for r in recent}
        assert "First" in titles
        assert "Second" in titles

    def test_stats(self, store: LessonStore):
        store.add(Lesson(category="bug", title="B1", content="Bug one."))
        store.add(Lesson(category="bug", title="B2", content="Bug two."))
        store.add(Lesson(category="tool", title="T1", content="Tool one."))

        stats = store.stats()
        assert stats["total"] == 3
        assert stats["by_category"]["bug"] == 2
        assert stats["by_category"]["tool"] == 1

    def test_delete(self, store: LessonStore):
        lesson_id = store.add(
            Lesson(category="general", title="Deletable", content="To be deleted.")
        )
        assert store.delete(lesson_id) is True
        results = store.recall("deleted")
        assert all(r.lesson_id != lesson_id for r in results)

    def test_dedup_reinforces(self, store: LessonStore):
        """Adding the same lesson twice should reinforce, not duplicate."""
        lesson = Lesson(
            category="pattern",
            title="Use type hints",
            content="Always add type hints to function signatures.",
        )
        id1 = store.add(lesson)
        id2 = store.add(lesson)
        assert id1 == id2
        stats = store.stats()
        assert stats["total"] == 1


class TestLessonExtractor:
    @pytest.fixture()
    def extractor_llm(self) -> FakeLLMClient:
        response = LLMResponse(
            raw_response=None,
            content=(
                "CATEGORY: bug\n"
                "TITLE: Check file existence before read\n"
                "LESSON: Always verify the file exists before attempting to read it."
            ),
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": ""},
        )
        return FakeLLMClient(scripted_responses=[response])

    async def test_extract_from_failure(self, store: LessonStore, extractor_llm: FakeLLMClient):
        extractor = LessonExtractor(store, extractor_llm)
        lesson = await extractor.extract_from_failure(
            task="Read config file",
            error="FileNotFoundError: /etc/app/config.yaml not found",
            session_id="test-session",
        )
        assert lesson is not None
        assert lesson.category == "bug"
        assert "file" in lesson.title.lower()
        results = store.recall("file existence")
        assert len(results) >= 1

    async def test_extract_skips_short_error(self, store: LessonStore, extractor_llm: FakeLLMClient):
        extractor = LessonExtractor(store, extractor_llm)
        lesson = await extractor.extract_from_failure(task="x", error="short")
        assert lesson is None

    def test_recall_relevant(self, store: LessonStore, extractor_llm: FakeLLMClient):
        store.add(
            Lesson(
                category="bug",
                title="Null check",
                content="Always check for None before accessing attributes.",
            )
        )
        extractor = LessonExtractor(store, extractor_llm)
        results = extractor.recall_relevant("handle None attribute error")
        assert len(results) >= 1

    def test_format_for_context(self, store: LessonStore, extractor_llm: FakeLLMClient):
        extractor = LessonExtractor(store, extractor_llm)
        lessons = [
            Lesson(category="bug", title="Test", content="Do the thing."),
        ]
        formatted = extractor.format_for_context(lessons)
        assert "## Lessons from Past Sessions" in formatted
        assert "Test" in formatted

    def test_format_empty(self, store: LessonStore, extractor_llm: FakeLLMClient):
        extractor = LessonExtractor(store, extractor_llm)
        assert extractor.format_for_context([]) == ""
