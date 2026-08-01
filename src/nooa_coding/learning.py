"""Cross-session learning: extract and recall lessons from past failures.

Backed by nooa-memory's MemoryManager — lessons are stored as SKILL-type
memories with a ``lesson`` tag, gaining vector retrieval, dedup-on-write,
reinforcement, reflection/consolidation, and forgetting for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from nooa_memory import MemoryManager
    from nooa_memory.schema import Memory

# Tag prefix used to distinguish lessons from other memories.
LESSON_TAG = "lesson"


class Lesson(BaseModel):
    """A durable lesson extracted from a coding session (view model)."""

    lesson_id: str = ""
    category: str = "general"  # "bug", "pattern", "tool", "workflow", "general"
    title: str
    content: str
    context: str = ""  # What was being attempted
    source_session: str = ""
    created_at: str = ""
    recall_count: int = 0
    last_recalled: str | None = None


def _memory_to_lesson(mem: Memory) -> Lesson:
    """Convert a nooa-memory Memory object to our Lesson view model."""
    tags = mem.tags or []
    category = "general"
    for tag in tags:
        if tag.startswith("cat:"):
            category = tag[4:]
            break

    return Lesson(
        lesson_id=mem.id,
        category=category,
        title=mem.title or mem.content[:80],
        content=mem.content,
        context=getattr(mem, "source_task_ref", "") or "",
        source_session="",
        created_at=str(mem.created_at) if hasattr(mem, "created_at") else "",
        recall_count=getattr(mem, "reinforcement_count", 0),
        last_recalled=None,
    )


class LessonStore:
    """Persistent storage for cross-session lessons backed by MemoryManager.

    This is a thin adapter that maps the lesson API onto nooa-memory's
    remember/recall/forget, giving us vector similarity search, dedup,
    reinforcement, and reflection for free.
    """

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    def add(self, lesson: Lesson) -> str:
        """Store a new lesson. Returns the lesson_id (memory id)."""
        from nooa_memory.schema import MemoryType

        tags = [LESSON_TAG, f"cat:{lesson.category}"]
        if lesson.source_session:
            tags.append(f"session:{lesson.source_session[:12]}")

        content = lesson.content
        if lesson.context:
            content = f"{lesson.content}\n\nContext: {lesson.context}"

        memory_id = self._memory.remember(
            content,
            type=MemoryType.SKILL,
            title=lesson.title,
            tags=tags,
            importance=0.7,
            source_task_ref=lesson.context or None,
            dedup=True,
        )
        return memory_id

    def recall(
        self,
        query: str = "",
        *,
        category: str | None = None,
        limit: int = 5,
    ) -> list[Lesson]:
        """Recall relevant lessons using vector similarity search."""
        if not query:
            # No query — return recent lessons.
            return self.recent(limit=limit)

        memories = self._memory.recall(query, k=limit * 2)
        lessons = []
        for mem in memories:
            # Filter to lesson-tagged memories only.
            if LESSON_TAG not in (mem.tags or []):
                continue
            if category:
                if f"cat:{category}" not in (mem.tags or []):
                    continue
            lessons.append(_memory_to_lesson(mem))
            if len(lessons) >= limit:
                break
        return lessons

    def recent(self, limit: int = 10) -> list[Lesson]:
        """Get the most recent lessons."""
        # Use store directly to get all lesson memories sorted by time.
        all_mems = self._memory.store.all_memories(owner=self._memory.role)
        lesson_mems = [
            m for m in all_mems if LESSON_TAG in (m.tags or [])
        ]
        # Sort by created_at descending.
        lesson_mems.sort(key=lambda m: m.created_at, reverse=True)
        return [_memory_to_lesson(m) for m in lesson_mems[:limit]]

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        all_mems = self._memory.store.all_memories(owner=self._memory.role)
        lesson_mems = [m for m in all_mems if LESSON_TAG in (m.tags or [])]
        by_category: dict[str, int] = {}
        for m in lesson_mems:
            cat = "general"
            for tag in m.tags or []:
                if tag.startswith("cat:"):
                    cat = tag[4:]
                    break
            by_category[cat] = by_category.get(cat, 0) + 1
        return {"total": len(lesson_mems), "by_category": by_category}

    def delete(self, lesson_id: str) -> bool:
        """Delete a lesson by ID. Returns True if deleted."""
        return self._memory.forget(lesson_id)

    def close(self) -> None:
        """No-op — MemoryManager lifecycle is owned by the agent."""


class LessonExtractor:
    """Extract lessons from failed or blocked coding tasks using LLM."""

    def __init__(self, store: LessonStore, llm: Any) -> None:
        self._store = store
        self._llm = llm

    async def extract_from_failure(
        self,
        task: str,
        error: str,
        evidence: str = "",
        session_id: str = "",
    ) -> Lesson | None:
        """Analyze a failure and extract a reusable lesson."""
        if not error or len(error) < 20:
            return None  # Too short to be meaningful.

        extract_prompt = (
            "You are a learning system. Analyze this coding failure and extract "
            "ONE concise, reusable lesson that would help avoid similar failures "
            "in future sessions.\n\n"
            f"TASK ATTEMPTED: {task[:500]}\n\n"
            f"ERROR/FAILURE: {error[:1000]}\n\n"
        )
        if evidence:
            extract_prompt += f"EVIDENCE: {evidence[:500]}\n\n"
        extract_prompt += (
            "Respond in this EXACT format:\n"
            "CATEGORY: <bug|pattern|tool|workflow|general>\n"
            "TITLE: <short descriptive title, max 80 chars>\n"
            "LESSON: <the reusable lesson, 1-3 sentences>\n"
        )

        try:
            response = await self._llm.acall(
                [{"role": "user", "content": extract_prompt}]
            )
            content = response.content or ""
        except Exception:
            return None

        # Parse the response.
        category = "general"
        title = ""
        lesson_text = ""

        for line in content.splitlines():
            line = line.strip()
            upper = line.upper()
            if upper.startswith("CATEGORY:"):
                cat = line[9:].strip().lower()
                if cat in ("bug", "pattern", "tool", "workflow", "general"):
                    category = cat
            elif upper.startswith("TITLE:"):
                title = line[6:].strip()[:80]
            elif upper.startswith("LESSON:"):
                lesson_text = line[7:].strip()

        if not title or not lesson_text:
            return None

        lesson = Lesson(
            category=category,
            title=title,
            content=lesson_text,
            context=task[:200],
            source_session=session_id,
        )
        self._store.add(lesson)
        return lesson

    def recall_relevant(self, task: str, limit: int = 3) -> list[Lesson]:
        """Recall lessons that might be relevant to a new task via vector search."""
        return self._store.recall(task, limit=limit)

    def format_for_context(self, lessons: list[Lesson]) -> str:
        """Format lessons as context to inject into the system prompt."""
        if not lessons:
            return ""
        lines = ["## Lessons from Past Sessions\n"]
        for lesson in lessons:
            lines.append(f"- **{lesson.title}** ({lesson.category}): {lesson.content}")
        return "\n".join(lines)


__all__ = [
    "LESSON_TAG",
    "Lesson",
    "LessonExtractor",
    "LessonStore",
]
