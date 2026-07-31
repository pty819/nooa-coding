"""Cross-session learning: extract and recall lessons from past failures."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Lesson(BaseModel):
    """A durable lesson extracted from a coding session."""

    lesson_id: str = ""
    category: str = "general"  # "bug", "pattern", "tool", "workflow", "general"
    title: str
    content: str
    context: str = ""  # What was being attempted
    source_session: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    recall_count: int = 0
    last_recalled: str | None = None


class LessonStore:
    """Persistent storage for cross-session lessons using SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                lesson_id TEXT PRIMARY KEY,
                category TEXT NOT NULL DEFAULT 'general',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT DEFAULT '',
                source_session TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                recall_count INTEGER DEFAULT 0,
                last_recalled TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lessons_category ON lessons(category)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lessons_created ON lessons(created_at)
        """)
        self._conn.commit()

    def add(self, lesson: Lesson) -> str:
        """Store a new lesson. Returns the lesson_id."""
        import uuid

        if not lesson.lesson_id:
            lesson.lesson_id = uuid.uuid4().hex[:12]

        self._conn.execute(
            """
            INSERT OR REPLACE INTO lessons
            (lesson_id, category, title, content, context, source_session, created_at, recall_count, last_recalled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson.lesson_id,
                lesson.category,
                lesson.title,
                lesson.content,
                lesson.context,
                lesson.source_session,
                lesson.created_at,
                lesson.recall_count,
                lesson.last_recalled,
            ),
        )
        self._conn.commit()
        return lesson.lesson_id

    def recall(
        self,
        query: str = "",
        *,
        category: str | None = None,
        limit: int = 5,
    ) -> list[Lesson]:
        """Recall relevant lessons, optionally filtered by category or keyword."""
        conditions: list[str] = []
        params: list[Any] = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if query:
            conditions.append("(title LIKE ? OR content LIKE ? OR context LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM lessons {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        lessons = [Lesson(**dict(row)) for row in rows]

        # Update recall stats.
        now = datetime.now(UTC).isoformat()
        for lesson in lessons:
            self._conn.execute(
                "UPDATE lessons SET recall_count = recall_count + 1, last_recalled = ? WHERE lesson_id = ?",
                (now, lesson.lesson_id),
            )
        self._conn.commit()

        return lessons

    def recent(self, limit: int = 10) -> list[Lesson]:
        """Get the most recent lessons."""
        rows = self._conn.execute(
            "SELECT * FROM lessons ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Lesson(**dict(row)) for row in rows]

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        by_category = dict(
            self._conn.execute(
                "SELECT category, COUNT(*) FROM lessons GROUP BY category"
            ).fetchall()
        )
        return {"total": total, "by_category": by_category}

    def delete(self, lesson_id: str) -> bool:
        """Delete a lesson by ID. Returns True if deleted."""
        cursor = self._conn.execute(
            "DELETE FROM lessons WHERE lesson_id = ?", (lesson_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()


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
        """Recall lessons that might be relevant to a new task."""
        # Extract keywords from the task for matching.
        keywords = [w for w in task.lower().split() if len(w) > 3][:5]
        lessons: list[Lesson] = []
        seen_ids: set[str] = set()

        for keyword in keywords:
            for lesson in self._store.recall(keyword, limit=2):
                if lesson.lesson_id not in seen_ids:
                    lessons.append(lesson)
                    seen_ids.add(lesson.lesson_id)
            if len(lessons) >= limit:
                break

        return lessons[:limit]

    def format_for_context(self, lessons: list[Lesson]) -> str:
        """Format lessons as context to inject into the system prompt."""
        if not lessons:
            return ""
        lines = ["## Lessons from Past Sessions\n"]
        for lesson in lessons:
            lines.append(f"- **{lesson.title}** ({lesson.category}): {lesson.content}")
        return "\n".join(lines)


def get_default_store() -> LessonStore:
    """Get the default lesson store location."""
    default_path = Path.home() / ".config" / "nooa-coding" / "lessons.db"
    return LessonStore(default_path)


__all__ = [
    "Lesson",
    "LessonExtractor",
    "LessonStore",
    "get_default_store",
]
