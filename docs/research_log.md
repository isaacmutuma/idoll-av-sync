# Research Log — IDOLL AV Sync

Technical decisions and rationale. Newest entries at the top.

---

## 2026-05-19 — Project context persisted in repo

**Decision:** Store full project brief in `docs/PROJECT_CONTEXT.md` and load it via Cursor rule `.cursor/rules/idoll-project-context.mdc` (`alwaysApply: true`).

**Rationale:** Ensures every Cursor session has the same goals, structure, libraries, and code standards without re-pasting context. The rule is a short pointer; the markdown file is the single source of truth.

**Alternatives considered:** Pasting context at session start only (fragile); embedding entire brief in the rule file (too long for rule best practices).
