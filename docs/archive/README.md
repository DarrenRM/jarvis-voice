# Archive

These are the plans, audits and notes Darren directed in January 2026, while the bot was first built and then refactored in the "Phase 1" to "Phase 4" commits. They are kept as a record of the process.

They describe the bot as it was then: Claude and Llama on Groq, self-hosted Firecrawl, a single large `VoiceSession`. Statuses and file paths in them are not kept up to date. For the bot as it is now, read the top-level `README.md` and `CLAUDE.md`.

- `initial-plan.md` and `discord-voice-claude-bot.md`: the first plan and product description.
- `project-reference.md`: the January reference (timing constants, voice path, provider setup).
- `known-issues.md`: issues logged during January testing.
- `architecture-audit.md`, `architecture-refactor-plan.md`, `refactor-edge-cases.md` and `refactor-notes.md`: the audit, the phased refactor plan, the timing-sensitive behavior it had to preserve, and the progress log.
- `testing-harness-plan.md`: the plan for an AI-graded end-to-end voice test harness (today's harness is `scripts/e2e`).
