# AuraDerma Skills Registry

This directory stores skill summaries used by the agent router.

Each skill should live in its own subdirectory:

- `skills/<skill_name>/summary.md`
- `skills/<skill_name>/skill.md`
- optional extra artifacts: schema notes, examples

The CLI loads these summaries at runtime and passes only the relevant subset to the LLM router.
