"""W7 learning ledger: long-term memory as priors that update with evidence.

See `.claude/memory/proposals/design-w7-learning-final.md` (sections 3-7)
and slice S18 of the 2026-09-24 integration plan. Import-light on purpose:
`ledger` and `control` pull in the database and services, so callers import
the submodule they need (the postmortem and filing hooks import it lazily,
inside the function, to avoid import cycles).
"""
