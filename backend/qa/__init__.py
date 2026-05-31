"""QA harness — offline evaluation rig for single-stock memos.

Run as a CLI:

    python -m qa.run_matrix --smoke         # ~5 ticker × 5 question quick check
    python -m qa.run_matrix --full          # full matrix, longer
    python -m qa.run_matrix --resume        # pick up where the last run left off

Inputs:
    tickers.json    list of ticker dicts the matrix sweeps over
    questions.json  list of question templates (with expected agent
                    activations + score-rubric tags)

Outputs (under qa/out/<run-id>/):
    results.jsonl              one row per (ticker, question) attempt
    summary.csv                CSV with status + scores per row
    failure_taxonomy.json      issue counter + per-axis score buckets
    improvement_report.md      human-readable score + recommendation digest
    improvement_backlog.json   prioritized backlog drawn from the taxonomy
    run_state.json             completion + checkpoint metadata
"""
