"""The identity of a retrieved passage on the source ledger (bull/bear
design §5.3 step 4).

The filing analyst registers each passage it was given as
`chunk:<chunk id or accession>` (`filing_agent._retrieved_source`). The
debate registers its evidence-pool passages under the same ref form, so a
passage both of them saw is ONE source on the ledger, not two. The helper
stays in `filing_agent` (that module is not edited by the debate slice);
this module is the shared name both callers use.
"""
from __future__ import annotations

from typing import Any

from .filing_agent import _retrieved_source as retrieved_source

__all__ = ["chunk_ref", "retrieved_source"]


def chunk_ref(chunk: dict[str, Any], position: int) -> str:
    """`chunk:<id>` for one retrieved passage, exactly as the filing analyst
    registers it (`filing_agent.run`: `chunk:{chunk_id or ref}`)."""
    source = retrieved_source(chunk, position)
    return f"chunk:{source.get('chunk_id') or source['ref']}"
