# Filing citation provenance correction

The META v1 validation exposed a metadata defect: the filing analyst retrieved passages from multiple filings but labeled every passage with the selected primary 10-K's accession. The vector-result projection discarded the passage's actual accession and stored document ID before citations were assembled.

The repair retains each hit's identity, period and source metadata. Saved passage citations use their own accession; BM25 results retain their own source ID. When only a numeric stored filing ID is available, the reference explicitly uses `filing_doc:<id>`. A passage with no usable identity is labeled `unattributed_chunk:<id or position>`, classified as an `other` citation, excluded from the filing bibliography, and reported with the complete count and identifiers in finding data and a warning log. Primary MD&A and risk citations continue to cite the primary filing. Deterministic findings also retain retrieval provenance.

The analyst's outgoing prompt strings, selected text, budgets and model arguments are unchanged. A regression test changes only a hit's accession and verifies that the entire outgoing analyst request remains identical while the saved citation changes. Tests also cover mixed filing periods, BM25 source filtering, internal document IDs, completely unattributed passages and deterministic output.

This identifies the source of a retrieved passage. It does not prove that every model claim is supported by that passage, that the entire corpus has been reindexed, or that semantic retrieval succeeded. Claim-to-passage checks and supplying richer source labels to model prompts remain separate research-process proposals. Existing saved memos, including META v1, are not rewritten.
