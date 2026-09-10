import React from "react";
import LegalDoc, { LEGAL_DOCS, type LegalDocId } from "@/components/public/LegalDoc";
import PublicShell from "@/components/public/PublicShell";

/** /privacy, /terms, /billing-terms, /cookies — one page, four documents. */
export default function Legal({ doc }: { doc: LegalDocId }) {
  const d = LEGAL_DOCS[doc];
  return (
    <PublicShell title={d.title} description={d.description}>
      <div className="pt-12">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">{d.title}</h1>
        <p className="text-sm text-slate-400 mt-2 mb-6">{d.description}</p>
        <LegalDoc doc={doc} />
      </div>
    </PublicShell>
  );
}
