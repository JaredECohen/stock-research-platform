import React from "react";
import { FAQ as ENTRIES } from "@/content/faq";
import Disclosure from "@/components/public/Disclosure";
import FAQ from "@/components/public/FAQ";
import PublicShell from "@/components/public/PublicShell";

export default function FAQPage() {
  return (
    <PublicShell title="FAQ" description="Answers about the AI investment committee, the trial, Free Explorer allowances, data sources and privacy.">
      <div className="pt-12">
        <FAQ entries={ENTRIES} headingLevel={1} />
      </div>
      <div className="mt-12">
        <Disclosure />
      </div>
    </PublicShell>
  );
}
