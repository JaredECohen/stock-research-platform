import React, { useEffect } from "react";
import { track } from "@/lib/analytics";
import { featuredFAQ } from "@/content/faq";
import Disclosure from "@/components/public/Disclosure";
import FAQ from "@/components/public/FAQ";
import FeatureComparison from "@/components/public/FeatureComparison";
import Hero from "@/components/public/Hero";
import HowItWorks from "@/components/public/HowItWorks";
import PricingTable from "@/components/public/PricingTable";
import PublicShell from "@/components/public/PublicShell";
import SampleShowcase from "@/components/public/SampleShowcase";

export const LANDING_DESCRIPTION =
  "An AI investment committee that combines fundamentals, filings, earnings, valuation, risks, catalysts and scenarios into explainable research memos. Research and education only.";

/**
 * `/` — value proposition, how it works, the three samples, the plan
 * comparison, pricing, the featured FAQ and the disclosure. Signed-in
 * visitors are not redirected; the header offers "Continue to app".
 */
export default function Landing() {
  useEffect(() => {
    track("landing_view", { page: "landing" });
  }, []);
  return (
    <PublicShell title="Your AI investment committee" description={LANDING_DESCRIPTION}>
      <Hero />
      <HowItWorks />
      <SampleShowcase />
      <FeatureComparison compact />
      <PricingTable />
      <FAQ entries={featuredFAQ()} />
      <div className="mt-12">
        <Disclosure />
      </div>
    </PublicShell>
  );
}
