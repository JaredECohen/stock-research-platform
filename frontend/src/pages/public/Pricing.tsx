import React, { useEffect } from "react";
import { track } from "@/lib/analytics";
import { FAQ as ALL_FAQ } from "@/content/faq";
import Disclosure from "@/components/public/Disclosure";
import FAQ from "@/components/public/FAQ";
import FeatureComparison from "@/components/public/FeatureComparison";
import PricingTable from "@/components/public/PricingTable";
import PublicShell from "@/components/public/PublicShell";

const PRICING_FAQ_IDS = ["trial", "free", "periods", "distinct", "cancel"];

export default function Pricing() {
  useEffect(() => {
    track("pricing_view", { page: "pricing" });
  }, []);
  const entries = ALL_FAQ.filter((f) => PRICING_FAQ_IDS.includes(f.id));
  return (
    <PublicShell
      title="Pricing"
      description="Free Explorer opens the committee's stored work; Pro is for underwriting. Trial without a card, allowances per UTC calendar month."
    >
      <div className="pt-12">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">Plans and pricing</h1>
        <p className="text-slate-300 mt-2 max-w-2xl">
          Two plans, one rule: the allowances shown here are read from the same table the service enforces.
        </p>
      </div>
      <PricingTable />
      <FeatureComparison />
      <FAQ entries={entries} title="Pricing questions" />
      <div className="mt-12">
        <Disclosure />
      </div>
    </PublicShell>
  );
}
