"""Programmatic demo dataset for MarketMosaic.

This module builds a coherent, multi-year dataset for ~28 large-cap names
covering Tech, Financials, Consumer, Healthcare, Energy, Industrials, and
Utilities. The data is illustrative — it is built from rough public-company
shapes (margin profile, growth rate, capital structure, beta) but is NOT
real-time and should NEVER be used for actual investment decisions.

We construct everything from a compact `COMPANY_PROFILES` table and a few
deterministic generators so the screener, DCF, comps, and portfolio engines
can all run end-to-end without any external API keys.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Compact profile table — illustrative figures only (USD, $ in billions).
# Fields: market_cap_b, revenue_b, op_margin, fcf_margin, growth, beta,
# debt_to_ebitda, shares_b, price, sector, industry, sub_industry, cik,
# description, segments, drivers, risks, fye.
# ---------------------------------------------------------------------------
COMPANY_PROFILES: Dict[str, Dict] = {
    "NVDA": dict(
        company_name="NVIDIA Corporation", exchange="NASDAQ",
        sector="Technology", industry="Semiconductors", sub_industry="AI Accelerators",
        market_cap_b=2900.0, revenue_b=130.0, op_margin=0.61, fcf_margin=0.45,
        growth=0.55, beta=1.65, debt_to_ebitda=0.2, shares_b=24.5, price=118.0,
        cik="0001045810", fye="January",
        description="Designs GPUs and accelerated computing platforms for AI training and inference, gaming, professional visualization, and data center workloads.",
        segments=["Data Center", "Gaming", "Professional Visualization", "Automotive"],
        drivers=["Hyperscaler AI capex", "Sovereign AI projects", "Networking + software stack"],
        risks=["AI capex digestion", "Geopolitical export restrictions", "Custom-silicon competition"],
    ),
    "MSFT": dict(
        company_name="Microsoft Corporation", exchange="NASDAQ",
        sector="Technology", industry="Software", sub_industry="Diversified Software",
        market_cap_b=3100.0, revenue_b=245.0, op_margin=0.45, fcf_margin=0.30,
        growth=0.16, beta=0.95, debt_to_ebitda=0.4, shares_b=7.43, price=415.0,
        cik="0000789019", fye="June",
        description="Cloud, productivity, and platform software with Azure, Microsoft 365, Dynamics, GitHub, LinkedIn, and Windows.",
        segments=["Productivity & Business Processes", "Intelligent Cloud", "More Personal Computing"],
        drivers=["Azure AI growth", "Copilot monetization", "Enterprise cloud migration"],
        risks=["Azure deceleration", "Capex intensity", "Antitrust and AI regulation"],
    ),
    "GOOGL": dict(
        company_name="Alphabet Inc.", exchange="NASDAQ",
        sector="Communication Services", industry="Interactive Media", sub_industry="Search & Cloud",
        market_cap_b=2050.0, revenue_b=345.0, op_margin=0.32, fcf_margin=0.22,
        growth=0.13, beta=1.05, debt_to_ebitda=0.2, shares_b=12.3, price=167.0,
        cik="0001652044", fye="December",
        description="Search, YouTube, Android, Google Cloud, and a portfolio of Other Bets including Waymo.",
        segments=["Google Services", "Google Cloud", "Other Bets"],
        drivers=["AI Overviews monetization", "Cloud growth + GPU TPU stack", "YouTube Shorts + connected TV"],
        risks=["Search disruption from AI", "Antitrust remedies", "Capex on AI infrastructure"],
    ),
    "META": dict(
        company_name="Meta Platforms, Inc.", exchange="NASDAQ",
        sector="Communication Services", industry="Interactive Media", sub_industry="Social",
        market_cap_b=1300.0, revenue_b=160.0, op_margin=0.40, fcf_margin=0.30,
        growth=0.20, beta=1.20, debt_to_ebitda=0.3, shares_b=2.55, price=510.0,
        cik="0001326801", fye="December",
        description="Family of apps (Facebook, Instagram, WhatsApp, Threads), Reality Labs hardware, and AI infrastructure.",
        segments=["Family of Apps", "Reality Labs"],
        drivers=["AI-driven ad targeting", "Reels engagement and monetization", "Llama / agentic AI products"],
        risks=["Reality Labs cash burn", "Privacy/regulatory headwinds", "Ad cyclicality"],
    ),
    "AMZN": dict(
        company_name="Amazon.com, Inc.", exchange="NASDAQ",
        sector="Consumer Discretionary", industry="Internet Retail", sub_industry="E-commerce + Cloud",
        market_cap_b=1900.0, revenue_b=620.0, op_margin=0.10, fcf_margin=0.06,
        growth=0.12, beta=1.15, debt_to_ebitda=0.6, shares_b=10.5, price=183.0,
        cik="0001018724", fye="December",
        description="Online retail platform, AWS cloud computing, advertising, and a network of fulfillment infrastructure.",
        segments=["North America Retail", "International Retail", "AWS"],
        drivers=["AWS reacceleration with AI workloads", "Retail margin expansion", "Advertising monetization"],
        risks=["Capex outpacing returns", "Retail margin volatility", "Antitrust scrutiny"],
    ),
    "AVGO": dict(
        company_name="Broadcom Inc.", exchange="NASDAQ",
        sector="Technology", industry="Semiconductors", sub_industry="Networking + Software",
        market_cap_b=750.0, revenue_b=53.0, op_margin=0.45, fcf_margin=0.40,
        growth=0.40, beta=1.20, debt_to_ebitda=2.4, shares_b=4.66, price=160.0,
        cik="0001730168", fye="October",
        description="Networking, broadband, wireless, and infrastructure software (post-VMware) with AI ASIC custom-silicon exposure.",
        segments=["Semiconductor Solutions", "Infrastructure Software"],
        drivers=["AI ASIC + networking", "VMware monetization", "Custom silicon for hyperscalers"],
        risks=["AI ASIC competition", "Cyclicality in non-AI semis", "Integration risk on VMware"],
    ),
    "AMD": dict(
        company_name="Advanced Micro Devices, Inc.", exchange="NASDAQ",
        sector="Technology", industry="Semiconductors", sub_industry="CPU + GPU",
        market_cap_b=240.0, revenue_b=24.0, op_margin=0.10, fcf_margin=0.13,
        growth=0.18, beta=1.70, debt_to_ebitda=0.6, shares_b=1.63, price=147.0,
        cik="0000002488", fye="December",
        description="x86 server and client CPUs, gaming and data center GPUs (Instinct MI series), and adaptive compute.",
        segments=["Data Center", "Client", "Gaming", "Embedded"],
        drivers=["MI300 / MI325 ramp", "EPYC server share gains", "AI inference workload mix"],
        risks=["NVIDIA software moat", "Gaming/embedded weakness", "Wafer supply concentration"],
    ),
    "AAPL": dict(
        company_name="Apple Inc.", exchange="NASDAQ",
        sector="Technology", industry="Consumer Electronics", sub_industry="Hardware + Services",
        market_cap_b=2700.0, revenue_b=390.0, op_margin=0.31, fcf_margin=0.25,
        growth=0.04, beta=1.10, debt_to_ebitda=0.6, shares_b=15.20, price=180.0,
        cik="0000320193", fye="September",
        description="iPhone, Mac, iPad, wearables, and a high-margin services business that is the centerpiece of margin expansion.",
        segments=["iPhone", "Mac", "iPad", "Wearables", "Services"],
        drivers=["Services margin mix", "Apple Intelligence + iPhone replacement cycle", "India/EM growth"],
        risks=["China demand and regulatory risk", "Mature smartphone TAM", "Antitrust on App Store"],
    ),
    "JPM": dict(
        company_name="JPMorgan Chase & Co.", exchange="NYSE",
        sector="Financials", industry="Banks", sub_industry="Universal Bank",
        market_cap_b=580.0, revenue_b=170.0, op_margin=0.43, fcf_margin=0.28,
        growth=0.07, beta=1.10, debt_to_ebitda=4.0, shares_b=2.84, price=200.0,
        cik="0000019617", fye="December",
        description="Global universal bank with leadership in IB, markets, asset management, and consumer banking.",
        segments=["Consumer & Community Banking", "Corporate & Investment Bank", "Commercial Banking", "Asset & Wealth Management"],
        drivers=["NII tailwinds + deposit franchise", "Capital markets recovery", "Excess capital deployment"],
        risks=["Credit normalization", "Yield-curve dynamics", "Capital rules"],
    ),
    "BAC": dict(
        company_name="Bank of America Corporation", exchange="NYSE",
        sector="Financials", industry="Banks", sub_industry="Universal Bank",
        market_cap_b=300.0, revenue_b=98.0, op_margin=0.32, fcf_margin=0.25,
        growth=0.05, beta=1.20, debt_to_ebitda=4.5, shares_b=7.85, price=38.0,
        cik="0000070858", fye="December",
        description="Large U.S. consumer + commercial bank with a global markets and wealth management franchise.",
        segments=["Consumer Banking", "Global Wealth & Investment Management", "Global Banking", "Global Markets"],
        drivers=["Deposit franchise leverage to rates", "Loan growth", "Capital return"],
        risks=["AOCI overhang", "Office CRE", "Regulatory capital"],
    ),
    "GS": dict(
        company_name="The Goldman Sachs Group, Inc.", exchange="NYSE",
        sector="Financials", industry="Capital Markets", sub_industry="Investment Bank",
        market_cap_b=160.0, revenue_b=51.0, op_margin=0.32, fcf_margin=0.25,
        growth=0.10, beta=1.30, debt_to_ebitda=5.0, shares_b=0.32, price=510.0,
        cik="0000886982", fye="December",
        description="Investment bank with Global Banking & Markets and Asset & Wealth Management franchises.",
        segments=["Global Banking & Markets", "Asset & Wealth Management", "Platform Solutions"],
        drivers=["IB reopening", "Asset & wealth fee growth", "Capital markets re-rating"],
        risks=["Capital markets cyclicality", "CRE losses", "Capital deployment timing"],
    ),
    "MS": dict(
        company_name="Morgan Stanley", exchange="NYSE",
        sector="Financials", industry="Capital Markets", sub_industry="Wealth + IB",
        market_cap_b=160.0, revenue_b=58.0, op_margin=0.30, fcf_margin=0.22,
        growth=0.08, beta=1.20, debt_to_ebitda=4.5, shares_b=1.62, price=98.0,
        cik="0000895421", fye="December",
        description="Wealth management franchise scaled by E*TRADE, with markets and investment banking businesses.",
        segments=["Institutional Securities", "Wealth Management", "Investment Management"],
        drivers=["Net new asset gathering", "Fee-based asset mix", "IB reopening"],
        risks=["Wealth flows volatility", "Markets cyclicality"],
    ),
    "V": dict(
        company_name="Visa Inc.", exchange="NYSE",
        sector="Financials", industry="Payments", sub_industry="Card Networks",
        market_cap_b=550.0, revenue_b=36.0, op_margin=0.66, fcf_margin=0.55,
        growth=0.10, beta=0.95, debt_to_ebitda=0.7, shares_b=2.01, price=275.0,
        cik="0001403161", fye="September",
        description="Global payments network connecting issuers, acquirers, and merchants.",
        segments=["Payments Volume", "Cross-border", "Value-added Services"],
        drivers=["Cross-border travel rebound", "Real-time payments + value-added services", "EM digitization"],
        risks=["Regulatory pressure on fees", "FX strength reducing cross-border"],
    ),
    "MA": dict(
        company_name="Mastercard Incorporated", exchange="NYSE",
        sector="Financials", industry="Payments", sub_industry="Card Networks",
        market_cap_b=440.0, revenue_b=27.0, op_margin=0.58, fcf_margin=0.45,
        growth=0.11, beta=1.05, debt_to_ebitda=1.0, shares_b=0.93, price=470.0,
        cik="0001141391", fye="December",
        description="Global payments network with strong cross-border and value-added services exposure.",
        segments=["Payment Network", "Value-added Services"],
        drivers=["Cross-border rebound", "Services attach rate"],
        risks=["Regulatory pressure", "FX"],
    ),
    "COST": dict(
        company_name="Costco Wholesale Corporation", exchange="NASDAQ",
        sector="Consumer Staples", industry="Hypermarkets & Super Centers", sub_industry="Membership Warehouse",
        market_cap_b=400.0, revenue_b=255.0, op_margin=0.035, fcf_margin=0.025,
        growth=0.07, beta=0.85, debt_to_ebitda=0.5, shares_b=0.444, price=900.0,
        cik="0000909832", fye="August",
        description="Membership-based warehouse retailer with predictable membership economics.",
        segments=["U.S.", "International", "Other"],
        drivers=["Membership renewal rate >90%", "Price increase optionality", "Warehouse openings"],
        risks=["Premium valuation", "Membership saturation", "Wage pressure"],
    ),
    "WMT": dict(
        company_name="Walmart Inc.", exchange="NYSE",
        sector="Consumer Staples", industry="Hypermarkets & Super Centers", sub_industry="Mass Retail",
        market_cap_b=560.0, revenue_b=665.0, op_margin=0.045, fcf_margin=0.025,
        growth=0.05, beta=0.55, debt_to_ebitda=1.5, shares_b=8.04, price=70.0,
        cik="0000104169", fye="January",
        description="Mass retailer with growing advertising, marketplace, and Walmart+ businesses.",
        segments=["Walmart U.S.", "Walmart International", "Sam's Club"],
        drivers=["High-income customer trade-down", "Advertising + marketplace mix", "Margin expansion"],
        risks=["Wage pressure", "Tariffs on imported goods", "Grocery promotional cycles"],
    ),
    "HD": dict(
        company_name="The Home Depot, Inc.", exchange="NYSE",
        sector="Consumer Discretionary", industry="Home Improvement", sub_industry="Home Improvement Retail",
        market_cap_b=355.0, revenue_b=152.0, op_margin=0.14, fcf_margin=0.10,
        growth=0.02, beta=1.05, debt_to_ebitda=2.4, shares_b=0.99, price=360.0,
        cik="0000354950", fye="January",
        description="Largest home-improvement retailer, leveraged to housing turnover and Pro contractor spend.",
        segments=["U.S.", "Mexico/Canada"],
        drivers=["Housing turnover recovery", "Pro share gains", "Distribution efficiency"],
        risks=["Housing market weakness", "Big-ticket discretionary"],
    ),
    "MCD": dict(
        company_name="McDonald's Corporation", exchange="NYSE",
        sector="Consumer Discretionary", industry="Restaurants", sub_industry="QSR",
        market_cap_b=210.0, revenue_b=26.0, op_margin=0.45, fcf_margin=0.27,
        growth=0.04, beta=0.65, debt_to_ebitda=3.0, shares_b=0.73, price=290.0,
        cik="0000063908", fye="December",
        description="Global QSR franchisor with ~95% franchised mix and rent-stream economics.",
        segments=["U.S.", "International Operated Markets", "International Developmental Licensed"],
        drivers=["Loyalty and digital growth", "Value menu repositioning", "Unit growth in Asia"],
        risks=["Low-income consumer pressure", "Currency", "Franchisee margin pressure"],
    ),
    "NKE": dict(
        company_name="NIKE, Inc.", exchange="NYSE",
        sector="Consumer Discretionary", industry="Apparel & Footwear", sub_industry="Athletic Brands",
        market_cap_b=110.0, revenue_b=51.0, op_margin=0.10, fcf_margin=0.10,
        growth=-0.03, beta=1.10, debt_to_ebitda=1.6, shares_b=1.50, price=72.0,
        cik="0000320187", fye="May",
        description="Global athletic apparel and footwear brand with DTC + wholesale mix repositioning.",
        segments=["NIKE Brand", "Converse"],
        drivers=["Innovation pipeline reset", "China stabilization", "Wholesale reset"],
        risks=["Brand momentum", "China consumer", "DTC margin reset"],
    ),
    "SBUX": dict(
        company_name="Starbucks Corporation", exchange="NASDAQ",
        sector="Consumer Discretionary", industry="Restaurants", sub_industry="Coffee",
        market_cap_b=110.0, revenue_b=37.0, op_margin=0.15, fcf_margin=0.10,
        growth=0.02, beta=1.05, debt_to_ebitda=2.5, shares_b=1.13, price=98.0,
        cik="0000829224", fye="September",
        description="Global premium coffee retailer with ~38k stores, mobile order/pay, and rewards economics.",
        segments=["North America", "International", "Channel Development"],
        drivers=["U.S. transaction recovery", "China repositioning", "Reinvention plan execution"],
        risks=["Throughput / labor cost", "China softness", "Discretionary consumer weakness"],
    ),
    "LLY": dict(
        company_name="Eli Lilly and Company", exchange="NYSE",
        sector="Healthcare", industry="Pharmaceuticals", sub_industry="Diabetes/Obesity",
        market_cap_b=820.0, revenue_b=42.0, op_margin=0.34, fcf_margin=0.20,
        growth=0.30, beta=0.45, debt_to_ebitda=2.0, shares_b=0.95, price=860.0,
        cik="0000059478", fye="December",
        description="Pharmaceutical company centered on incretins (Mounjaro, Zepbound), oncology, and immunology.",
        segments=["Diabetes & Obesity", "Oncology", "Immunology", "Neuroscience"],
        drivers=["GLP-1 capacity ramp", "Oral GLP-1 readouts", "Manufacturing capacity"],
        risks=["Pricing pressure on GLP-1", "Competitive entries", "Capacity execution"],
    ),
    "UNH": dict(
        company_name="UnitedHealth Group Incorporated", exchange="NYSE",
        sector="Healthcare", industry="Managed Care", sub_industry="Diversified",
        market_cap_b=480.0, revenue_b=380.0, op_margin=0.085, fcf_margin=0.06,
        growth=0.08, beta=0.55, debt_to_ebitda=1.6, shares_b=0.92, price=520.0,
        cik="0000731766", fye="December",
        description="Largest U.S. managed care organization with UnitedHealthcare insurance and Optum services.",
        segments=["UnitedHealthcare", "Optum Health", "Optum Insight", "Optum Rx"],
        drivers=["Optum growth", "Medicare Advantage normalization", "Value-based care"],
        risks=["MA rate cuts", "Medical-cost trend", "Regulatory scrutiny"],
    ),
    "JNJ": dict(
        company_name="Johnson & Johnson", exchange="NYSE",
        sector="Healthcare", industry="Pharmaceuticals", sub_industry="Pharma + MedTech",
        market_cap_b=380.0, revenue_b=88.0, op_margin=0.27, fcf_margin=0.22,
        growth=0.05, beta=0.55, debt_to_ebitda=1.0, shares_b=2.41, price=158.0,
        cik="0000200406", fye="December",
        description="Diversified pharma + medtech post-Kenvue spin, with oncology, immunology, and surgical platforms.",
        segments=["Innovative Medicine", "MedTech"],
        drivers=["Oncology pipeline", "MedTech recovery", "Stelara biosimilar offsets"],
        risks=["Stelara biosimilar erosion", "Talc litigation", "Pricing/policy"],
    ),
    "MRK": dict(
        company_name="Merck & Co., Inc.", exchange="NYSE",
        sector="Healthcare", industry="Pharmaceuticals", sub_industry="Big Pharma",
        market_cap_b=290.0, revenue_b=63.0, op_margin=0.30, fcf_margin=0.22,
        growth=0.06, beta=0.40, debt_to_ebitda=1.4, shares_b=2.53, price=115.0,
        cik="0000310158", fye="December",
        description="Pharmaceutical company anchored by Keytruda, with growing animal health and oncology pipeline.",
        segments=["Pharmaceutical", "Animal Health"],
        drivers=["Keytruda subcutaneous launch", "Pipeline diversification", "Animal Health growth"],
        risks=["Keytruda LOE in 2028", "Pipeline execution"],
    ),
    "XOM": dict(
        company_name="Exxon Mobil Corporation", exchange="NYSE",
        sector="Energy", industry="Integrated Oil & Gas", sub_industry="Super-major",
        market_cap_b=480.0, revenue_b=345.0, op_margin=0.13, fcf_margin=0.10,
        growth=0.0, beta=0.85, debt_to_ebitda=0.6, shares_b=4.39, price=110.0,
        cik="0000034088", fye="December",
        description="Integrated super-major with Permian + Guyana growth and downstream/chemicals exposure.",
        segments=["Upstream", "Energy Products", "Chemical Products", "Specialty Products"],
        drivers=["Guyana volume growth", "Permian efficiency", "Capital discipline / buybacks"],
        risks=["Oil price drawdown", "Energy transition", "Stranded asset risk"],
    ),
    "NEE": dict(
        company_name="NextEra Energy, Inc.", exchange="NYSE",
        sector="Utilities", industry="Electric Utilities", sub_industry="Renewables + Regulated",
        market_cap_b=160.0, revenue_b=27.0, op_margin=0.25, fcf_margin=0.05,
        growth=0.06, beta=0.55, debt_to_ebitda=4.5, shares_b=2.05, price=78.0,
        cik="0000753308", fye="December",
        description="Florida regulated utility (FPL) plus Energy Resources, the largest U.S. renewables developer.",
        segments=["FPL", "NextEra Energy Resources"],
        drivers=["Renewables backlog", "Data center power demand", "Rate base growth"],
        risks=["Long-end rate sensitivity", "Permitting", "Capex execution"],
    ),
    "CAT": dict(
        company_name="Caterpillar Inc.", exchange="NYSE",
        sector="Industrials", industry="Construction & Mining Equipment", sub_industry="Heavy Equipment",
        market_cap_b=170.0, revenue_b=66.0, op_margin=0.20, fcf_margin=0.13,
        growth=0.03, beta=1.10, debt_to_ebitda=1.8, shares_b=0.49, price=345.0,
        cik="0000018230", fye="December",
        description="Global heavy-equipment OEM with construction, resource industries, and energy & transportation segments.",
        segments=["Construction Industries", "Resource Industries", "Energy & Transportation"],
        drivers=["Infrastructure spending", "Data center power gen", "Aftermarket services"],
        risks=["Construction cyclicality", "Mining capex", "Inventory destocking"],
    ),
    "CRM": dict(
        company_name="Salesforce, Inc.", exchange="NYSE",
        sector="Technology", industry="Software", sub_industry="CRM + Platform",
        market_cap_b=265.0, revenue_b=37.0, op_margin=0.20, fcf_margin=0.30,
        growth=0.10, beta=1.15, debt_to_ebitda=1.0, shares_b=0.97, price=275.0,
        cik="0001108524", fye="January",
        description="Customer relationship management platform with Data Cloud and Agentforce as the AI growth vector.",
        segments=["Sales Cloud", "Service Cloud", "Marketing & Commerce", "Platform & Other"],
        drivers=["Agentforce monetization", "Data Cloud growth", "Margin expansion"],
        risks=["Seat-based saturation", "Competition from MSFT", "Macro deal cycles"],
    ),
    "PLTR": dict(
        company_name="Palantir Technologies Inc.", exchange="NASDAQ",
        sector="Technology", industry="Software", sub_industry="AIP / Analytics Platform",
        market_cap_b=180.0, revenue_b=2.9, op_margin=0.18, fcf_margin=0.40,
        growth=0.30, beta=2.40, debt_to_ebitda=0.0, shares_b=2.30, price=78.0,
        cik="0001321655", fye="December",
        description="AIP and Foundry data-integration platforms for government and commercial customers; AI-ops focus.",
        segments=["Government", "Commercial"],
        drivers=["AIP commercial bootcamps converting to ACV", "Government renewals + new awards", "Operating leverage on FCF"],
        risks=["Multiple-driven valuation", "Government budget cycles", "Concentration in large customers"],
    ),
    "NFLX": dict(
        company_name="Netflix, Inc.", exchange="NASDAQ",
        sector="Communication Services", industry="Entertainment", sub_industry="Streaming",
        market_cap_b=370.0, revenue_b=39.0, op_margin=0.27, fcf_margin=0.18,
        growth=0.15, beta=1.30, debt_to_ebitda=1.5, shares_b=0.43, price=860.0,
        cik="0001065280", fye="December",
        description="Subscription streaming service with a content library, ads tier, and growing live-event programming.",
        segments=["UCAN", "EMEA", "LATAM", "APAC"],
        drivers=["Ads-tier ARPU ramp", "Password-sharing crackdown durability", "Live + games optionality"],
        risks=["Content cost inflation", "Subscriber saturation in mature markets", "FX translation"],
    ),
    "TSLA": dict(
        company_name="Tesla, Inc.", exchange="NASDAQ",
        sector="Consumer Discretionary", industry="Automobiles", sub_industry="Electric Vehicles",
        market_cap_b=900.0, revenue_b=97.0, op_margin=0.08, fcf_margin=0.04,
        growth=0.05, beta=2.10, debt_to_ebitda=0.4, shares_b=3.18, price=283.0,
        cik="0001318605", fye="December",
        description="Designs and manufactures EVs, battery storage (Powerwall, Megapack), and develops Full-Self-Driving software.",
        segments=["Automotive", "Energy Generation & Storage", "Services & Other"],
        drivers=["Robotaxi / FSD monetization", "Energy storage growth", "Lower-priced model launches"],
        risks=["Auto gross margin compression", "FSD timeline / regulatory", "EV demand cyclicality + China competition"],
    ),
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _years(n: int = 4) -> List[int]:
    """Return n trailing fiscal years ending in latest full year (2024)."""
    end = 2024
    return list(range(end - n + 1, end + 1))


def build_income_statements(profile: Dict) -> List[Dict]:
    """Generate ~4 years of annual income statements that compound to today's revenue."""
    years = _years(4)
    g = profile["growth"]
    target_rev = profile["revenue_b"] * 1e9
    rows: List[Dict] = []
    rev = target_rev / ((1 + g) ** (len(years) - 1)) if g > -0.5 else target_rev
    for y in years:
        op_margin = profile["op_margin"]
        if y < years[-1]:
            # Slight margin progression
            op_margin = max(0.02, op_margin - 0.01 * (years[-1] - y))
        gross_margin = max(op_margin + 0.10, op_margin * 1.6)
        gross = rev * gross_margin
        operating_income = rev * op_margin
        ebitda = operating_income + rev * 0.04
        rd = rev * (0.10 if profile["sector"] in ("Technology", "Healthcare") else 0.02)
        sga = max(rev * 0.06, gross - operating_income - rd)
        interest = (profile["debt_to_ebitda"] * ebitda) * 0.05
        pretax = operating_income - interest
        tax_rate = 0.21
        tax = max(0.0, pretax * tax_rate)
        net = pretax - tax
        eps = net / (profile["shares_b"] * 1e9) if profile["shares_b"] else None
        rows.append(dict(
            period=str(y),
            revenue=rev,
            cost_of_revenue=rev - gross,
            gross_profit=gross,
            r_and_d=rd,
            sga=sga,
            operating_income=operating_income,
            ebit=operating_income,
            ebitda=ebitda,
            interest_expense=interest,
            pretax_income=pretax,
            tax_expense=tax,
            net_income=net,
            eps_basic=eps,
            eps_diluted=eps,
            weighted_avg_shares_basic=profile["shares_b"] * 1e9,
            weighted_avg_shares_diluted=profile["shares_b"] * 1e9,
        ))
        rev = rev * (1 + g)
    return rows


def build_cash_flows(profile: Dict, income_statements: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for inc in income_statements:
        rev = inc["revenue"]
        cfo = rev * (profile["fcf_margin"] + 0.05)
        capex = rev * 0.05
        fcf = cfo - capex
        da = rev * 0.04
        sbc = rev * (0.04 if profile["sector"] == "Technology" else 0.005)
        nwc = rev * 0.005
        rows.append(dict(
            period=inc["period"],
            cash_from_operations=cfo,
            capex=-abs(capex),
            free_cash_flow=fcf,
            depreciation_and_amortization=da,
            stock_based_compensation=sbc,
            change_in_working_capital=nwc,
            acquisitions=0.0,
            dividends_paid=-rev * 0.01,
            share_repurchases=-rev * 0.03,
            debt_issuance_repayment=0.0,
        ))
    return rows


def build_balance_sheets(profile: Dict, income_statements: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for inc in income_statements:
        rev = inc["revenue"]
        ebitda = inc["ebitda"]
        debt = profile["debt_to_ebitda"] * ebitda
        cash = rev * 0.10
        ar = rev * 0.10
        inv = rev * 0.05
        ppe = rev * 0.30
        intangibles = rev * 0.10
        goodwill = rev * 0.10
        total_assets = cash + ar + inv + ppe + intangibles + goodwill
        ap = rev * 0.06
        total_liab = ap + debt
        equity = max(rev * 0.50, total_assets - total_liab)
        total_liab = total_assets - equity
        rows.append(dict(
            period=inc["period"],
            cash_and_equivalents=cash * 0.7,
            short_term_investments=cash * 0.3,
            accounts_receivable=ar,
            inventory=inv,
            current_assets=cash + ar + inv,
            ppe_net=ppe,
            goodwill=goodwill,
            intangibles=intangibles,
            total_assets=total_assets,
            accounts_payable=ap,
            short_term_debt=debt * 0.2,
            current_liabilities=ap + debt * 0.2,
            long_term_debt=debt * 0.8,
            total_debt=debt,
            total_liabilities=total_liab,
            minority_interest=0.0,
            preferred_stock=0.0,
            shareholders_equity=equity,
        ))
    return rows


def build_price_history(profile: Dict, days: int = 252) -> List[Dict]:
    """Deterministic synthetic price path that ends at the profile's price."""
    end = date.today()
    seed = sum(ord(c) for c in profile["company_name"]) % 1000
    price = profile["price"]
    series: List[Dict] = []
    annual_drift = profile["growth"] * 0.6
    daily_drift = annual_drift / 252
    annual_vol = 0.20 + max(0.0, profile["beta"] - 1.0) * 0.10
    daily_vol = annual_vol / math.sqrt(252)
    # Build forward from past
    px = price / ((1 + annual_drift) ** (days / 252))
    for i in range(days):
        d = end - timedelta(days=days - i - 1)
        # deterministic pseudo-noise
        n = math.sin(seed + i * 0.37) * 0.4 + math.cos(seed * 0.13 + i * 0.11) * 0.6
        ret = daily_drift + daily_vol * n
        px = px * (1 + ret)
        series.append(dict(
            date=d.isoformat(),
            open=round(px * (1 - 0.002), 4),
            high=round(px * 1.01, 4),
            low=round(px * 0.99, 4),
            close=round(px, 4),
            adjusted_close=round(px, 4),
            volume=int(2_000_000 + 100_000 * (i % 7)),
            dividends=0.0,
            splits=0.0,
            total_return_adjusted_price=round(px, 4),
        ))
    # Anchor last price to profile
    if series:
        series[-1]["close"] = profile["price"]
        series[-1]["adjusted_close"] = profile["price"]
        series[-1]["total_return_adjusted_price"] = profile["price"]
    return series


def build_ratios(profile: Dict, income_statements: List[Dict],
                 balance_sheets: List[Dict], cash_flows: List[Dict]) -> Dict:
    latest = income_statements[-1]
    bs = balance_sheets[-1]
    cf = cash_flows[-1]
    market_cap = profile["market_cap_b"] * 1e9
    rev = latest["revenue"]
    ebitda_v = latest["ebitda"]
    debt = bs["total_debt"]
    cash = bs["cash_and_equivalents"] + bs["short_term_investments"]
    ev = market_cap + debt - cash
    return dict(
        revenue_growth=profile["growth"],
        gross_margin=latest["gross_profit"] / rev,
        operating_margin=latest["operating_income"] / rev,
        net_margin=latest["net_income"] / rev,
        fcf_margin=cf["free_cash_flow"] / rev,
        ROE=latest["net_income"] / bs["shareholders_equity"],
        ROA=latest["net_income"] / bs["total_assets"],
        ROIC=(latest["operating_income"] * (1 - 0.21)) / max(1.0, bs["shareholders_equity"] + debt),
        debt_to_ebitda=debt / max(1.0, ebitda_v),
        net_debt_to_ebitda=(debt - cash) / max(1.0, ebitda_v),
        interest_coverage=latest["operating_income"] / max(1.0, latest["interest_expense"]),
        current_ratio=bs["current_assets"] / max(1.0, bs["current_liabilities"]),
        quick_ratio=(bs["current_assets"] - bs["inventory"]) / max(1.0, bs["current_liabilities"]),
        asset_turnover=rev / max(1.0, bs["total_assets"]),
        inventory_turnover=rev / max(1.0, bs["inventory"]),
        PE=market_cap / max(1.0, latest["net_income"]),
        forward_PE=(market_cap / max(1.0, latest["net_income"])) * 0.92,
        EV_Revenue=ev / rev,
        EV_EBITDA=ev / max(1.0, ebitda_v),
        EV_EBIT=ev / max(1.0, latest["operating_income"]),
        PS=market_cap / rev,
        PB=market_cap / max(1.0, bs["shareholders_equity"]),
        PFCF=market_cap / max(1.0, cf["free_cash_flow"]),
        FCF_yield=cf["free_cash_flow"] / market_cap,
        dividend_yield=0.012 if profile["sector"] != "Technology" else 0.005,
    )


def build_earnings(profile: Dict, income_statements: List[Dict]) -> Dict:
    today = date.today()
    next_earnings = today + timedelta(days=45)
    last_earnings = today - timedelta(days=45)
    # Surprises
    surprises = [0.04, 0.03, 0.05, 0.02][: len(income_statements)]
    quarters = []
    for inc, surp in zip(income_statements, surprises):
        rev_q = inc["revenue"] / 4
        eps_q = inc["eps_diluted"] / 4 if inc.get("eps_diluted") else None
        quarters.append(dict(
            period=inc["period"],
            eps_actual=eps_q,
            eps_estimate=eps_q / (1 + surp) if eps_q else None,
            revenue_actual=rev_q,
            revenue_estimate=rev_q / (1 + surp / 2),
            surprise_pct=surp,
            price_reaction=surp * 2,
        ))
    return dict(
        last_earnings_date=last_earnings.isoformat(),
        next_earnings_date=next_earnings.isoformat(),
        quarters=quarters,
    )


def build_transcript(profile: Dict) -> Dict:
    drivers_text = ", ".join(profile["drivers"][:3])
    risks_text = ", ".join(profile["risks"][:2])
    return dict(
        ticker=None,
        period=str(_years(1)[-1]) + "Q4",
        date=(date.today() - timedelta(days=45)).isoformat(),
        speakers=["CEO", "CFO", "Analyst"],
        prepared_remarks=(
            f"Management opened by highlighting {drivers_text}. "
            f"Demand commentary remained constructive across {profile['segments'][0]} and {profile['segments'][-1]}. "
            f"Operating margin held at {profile['op_margin']:.0%}, with management reiterating the framework on capital allocation. "
            f"Guidance pointed to continued execution against {drivers_text}. "
            f"Capital priorities remain organic investment, dividends, and opportunistic buybacks."
        ),
        qa=(
            "Q (Analyst): Demand visibility into next year? "
            "A (CEO): Constructive signals; backlog supports the framework. "
            "Q (Analyst): Capital intensity? "
            "A (CFO): We expect capex to remain elevated as we invest behind durable returns. "
            f"Q (Analyst): Risk to the thesis? A (CEO): We continue to monitor {risks_text}."
        ),
        bullish_takeaways=[
            f"Management leaning into {profile['drivers'][0]}",
            f"Margin commentary supportive at {profile['op_margin']:.0%}",
            "Capital return remains a priority",
        ],
        bearish_takeaways=[
            f"Watch item: {profile['risks'][0]}",
            "Tone slightly more measured on near-term guidance versus prior quarter",
        ],
        management_tone="constructive",
    )


def build_filings(profile: Dict) -> List[Dict]:
    today = date.today()
    return [
        dict(
            type="10-K",
            period_end=str(_years(1)[-1]) + "-12-31",
            filing_date=(today - timedelta(days=120)).isoformat(),
            accession_number="0001-DEMO-10K",
            url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + profile.get("cik", ""),
            business_description=profile["description"],
            risk_factors=[
                f"Risk factor: {r}" for r in profile["risks"]
            ],
            mda=(
                f"Management discussed {profile['drivers'][0]} as the most important growth vector. "
                f"Operating margin moved consistent with {profile['op_margin']:.0%}. "
                "Liquidity remains strong; capital priorities are stable."
            ),
            segments=[
                dict(name=s, revenue_share=round(1 / len(profile["segments"]), 2)) for s in profile["segments"]
            ],
            legal_or_regulatory=[
                "Routine legal proceedings; no material disclosed exposure beyond filings.",
            ],
        ),
        dict(
            type="10-Q",
            period_end=str(_years(1)[-1]) + "-09-30",
            filing_date=(today - timedelta(days=200)).isoformat(),
            accession_number="0001-DEMO-10Q",
            url="https://www.sec.gov/",
            business_description=profile["description"],
            mda=(
                f"Quarterly results consistent with full-year framework, driven by {profile['drivers'][0]}. "
                f"Watch items include {profile['risks'][0]}."
            ),
        ),
        dict(
            type="8-K",
            period_end=str(_years(1)[-1]) + "-11-15",
            filing_date=(today - timedelta(days=160)).isoformat(),
            accession_number="0001-DEMO-8K",
            url="https://www.sec.gov/",
            description="Material announcement regarding capital allocation framework.",
        ),
    ]


def build_news(profile: Dict, ticker: str) -> List[Dict]:
    today = datetime.utcnow()
    items = [
        dict(
            title=f"{profile['company_name']} highlights progress on {profile['drivers'][0]}",
            source="MarketMosaic Demo Wire",
            published_at=(today - timedelta(days=2)).isoformat(),
            url="https://example.com/news/1",
            summary=f"{profile['company_name']} provided commentary on {profile['drivers'][0]}, reaffirming the broader framework.",
            tickers=[ticker],
            topics=[profile["sector"]],
            sentiment="positive",
            relevance_score=0.85,
        ),
        dict(
            title=f"Analyst notes on {ticker}: monitoring {profile['risks'][0]}",
            source="MarketMosaic Demo Wire",
            published_at=(today - timedelta(days=5)).isoformat(),
            url="https://example.com/news/2",
            summary=f"Sell-side commentary flagged {profile['risks'][0]} as the principal watch item for {ticker}.",
            tickers=[ticker],
            topics=[profile["sector"]],
            sentiment="neutral",
            relevance_score=0.65,
        ),
        dict(
            title=f"Sector update: {profile['industry']}",
            source="MarketMosaic Demo Wire",
            published_at=(today - timedelta(days=8)).isoformat(),
            url="https://example.com/news/3",
            summary=f"Industry trends across {profile['industry']} continue to be shaped by {profile['drivers'][0]}.",
            tickers=[ticker],
            topics=[profile["sector"]],
            sentiment="neutral",
            relevance_score=0.55,
        ),
    ]
    return items


def build_estimates(profile: Dict) -> Dict:
    return dict(
        revenue_estimates=[
            dict(period="FY+1", value=profile["revenue_b"] * 1e9 * (1 + profile["growth"])),
            dict(period="FY+2", value=profile["revenue_b"] * 1e9 * (1 + profile["growth"]) * (1 + profile["growth"] * 0.85)),
        ],
        eps_estimates=[],
        ebitda_estimates=[],
        price_targets=dict(median=profile["price"] * 1.10, high=profile["price"] * 1.30, low=profile["price"] * 0.85),
        recommendation_consensus="Buy / Hold blend (illustrative).",
    )


def build_macro_series() -> List[Dict]:
    today = date.today()
    series_specs = [
        ("FEDFUNDS", "Federal Funds Rate", 5.25, "%"),
        ("DGS2", "2-Year Treasury", 4.40, "%"),
        ("DGS10", "10-Year Treasury", 4.20, "%"),
        ("DGS30", "30-Year Treasury", 4.45, "%"),
        ("CPIAUCSL", "Headline CPI YoY", 2.9, "%"),
        ("CORESTICKM159SFRBATL", "Sticky Core CPI YoY", 3.6, "%"),
        ("PCEPI", "PCE Inflation YoY", 2.5, "%"),
        ("UNRATE", "Unemployment Rate", 4.1, "%"),
        ("PAYEMS", "Nonfarm Payrolls (k)", 200, "thousand"),
        ("GDPC1", "Real GDP YoY", 2.5, "%"),
        ("RSAFS", "Retail Sales YoY", 2.8, "%"),
        ("BAMLH0A0HYM2", "High-Yield Spread", 3.4, "%"),
        ("DCOILWTICO", "Oil Price (WTI)", 78, "$/bbl"),
    ]
    series = []
    for sid, name, latest, units in series_specs:
        points = []
        for i in range(24):
            d = today - timedelta(days=30 * (24 - i - 1))
            jitter = math.sin(i * 0.7 + len(sid)) * 0.3
            value = max(0.1, latest + jitter * (latest * 0.05 if latest > 5 else 0.3))
            points.append(dict(date=d.isoformat(), value=round(value, 3)))
        # ensure last value matches "latest"
        if points:
            points[-1]["value"] = latest
        series.append(dict(series_id=sid, name=name, units=units, points=points))
    return series


# ---------------------------------------------------------------------------
# Top-level dataset assembly
# ---------------------------------------------------------------------------

def build_dataset() -> Dict[str, Dict]:
    """Build the full dataset keyed by ticker, plus macro under '_macro'."""
    dataset: Dict[str, Dict] = {}
    for ticker, profile in COMPANY_PROFILES.items():
        income = build_income_statements(profile)
        cash = build_cash_flows(profile, income)
        balance = build_balance_sheets(profile, income)
        ratios = build_ratios(profile, income, balance, cash)
        prices = build_price_history(profile)
        earnings = build_earnings(profile, income)
        transcript = build_transcript(profile)
        transcript["ticker"] = ticker
        filings = build_filings(profile)
        news = build_news(profile, ticker)
        estimates = build_estimates(profile)
        dataset[ticker] = dict(
            profile=dict(
                ticker=ticker,
                company_name=profile["company_name"],
                exchange=profile["exchange"],
                sector=profile["sector"],
                industry=profile["industry"],
                sub_industry=profile.get("sub_industry"),
                country="US",
                currency="USD",
                market_cap=profile["market_cap_b"] * 1e9,
                cik=profile.get("cik"),
                business_description=profile["description"],
                fiscal_year_end=profile.get("fye"),
                is_active=True,
                is_etf=False,
                beta=profile["beta"],
                shares_outstanding=profile["shares_b"] * 1e9,
                last_price=profile["price"],
                segments=profile["segments"],
                drivers=profile["drivers"],
                risks=profile["risks"],
            ),
            income_statements=income,
            balance_sheets=balance,
            cash_flows=cash,
            ratios=ratios,
            prices=prices,
            earnings=earnings,
            transcripts=[transcript],
            filings=filings,
            news=news,
            estimates=estimates,
        )
    dataset["_macro"] = dict(series=build_macro_series())
    return dataset


def export_to_disk(out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Dump split JSON files to backend/app/data/ for inspection / shipping."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent
    dataset = build_dataset()
    files = {
        "demo_companies.json": {t: d["profile"] for t, d in dataset.items() if t != "_macro"},
        "demo_financials.json": {
            t: dict(income=d["income_statements"], balance=d["balance_sheets"], cash=d["cash_flows"], ratios=d["ratios"])
            for t, d in dataset.items() if t != "_macro"
        },
        "demo_prices.json": {t: d["prices"] for t, d in dataset.items() if t != "_macro"},
        "demo_transcripts.json": {t: d["transcripts"] for t, d in dataset.items() if t != "_macro"},
        "demo_filings.json": {t: d["filings"] for t, d in dataset.items() if t != "_macro"},
        "demo_news.json": {t: d["news"] for t, d in dataset.items() if t != "_macro"},
        "demo_earnings.json": {t: d["earnings"] for t, d in dataset.items() if t != "_macro"},
        "demo_estimates.json": {t: d["estimates"] for t, d in dataset.items() if t != "_macro"},
        "demo_macro.json": dataset.get("_macro", {}),
    }
    written: Dict[str, Path] = {}
    for fname, payload in files.items():
        path = out_dir / fname
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        written[fname] = path
    return written


if __name__ == "__main__":
    paths = export_to_disk()
    for k, v in paths.items():
        print(f"wrote {v}")
