"""Curated series registry.

Each `SeriesSpec` describes one fetchable time series — what it measures,
who publishes it, how often, which sectors care about it, and which
region it covers. The sector analyst reads this registry to decide
"for THIS ticker, what data should I look at?" before any fetch happens.

Adding a series means: (a) append to `SERIES_REGISTRY` with descriptive
metadata, (b) make sure the corresponding provider can fetch by that
`series_id`. No agent-prompt edits needed — the catalog is the contract.

Conventions:
    source       - "FRED" | "EIA" | "BLS" | "Census" (matches provider.name uppercased)
    category     - high-level bucket: rates | inflation | labor | housing |
                   real_estate | retail | energy | manufacturing | macro_growth |
                   credit | sentiment | demographics
    region       - "US" for national; "metro:<code>" for metro-level;
                   "state:<XX>" for state-level; "global" when applicable
    sector_tags  - list of GICS sectors most likely to care
                   ("Real Estate", "Consumer Discretionary", "Energy",
                    "Financials", "Utilities", "Industrials",
                    "Consumer Staples", "Communication Services",
                    "Health Care", "Information Technology", "Materials")
    sub_industry_tags - more granular hints when relevant
                   (e.g. "Retail REITs", "Oil & Gas E&P", "Auto Manufacturers")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    name: str
    source: str
    category: str
    units: str
    frequency: str           # daily | weekly | monthly | quarterly | annual
    description: str
    sector_tags: tuple[str, ...] = field(default_factory=tuple)
    sub_industry_tags: tuple[str, ...] = field(default_factory=tuple)
    region: str = "US"
    documentation_url: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "series_id": self.series_id,
            "name": self.name,
            "source": self.source,
            "category": self.category,
            "units": self.units,
            "frequency": self.frequency,
            "description": self.description,
            "sector_tags": list(self.sector_tags),
            "sub_industry_tags": list(self.sub_industry_tags),
            "region": self.region,
            "documentation_url": self.documentation_url,
        }


# ---------------------------------------------------------------------------
# FRED — core macro + rates + inflation (extends the original 13 series)
# ---------------------------------------------------------------------------

_FRED_CORE: tuple[SeriesSpec, ...] = (
    SeriesSpec("FEDFUNDS", "Federal Funds Rate", "FRED", "rates", "%", "monthly",
               "Effective overnight policy rate. Drives every discount-rate-sensitive asset.",
               sector_tags=("Financials", "Real Estate", "Utilities", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/FEDFUNDS"),
    SeriesSpec("DGS2", "2-Year Treasury", "FRED", "rates", "%", "daily",
               "Front-end of the curve; tracks Fed expectations 18-24 months out.",
               sector_tags=("Financials",),
               documentation_url="https://fred.stlouisfed.org/series/DGS2"),
    SeriesSpec("DGS10", "10-Year Treasury", "FRED", "rates", "%", "daily",
               "Long-duration discount rate; the headline yield for equity risk premium.",
               sector_tags=("Real Estate", "Utilities", "Financials", "Information Technology"),
               documentation_url="https://fred.stlouisfed.org/series/DGS10"),
    SeriesSpec("DGS30", "30-Year Treasury", "FRED", "rates", "%", "daily",
               "Long bond yield; mortgage-rate anchor.",
               sector_tags=("Real Estate", "Financials"),
               documentation_url="https://fred.stlouisfed.org/series/DGS30"),
    SeriesSpec("T10Y2Y", "10Y-2Y Term Spread", "FRED", "rates", "%", "daily",
               "Yield curve slope. Inverted curve historically precedes recessions.",
               sector_tags=("Financials",),
               documentation_url="https://fred.stlouisfed.org/series/T10Y2Y"),
    SeriesSpec("BAMLH0A0HYM2", "High-Yield Credit Spread", "FRED", "credit", "%", "daily",
               "ICE BofA US High Yield Index OAS. Widening signals stress in junk credit.",
               sector_tags=("Financials", "Energy", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/BAMLH0A0HYM2"),
    SeriesSpec("CPIAUCSL", "Headline CPI", "FRED", "inflation", "index", "monthly",
               "Consumer Price Index All Urban Consumers. Top-line inflation print.",
               sector_tags=("Consumer Staples", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/CPIAUCSL"),
    SeriesSpec("CORESTICKM159SFRBATL", "Sticky Core CPI YoY", "FRED", "inflation", "%", "monthly",
               "Atlanta Fed sticky-price CPI ex food and energy YoY. The Fed's preferred inflation persistence gauge.",
               documentation_url="https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL"),
    SeriesSpec("PCEPI", "PCE Price Index", "FRED", "inflation", "index", "monthly",
               "Personal Consumption Expenditures price index. The FOMC's preferred inflation measure.",
               documentation_url="https://fred.stlouisfed.org/series/PCEPI"),
    SeriesSpec("UNRATE", "Unemployment Rate", "FRED", "labor", "%", "monthly",
               "Headline U-3 unemployment rate.",
               sector_tags=("Consumer Discretionary", "Financials"),
               documentation_url="https://fred.stlouisfed.org/series/UNRATE"),
    SeriesSpec("PAYEMS", "Nonfarm Payrolls", "FRED", "labor", "thousands", "monthly",
               "All-employees total nonfarm payroll level.",
               documentation_url="https://fred.stlouisfed.org/series/PAYEMS"),
    SeriesSpec("CES4300000001", "Transportation & Warehousing Employment", "FRED", "labor",
               "thousands", "monthly",
               "Truck transport, warehousing, courier headcount. Leading indicator for goods volumes.",
               sector_tags=("Industrials", "Consumer Discretionary"),
               sub_industry_tags=("Trucking", "Air Freight", "Marine Ports"),
               documentation_url="https://fred.stlouisfed.org/series/CES4300000001"),
    SeriesSpec("GDPC1", "Real GDP", "FRED", "macro_growth", "billions $", "quarterly",
               "Chained-dollars real GDP.",
               documentation_url="https://fred.stlouisfed.org/series/GDPC1"),
    SeriesSpec("UMCSENT", "U Michigan Consumer Sentiment", "FRED", "sentiment", "index",
               "monthly",
               "University of Michigan consumer sentiment headline index.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://fred.stlouisfed.org/series/UMCSENT"),
    SeriesSpec("INDPRO", "Industrial Production Index", "FRED", "manufacturing", "index",
               "monthly",
               "Total industrial production. Cyclical manufacturing barometer.",
               sector_tags=("Industrials", "Materials", "Energy"),
               documentation_url="https://fred.stlouisfed.org/series/INDPRO"),
    SeriesSpec("ISRATIO", "Inventories to Sales Ratio", "FRED", "manufacturing", "ratio",
               "monthly",
               "Manufacturers & trade inventories to sales. Rising = demand cooling vs supply.",
               sector_tags=("Industrials", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/ISRATIO"),
)

# ---------------------------------------------------------------------------
# FRED — housing & real estate (Case-Shiller, mortgage rates, starts, rents)
# ---------------------------------------------------------------------------

_FRED_HOUSING: tuple[SeriesSpec, ...] = (
    SeriesSpec("CSUSHPISA", "Case-Shiller National Home Price Index", "FRED",
               "real_estate", "index", "monthly",
               "S&P/Case-Shiller US National Home Price seasonally-adjusted index. The headline US house-price benchmark.",
               sector_tags=("Real Estate", "Financials", "Consumer Discretionary"),
               sub_industry_tags=("Residential REITs", "Homebuilders", "Mortgage Finance"),
               documentation_url="https://fred.stlouisfed.org/series/CSUSHPISA"),
    SeriesSpec("MORTGAGE30US", "30-Year Fixed Mortgage Rate", "FRED", "real_estate", "%",
               "weekly",
               "Freddie Mac weekly survey of 30-year fixed mortgage rates. Drives housing affordability.",
               sector_tags=("Real Estate", "Financials"),
               sub_industry_tags=("Homebuilders", "Mortgage Finance", "Residential REITs"),
               documentation_url="https://fred.stlouisfed.org/series/MORTGAGE30US"),
    SeriesSpec("HOUST", "Housing Starts", "FRED", "real_estate", "thousands", "monthly",
               "Privately-owned housing starts (annual rate). Leading housing-cycle indicator.",
               sector_tags=("Real Estate", "Materials", "Industrials"),
               sub_industry_tags=("Homebuilders", "Building Products"),
               documentation_url="https://fred.stlouisfed.org/series/HOUST"),
    SeriesSpec("PERMIT", "Building Permits", "FRED", "real_estate", "thousands", "monthly",
               "New private housing units authorized by building permits. Leads starts by ~1 month.",
               sector_tags=("Real Estate", "Materials"),
               sub_industry_tags=("Homebuilders",),
               documentation_url="https://fred.stlouisfed.org/series/PERMIT"),
    SeriesSpec("EXHOSLUSM495S", "Existing Home Sales", "FRED", "real_estate", "thousands",
               "monthly",
               "NAR existing single-family home sales annualized rate.",
               sector_tags=("Real Estate", "Financials"),
               documentation_url="https://fred.stlouisfed.org/series/EXHOSLUSM495S"),
    SeriesSpec("RRVRUSQ156N", "Rental Vacancy Rate", "FRED", "real_estate", "%",
               "quarterly",
               "Census Bureau rental vacancy rate. Inverse measure of apartment REIT pricing power.",
               sector_tags=("Real Estate",),
               sub_industry_tags=("Apartment REITs", "Residential REITs"),
               documentation_url="https://fred.stlouisfed.org/series/RRVRUSQ156N"),
    SeriesSpec("CUUR0000SEHA", "CPI: Rent of Primary Residence", "FRED",
               "real_estate", "index", "monthly",
               "Shelter inflation component. Lags spot rents by ~12 months.",
               sector_tags=("Real Estate",),
               sub_industry_tags=("Apartment REITs", "Residential REITs"),
               documentation_url="https://fred.stlouisfed.org/series/CUUR0000SEHA"),
    SeriesSpec("MSPUS", "Median Sales Price of Houses Sold", "FRED", "real_estate", "$",
               "quarterly",
               "Census Bureau median sales price for new houses sold in the United States.",
               sector_tags=("Real Estate", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/MSPUS"),
    # Metro-level Case-Shiller (subset — add metros as needed)
    SeriesSpec("SFXRSA", "Case-Shiller San Francisco HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller San Francisco metro home price index.",
               sector_tags=("Real Estate",), region="metro:SF",
               documentation_url="https://fred.stlouisfed.org/series/SFXRSA"),
    SeriesSpec("NYXRSA", "Case-Shiller New York HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller New York metro home price index.",
               sector_tags=("Real Estate",), region="metro:NYC",
               documentation_url="https://fred.stlouisfed.org/series/NYXRSA"),
    SeriesSpec("LXXRSA", "Case-Shiller Los Angeles HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Los Angeles metro home price index.",
               sector_tags=("Real Estate",), region="metro:LA",
               documentation_url="https://fred.stlouisfed.org/series/LXXRSA"),
    SeriesSpec("ATXRSA", "Case-Shiller Atlanta HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Atlanta metro home price index.",
               sector_tags=("Real Estate",), region="metro:ATL",
               documentation_url="https://fred.stlouisfed.org/series/ATXRSA"),
    SeriesSpec("DAXRSA", "Case-Shiller Dallas HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Dallas metro home price index.",
               sector_tags=("Real Estate",), region="metro:DAL",
               documentation_url="https://fred.stlouisfed.org/series/DAXRSA"),
    SeriesSpec("PHXRSA", "Case-Shiller Phoenix HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Phoenix metro home price index.",
               sector_tags=("Real Estate",), region="metro:PHX",
               documentation_url="https://fred.stlouisfed.org/series/PHXRSA"),
    SeriesSpec("MIXRSA", "Case-Shiller Miami HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Miami metro home price index.",
               sector_tags=("Real Estate",), region="metro:MIA",
               documentation_url="https://fred.stlouisfed.org/series/MIXRSA"),
    SeriesSpec("CHXRSA", "Case-Shiller Chicago HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Chicago metro home price index.",
               sector_tags=("Real Estate",), region="metro:CHI",
               documentation_url="https://fred.stlouisfed.org/series/CHXRSA"),
    SeriesSpec("BOXRSA", "Case-Shiller Boston HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Boston metro home price index.",
               sector_tags=("Real Estate",), region="metro:BOS",
               documentation_url="https://fred.stlouisfed.org/series/BOXRSA"),
    SeriesSpec("SEXRSA", "Case-Shiller Seattle HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Seattle metro home price index.",
               sector_tags=("Real Estate",), region="metro:SEA",
               documentation_url="https://fred.stlouisfed.org/series/SEXRSA"),
    SeriesSpec("WDXRSA", "Case-Shiller Washington DC HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Washington DC metro home price index.",
               sector_tags=("Real Estate",), region="metro:DCA",
               documentation_url="https://fred.stlouisfed.org/series/WDXRSA"),
    SeriesSpec("DNXRSA", "Case-Shiller Denver HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Denver metro home price index.",
               sector_tags=("Real Estate",), region="metro:DEN",
               documentation_url="https://fred.stlouisfed.org/series/DNXRSA"),
    SeriesSpec("MNXRSA", "Case-Shiller Minneapolis HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Minneapolis metro home price index.",
               sector_tags=("Real Estate",), region="metro:MIN",
               documentation_url="https://fred.stlouisfed.org/series/MNXRSA"),
    SeriesSpec("POXRSA", "Case-Shiller Portland HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Portland metro home price index.",
               sector_tags=("Real Estate",), region="metro:POR",
               documentation_url="https://fred.stlouisfed.org/series/POXRSA"),
    SeriesSpec("LVXRSA", "Case-Shiller Las Vegas HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Las Vegas metro home price index.",
               sector_tags=("Real Estate",), region="metro:LAS",
               documentation_url="https://fred.stlouisfed.org/series/LVXRSA"),
    SeriesSpec("DEXRSA", "Case-Shiller Detroit HPI", "FRED", "real_estate", "index",
               "monthly", "Case-Shiller Detroit metro home price index.",
               sector_tags=("Real Estate",), region="metro:DET",
               documentation_url="https://fred.stlouisfed.org/series/DEXRSA"),
)

# ---------------------------------------------------------------------------
# FRED — retail & consumer
# ---------------------------------------------------------------------------

_FRED_RETAIL: tuple[SeriesSpec, ...] = (
    SeriesSpec("RSAFS", "Advance Retail Sales", "FRED", "retail", "millions $",
               "monthly",
               "Advance monthly retail and food-services sales. Headline consumer-spending print.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://fred.stlouisfed.org/series/RSAFS"),
    SeriesSpec("RRSFS", "Real Retail Sales", "FRED", "retail", "millions $", "monthly",
               "Inflation-adjusted retail and food services sales.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://fred.stlouisfed.org/series/RRSFS"),
    SeriesSpec("RSXFS", "Retail Sales ex Food Services", "FRED", "retail", "millions $",
               "monthly", "Retail trade excluding food services. Closer goods-only print.",
               sector_tags=("Consumer Discretionary",),
               documentation_url="https://fred.stlouisfed.org/series/RSXFS"),
    SeriesSpec("DSPIC96", "Real Disposable Personal Income", "FRED", "retail",
               "billions $", "monthly",
               "Real disposable personal income — the wallet behind consumption.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://fred.stlouisfed.org/series/DSPIC96"),
    SeriesSpec("PSAVERT", "Personal Saving Rate", "FRED", "retail", "%", "monthly",
               "Personal saving as % of disposable income. Falling rate signals consumption pulled forward.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://fred.stlouisfed.org/series/PSAVERT"),
    SeriesSpec("TOTALSL", "Total Consumer Credit Outstanding", "FRED", "credit",
               "billions $", "monthly",
               "Total consumer credit. Reading on household leverage cycle.",
               sector_tags=("Financials", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/TOTALSL"),
    SeriesSpec("DRCCLACBS", "Credit Card Delinquency Rate", "FRED", "credit", "%",
               "quarterly",
               "Delinquency rate on consumer credit card loans at commercial banks.",
               sector_tags=("Financials",),
               sub_industry_tags=("Consumer Finance",),
               documentation_url="https://fred.stlouisfed.org/series/DRCCLACBS"),
    SeriesSpec("VMTD11", "Vehicle Miles Traveled", "FRED", "retail", "millions",
               "monthly",
               "Total US vehicle miles. Leading proxy for fuel demand and travel.",
               sector_tags=("Energy", "Consumer Discretionary"),
               documentation_url="https://fred.stlouisfed.org/series/VMTD11"),
)

# ---------------------------------------------------------------------------
# EIA — energy supply & demand
# ---------------------------------------------------------------------------

_EIA_ENERGY: tuple[SeriesSpec, ...] = (
    SeriesSpec("PET.WCESTUS1.W", "US Crude Oil Inventories", "EIA", "energy",
               "thousand barrels", "weekly",
               "Weekly US crude oil ending stocks ex SPR. Surprise builds pressure WTI; draws bullish.",
               sector_tags=("Energy",),
               sub_industry_tags=("Oil & Gas E&P", "Oil & Gas Refining"),
               documentation_url="https://www.eia.gov/dnav/pet/pet_stoc_wstk_dcu_nus_w.htm"),
    SeriesSpec("PET.WGTSTUS1.W", "US Gasoline Inventories", "EIA", "energy",
               "thousand barrels", "weekly",
               "Weekly US gasoline ending stocks. Driving-season demand barometer.",
               sector_tags=("Energy", "Consumer Discretionary"),
               documentation_url="https://www.eia.gov/petroleum/weekly/gasoline.php"),
    SeriesSpec("PET.WDISTUS1.W", "US Distillate Inventories", "EIA", "energy",
               "thousand barrels", "weekly",
               "Weekly US distillate (diesel/heating oil) stocks. Industrial + freight demand proxy.",
               sector_tags=("Energy", "Industrials"),
               documentation_url="https://www.eia.gov/petroleum/weekly/distillate.php"),
    SeriesSpec("NG.NW2_EPG0_SWO_R48_BCF.W", "US Natural Gas Storage", "EIA",
               "energy", "Bcf", "weekly",
               "Weekly working gas in underground storage (Lower 48). Winter-draw tracker.",
               sector_tags=("Energy", "Utilities"),
               sub_industry_tags=("Oil & Gas E&P", "Natural Gas Utilities"),
               documentation_url="https://www.eia.gov/naturalgas/weekly/"),
    SeriesSpec("PET.RWTC.D", "WTI Crude Oil Price", "EIA", "energy", "$/bbl",
               "daily", "Cushing OK WTI spot crude price.",
               sector_tags=("Energy",),
               documentation_url="https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm"),
    SeriesSpec("NG.RNGWHHD.D", "Henry Hub Natural Gas Price", "EIA", "energy",
               "$/MMBtu", "daily", "Henry Hub natural gas spot price.",
               sector_tags=("Energy", "Utilities"),
               documentation_url="https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm"),
    SeriesSpec("ELEC.GEN.ALL-US-99.M", "US Total Electricity Generation",
               "EIA", "energy", "GWh", "monthly",
               "Total US net electricity generation, all fuel sources. Demand barometer for utilities.",
               sector_tags=("Utilities", "Industrials"),
               documentation_url="https://www.eia.gov/electricity/data/browser/"),
)

# ---------------------------------------------------------------------------
# BLS — CPI components, regional employment, productivity
# ---------------------------------------------------------------------------

_BLS_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("CUUR0000SA0", "BLS Headline CPI", "BLS", "inflation", "index",
               "monthly", "BLS Consumer Price Index for All Urban Consumers, all items.",
               documentation_url="https://www.bls.gov/cpi/"),
    SeriesSpec("CUUR0000SAF1", "BLS Food at Home CPI", "BLS", "inflation", "index",
               "monthly",
               "Grocery price component of CPI. Direct margin pressure on supermarkets.",
               sector_tags=("Consumer Staples",),
               sub_industry_tags=("Food Retail", "Packaged Foods"),
               documentation_url="https://www.bls.gov/cpi/"),
    SeriesSpec("CUUR0000SETA01", "BLS New Vehicle CPI", "BLS", "inflation", "index",
               "monthly",
               "New car price component. Pricing power for OEMs and dealers.",
               sector_tags=("Consumer Discretionary",),
               sub_industry_tags=("Auto Manufacturers", "Auto Retail"),
               documentation_url="https://www.bls.gov/cpi/"),
    SeriesSpec("CUUR0000SETA02", "BLS Used Vehicle CPI", "BLS", "inflation", "index",
               "monthly",
               "Used vehicle price component. Pricing for CarMax, Carvana, AutoNation used books.",
               sector_tags=("Consumer Discretionary", "Financials"),
               sub_industry_tags=("Auto Retail", "Consumer Finance"),
               documentation_url="https://www.bls.gov/cpi/"),
    SeriesSpec("CUUR0000SAM", "BLS Medical Care CPI", "BLS", "inflation", "index",
               "monthly",
               "Medical care services + commodities. Hospital and pharma pricing pass-through.",
               sector_tags=("Health Care",),
               documentation_url="https://www.bls.gov/cpi/"),
    SeriesSpec("CUUR0000SETB01", "BLS Gasoline CPI", "BLS", "inflation", "index",
               "monthly", "Retail gasoline price index.",
               sector_tags=("Energy", "Consumer Discretionary"),
               documentation_url="https://www.bls.gov/cpi/"),
    SeriesSpec("CES4348400001", "BLS Trucking Employment", "BLS", "labor",
               "thousands", "monthly",
               "Truck transportation industry employment. Leading goods-volume indicator.",
               sector_tags=("Industrials", "Consumer Discretionary"),
               sub_industry_tags=("Trucking",),
               documentation_url="https://www.bls.gov/ces/"),
    SeriesSpec("CES6562000101", "BLS Hospital Employment", "BLS", "labor",
               "thousands", "monthly",
               "Hospital industry employment. Labor-cost pressure on hospital operators.",
               sector_tags=("Health Care",),
               sub_industry_tags=("Healthcare Facilities",),
               documentation_url="https://www.bls.gov/ces/"),
    SeriesSpec("CES4244000001", "BLS Retail Trade Employment", "BLS", "labor",
               "thousands", "monthly",
               "Retail trade industry employment headcount.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://www.bls.gov/ces/"),
    SeriesSpec("CES7072000001", "BLS Restaurant Employment", "BLS", "labor",
               "thousands", "monthly",
               "Food services and drinking places employment.",
               sector_tags=("Consumer Discretionary",),
               sub_industry_tags=("Restaurants",),
               documentation_url="https://www.bls.gov/ces/"),
    SeriesSpec("LAUST060000000000003", "California Unemployment Rate", "BLS",
               "labor", "%", "monthly",
               "California state unemployment rate. Big-state economic health proxy.",
               region="state:CA",
               documentation_url="https://www.bls.gov/lau/"),
    SeriesSpec("LAUST480000000000003", "Texas Unemployment Rate", "BLS",
               "labor", "%", "monthly",
               "Texas state unemployment rate.",
               region="state:TX",
               documentation_url="https://www.bls.gov/lau/"),
    SeriesSpec("LAUST360000000000003", "New York Unemployment Rate", "BLS",
               "labor", "%", "monthly",
               "New York state unemployment rate.",
               region="state:NY",
               documentation_url="https://www.bls.gov/lau/"),
    SeriesSpec("LAUST120000000000003", "Florida Unemployment Rate", "BLS",
               "labor", "%", "monthly",
               "Florida state unemployment rate.",
               region="state:FL",
               documentation_url="https://www.bls.gov/lau/"),
)

# ---------------------------------------------------------------------------
# Census — retail trade, construction, e-commerce
# ---------------------------------------------------------------------------

_CENSUS_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("MARTS_44X72", "Census Retail & Food Services Sales", "Census",
               "retail", "millions $", "monthly",
               "Monthly Advance Retail Trade Survey: total retail and food services.",
               sector_tags=("Consumer Discretionary", "Consumer Staples"),
               documentation_url="https://www.census.gov/retail/marts/"),
    SeriesSpec("MARTS_445", "Census Food & Beverage Store Sales", "Census",
               "retail", "millions $", "monthly",
               "Monthly retail sales for food and beverage stores (NAICS 445).",
               sector_tags=("Consumer Staples",),
               sub_industry_tags=("Food Retail",),
               documentation_url="https://www.census.gov/retail/marts/"),
    SeriesSpec("MARTS_448", "Census Clothing & Accessories Store Sales",
               "Census", "retail", "millions $", "monthly",
               "Monthly retail sales for clothing and clothing accessories stores (NAICS 448).",
               sector_tags=("Consumer Discretionary",),
               sub_industry_tags=("Apparel Retail", "Department Stores"),
               documentation_url="https://www.census.gov/retail/marts/"),
    SeriesSpec("MARTS_454", "Census Nonstore (E-Commerce) Sales", "Census",
               "retail", "millions $", "monthly",
               "Monthly retail sales for nonstore retailers (NAICS 454) — largely e-commerce.",
               sector_tags=("Consumer Discretionary",),
               sub_industry_tags=("Internet & Direct Marketing Retail",),
               documentation_url="https://www.census.gov/retail/marts/"),
    SeriesSpec("MARTS_447", "Census Gasoline Station Sales", "Census", "retail",
               "millions $", "monthly",
               "Monthly retail sales for gasoline stations (NAICS 447).",
               sector_tags=("Energy", "Consumer Staples"),
               sub_industry_tags=("Oil & Gas Refining",),
               documentation_url="https://www.census.gov/retail/marts/"),
    SeriesSpec("MARTS_722", "Census Food Services & Drinking Places Sales",
               "Census", "retail", "millions $", "monthly",
               "Monthly sales for food services and drinking places (NAICS 722).",
               sector_tags=("Consumer Discretionary",),
               sub_industry_tags=("Restaurants",),
               documentation_url="https://www.census.gov/retail/marts/"),
    SeriesSpec("RESCONST_TOTAL", "Census Residential Construction Spending",
               "Census", "real_estate", "millions $", "monthly",
               "Total private residential construction put-in-place.",
               sector_tags=("Real Estate", "Industrials", "Materials"),
               sub_industry_tags=("Homebuilders", "Building Products"),
               documentation_url="https://www.census.gov/construction/c30/"),
    SeriesSpec("NONRESCONST_TOTAL", "Census Nonresidential Construction Spending",
               "Census", "real_estate", "millions $", "monthly",
               "Total private nonresidential construction put-in-place (offices, warehouses, retail).",
               sector_tags=("Real Estate", "Industrials"),
               sub_industry_tags=("Office REITs", "Industrial REITs"),
               documentation_url="https://www.census.gov/construction/c30/"),
)


SERIES_REGISTRY: tuple[SeriesSpec, ...] = (
    *_FRED_CORE,
    *_FRED_HOUSING,
    *_FRED_RETAIL,
    *_EIA_ENERGY,
    *_BLS_SERIES,
    *_CENSUS_SERIES,
)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

_BY_ID: dict[str, SeriesSpec] = {s.series_id: s for s in SERIES_REGISTRY}


def by_id(series_id: str) -> Optional[SeriesSpec]:
    return _BY_ID.get(series_id)


def by_category(category: str) -> List[SeriesSpec]:
    cat = category.lower()
    return [s for s in SERIES_REGISTRY if s.category == cat]


def by_sector_tag(sector: str) -> List[SeriesSpec]:
    """Return series whose `sector_tags` include the given GICS sector."""
    if not sector:
        return []
    target = sector.strip()
    return [s for s in SERIES_REGISTRY if target in s.sector_tags]


def by_sub_industry_tag(sub_industry: str) -> List[SeriesSpec]:
    """Return series tagged for a specific sub-industry."""
    if not sub_industry:
        return []
    target = sub_industry.strip()
    return [s for s in SERIES_REGISTRY if target in s.sub_industry_tags]


def list_categories() -> List[str]:
    return sorted({s.category for s in SERIES_REGISTRY})


def list_regions() -> List[str]:
    return sorted({s.region for s in SERIES_REGISTRY})


def list_sector_tags() -> List[str]:
    tags: set[str] = set()
    for s in SERIES_REGISTRY:
        tags.update(s.sector_tags)
    return sorted(tags)


def search(
    *,
    sector: Optional[str] = None,
    sub_industry: Optional[str] = None,
    categories: Optional[Iterable[str]] = None,
    region: Optional[str] = None,
    sources: Optional[Iterable[str]] = None,
    keywords: Optional[Iterable[str]] = None,
) -> List[SeriesSpec]:
    """Filter the registry by any combination of axes.

    All filters are AND-combined. Keyword match is case-insensitive and
    looks at name + description.
    """
    cats = {c.lower() for c in categories} if categories else None
    srcs = {s.upper() for s in sources} if sources else None
    kws = [k.lower() for k in keywords] if keywords else None
    results: List[SeriesSpec] = []
    for spec in SERIES_REGISTRY:
        if sector and sector not in spec.sector_tags:
            continue
        if sub_industry and sub_industry not in spec.sub_industry_tags:
            continue
        if cats and spec.category not in cats:
            continue
        if region and spec.region != region:
            continue
        if srcs and spec.source.upper() not in srcs:
            continue
        if kws:
            haystack = f"{spec.name} {spec.description}".lower()
            if not all(kw in haystack for kw in kws):
                continue
        results.append(spec)
    return results
