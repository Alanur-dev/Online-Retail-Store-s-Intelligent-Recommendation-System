# Online-Retail-Store-s-Intelligent-Recommendation-System
Driving Revenue Growth and Risk Control through  Product Recommendations


A dual-lens intelligent recommendation system built on the [UCI Online Retail II dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci). The system combines Market Basket Analysis, Collaborative Filtering, and Cancellation Analysis to increase revenue and reduce operational loss — delivered through both a customer-facing recommendation layer and a stakeholder-facing decision support dashboard.

---

## Business Context

| Metric | Value |
|--------|-------|
| Revenue | £8.9M |
| Transactions | 1.07M |
| Customers | 5.9k |
| Countries | 43 |

**The problem:** Revenue is concentrated in a small group of customers and products. There are no personalised recommendations, product relationships are underutilised (bundling opportunity), and revenue is partially lost to cancellations. Top 1,352 customers drive the majority of sales.

**The opportunity:** Leverage Machine Learning to increase revenue and reduce loss.

---

## Solution Overview — Dual-Lens System

```
Intelligent Recommendation System
│
├── Customer Lens — Product Recommendations
│   ├── Product bundling & cross-sell
│   ├── Website recommendations
│   ├── Checkout suggestions
│   └── Email personalisation
│
└── Stakeholder Lens — Decision Support Tools
    ├── Interactive Dashboard
    └── ML integration (MBA + CF + Cancellation)
```

---

## Dataset

- **Source:** [UCI Online Retail II — Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- **Period:** December 2009 – December 2011
- **Rows:** 1,067,371
- **Transactions:** 53,628
- **Products:** 5,699
- **Customers:** 5,942
- **Countries:** 43

### Data Cleaning

| Issue | Decision |
|-------|----------|
| Missing `Customer_ID` (~23%) | Kept for basket analysis; excluded from collaborative filtering |
| Missing `Description` (~4.4k rows) | Dropped to ensure data quality |
| Returns / zero-price transfers | Removed |
| Cancellation invoices (prefix `C`) | Extracted separately for risk analysis, then excluded from main models |
| International orders | Basket analysis scoped to UK only for behavioural homogeneity |

### Feature Engineering
- `Revenue` derived as `Quantity × Price`
- Data aggregated at three levels: transaction, customer, and product
- Supports both basket analysis and collaborative filtering pipelines

---

## Models & Analysis

### 1. Market Basket Analysis (MBA)
*Which products are commonly bought together?*

Uses the **Apriori algorithm** on transactional data to identify frequently co-occurring product combinations, generating association rules of the form `{Product A} → {Product B}`.

**Metrics:**
- **Support** — frequency of itemset occurrence across total transactions
- **Confidence** — probability of purchasing B given A is purchased
- **Lift** — strength of association compared to random co-occurrence

**Results:**
- High lift values (>10×) confirm associations are non-random and meaningful
- Patterns show customers buying product variants or shopping within a theme
- Strong confidence values indicate consistent, reliable product pairings

**Strategies enabled:**

| Strategy | Example | Expected Impact |
|----------|---------|-----------------|
| Bundle & Set | Regency Teacup Set (Pink + Green + Roses) | AOV increase of 10–25% |
| Cross-Sell | Suggest Strawberry Box after Sweetheart Box | Basket size +8–14%, conversion +5–10% |
| Volume Promotion | Red & White Hanging Heart T-Light Holders | Repeat purchase rate +5–8% |

---

### 2. Collaborative Filtering (CF)
*What should each customer buy next?*

**Item-Based Collaborative Filtering** identifies products that are similar based on shared customer purchasing behaviour. A customer–product matrix is constructed, L2-normalised to reduce high-volume buyer bias, and cosine similarity is computed between product pairs to generate personalised recommendations.

**Metrics:**
- **Similarity Score (Cosine)** — measures how similar two products are based on purchase patterns
- **Coverage** — percentage of customers receiving at least one recommendation
- **Diversity** — variety of products recommended

**Results:**

| Metric | Value |
|--------|-------|
| Customer coverage | 99.5% |
| Max similarity score | 0.982 |
| Agreement with MBA | 90% |
| Catalogue diversity | 12.9% |

The 90% agreement with MBA (two independent methods — basket-level and customer-level) validates that the associations reflect real purchasing behaviour, not statistical artefacts.

**Strategies enabled:**

| Strategy | Example | Expected Impact |
|----------|---------|-----------------|
| Personalised Recommendations | Items based on individual purchase history | Conversion rate +5–10% |
| Targeted Campaigns | Personalised email promotions | Improved customer retention |
| Product Discovery | "You may also like" sections | Increased product exposure |

---

### 3. Cancellation Analysis
*Where are the key cancellation risks in the business?*

Examines patterns in cancelled transactions to identify high-risk products, customers, and time periods, enabling proactive action to minimise revenue loss.

**Metrics:**
- **Cancellation Rate** — percentage of transactions cancelled
- **Revenue at Risk** — total value lost from cancellations
- **Cancellation Frequency** — number of cancelled invoices
- **Customer Risk Level** — identification of high-cancellation customers

**Results:**
- A substantial portion of revenue is at risk from cancellations
- Losses are concentrated in specific products, not evenly distributed
- Certain customers exhibit systematic cancellation behaviour
- Strong seasonal spikes (e.g. December) indicate predictable, manageable risk periods

**Risk control strategies:**

| Strategy | Example | Expected Impact |
|----------|---------|-----------------|
| Product Risk Control | Flag Paper Craft, Little Birdie; Medium Ceramic Top Storage Jar | Return-related losses reduced 10–20% |
| Customer Risk Monitoring | Flag customers with >50% cancellation rate | Reduced fraudulent/excessive cancellation |
| Seasonal Risk Planning | Prepare for peak cancellation periods (December) | Reduced operational shock |

---

## Customer Journey — How Recommendations Are Delivered

| Touchpoint | What Happens |
|-----------|--------------|
| **Browsing** | MBA-powered "frequently bought together"; CF-powered similar products; bundle highlights; high-risk products excluded |
| **Checkout** | Real-time cross-sell suggestions; bundle completion prompts ("complete the set"); low-risk recommendations only |
| **Post-Purchase** | Personalised email recommendations; targeted promo campaigns; repeat purchase encouragement |

---

## Decision Support Dashboard

An interactive, ML-driven dashboard for stakeholders with four tabs:

| Tab | Description |
|-----|-------------|
| **Executive Summary** | High-level revenue trends, KPIs, top products and customers |
| **Customer-Based Recommendations** | CF-powered personalised recommendations by customer |
| **Product-Based Recommendations** | MBA-driven bundle and cross-sell opportunities |
| **Risk Control & Monitoring** | Cancellation patterns, high-risk products, customers, and time periods |

---

## Getting Started

### Prerequisites

```bash
pip install kagglehub pandas numpy matplotlib seaborn networkx scikit-learn
```

### Run the Notebook

1. Clone this repository
2. Open `Online_Retail_Analysis_Part_3.ipynb` in Jupyter or Kaggle
3. On first run, uncomment the install line in the Setup cell:
   ```python
   # !pip install kagglehub --quiet
   ```
4. Run all cells top to bottom — the dataset downloads automatically via `kagglehub`

---

## Output Charts

| File | Description |
|------|-------------|
| `eda_monthly_trends.png` | Monthly revenue, transaction volume, and active customers |
| `eda_top_products_countries.png` | Top 15 products and top 10 export markets by revenue |
| `eda_pareto.png` | Customer revenue concentration (Pareto curve) |
| `cf_heatmap.png` | Item–item cosine similarity heatmap |
| `cf_recs_profile.png` | Personalised recommendation profiles for top 3 customers |
| `canc_trends.png` | Monthly cancellation volume and revenue lost |
| `canc_patterns.png` | Cancellation patterns by day of week and hour of day |
| `canc_top_products.png` | Top 15 most-cancelled products by revenue impact |

---

## Limitations

- **UK-only scope for MBA** — international orders may exhibit different buying patterns; separate models recommended for top export markets
- **23% anonymous customers excluded from CF** — session-based or context-aware methods can address this gap
- **Transactional data only** — the dataset contains 8 variables; additional external data (demographics, product attributes) could enhance model depth
- **Popularity bias** — recommendations may skew towards bestsellers; a popularity-penalty re-ranking step is recommended
- **No real-time deployment or A/B testing** — experimental validation is required before production deployment
- **Temporal drift** — models trained on 2009–2011 data should be retrained on current data before use

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling and feature engineering |
| `matplotlib`, `seaborn` | Visualisation |
| `networkx` | Association rule graph (MBA) |
| `scikit-learn` | Cosine similarity (CF engine) |
| `kagglehub` | Automatic dataset download |

---
