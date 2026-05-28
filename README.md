# A/B Test — Average Bidding vs. Maximum Bidding

## Case Study

An e-commerce platform recently introduced a new type of bid called **Average Bidding** as an alternative to the existing **Maximum Bidding** mechanism. One of the platform's customers decided to test this new feature and requested an A/B test to understand whether average bidding returns better results than maximum bidding.

The dataset contains website activity data for this customer — including the number of advertisements users viewed and clicked on, the number of purchases made, and the revenue generated. The experiment ran for **40 days**, with the customer split into two groups:

- **Control Group** — continued using Maximum Bidding (existing system)
- **Test Group** — switched to Average Bidding (new system)

The central question: **does average bidding generate more revenue than maximum bidding, and is any observed difference statistically significant?**

---

## Result

> **Average Bidding generates +31.77% more revenue** — statistically significant (p ≈ 0, Cohen's d = 2.07), with a 95% confidence interval of [+476, +737] on the mean difference.

**Recommendation: Switch to Average Bidding.**

---

## What the Data Shows

| Variable | Change | Significant | Cohen's d | Verdict |
|----------|--------|-------------|-----------|---------|
| Impression | +18.48% | ✓ YES | 0.96 (large) | More reach |
| Click | −22.21% | ✓ YES | −0.99 (large) | Lower CTR |
| Purchase | +5.67% | ✗ NO | 0.21 (small) | Inconclusive |
| **Earning** | **+31.77%** | **✓ YES** | **2.07 (large)** | **Clear winner** |

### Derived Business Metrics

| Metric | Control (Max Bid) | Test (Avg Bid) | Change |
|--------|---------|------|--------|
| CTR (Click / Impression) | 5.36% | 3.42% | −36.2% |
| Conversion Rate (Purchase / Click) | 11.6% | 15.7% | **+35.3%** |
| Revenue per Click | 0.408 | 0.668 | **+63.7%** |
| Revenue per Impression | 0.0195 | 0.0214 | +9.7% |

**The mechanism:** Average bidding reaches a broader audience (more impressions) with a lower click-through rate, but the users who *do* click convert at a 35% higher rate and spend 64% more per click. The lower click count is not a failure — it reflects more selective, higher-intent audience targeting.

---

## Dataset

| Variable | Description |
|----------|-------------|
| Impression | Number of times the advertisement was displayed |
| Click | Number of clicks on the advertisement |
| Purchase | Number of completed purchases |
| Earning | Revenue generated |

- **Control Group:** Maximum Bidding — 40 daily observations
- **Test Group:** Average Bidding — 40 daily observations

---

## Methodology

**Three-step statistical testing pipeline** applied to each variable:

1. **Shapiro-Wilk** — normality assumption
2. **Levene** — variance homogeneity (only if both groups pass normality)
3. **Student t-test** (equal variance) · **Welch t-test** (unequal variance) · **Mann-Whitney U** (non-normal fallback)

**Beyond p-values:**
- **Cohen's d** — effect size to quantify practical significance
- **95% Confidence Intervals** on the mean difference
- **Derived metrics** — CTR, conversion rate, revenue per click, revenue per impression

Significance level: α = 0.05

---

## Files

```
├── ab_test_improved.ipynb   # Full analysis notebook
├── ab_testing.xlsx          # Dataset (Control Group + Test Group sheets)
└── README.md
```

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl

jupyter notebook ab_test_improved.ipynb
```

The notebook produces three output figures:
- `ab_distributions.png` — box plots + KDE distributions for all variables
- `ab_derived_metrics.png` — CTR, conversion rate, revenue efficiency comparison
- `ab_summary_dashboard.png` — % change with CI whiskers + Cohen's d chart


<img width="2385" height="594" alt="image" src="https://github.com/user-attachments/assets/702c6a25-ced5-4dd3-98d0-6f89d40986bc" />
<img width="2685" height="1182" alt="image" src="https://github.com/user-attachments/assets/670caa26-7f1d-432f-91ed-c4cbfb137cdc" />
<img width="2042" height="1442" alt="image" src="https://github.com/user-attachments/assets/edf36787-f111-4c93-8cf2-f83f17f86efd" />




