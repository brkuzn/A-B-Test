# A/B Test — Average Bidding vs. Maximum Bidding

A complete A/B test analysis evaluating whether a new ad bidding mechanism (**Average Bidding**) outperforms the existing one (**Maximum Bidding**) across four business metrics: impressions, clicks, purchases, and revenue.

---

## Result

> **Average Bidding generates +31.77% more revenue** — statistically significant (p ≈ 0, Cohen's d = 2.07), with a 95% confidence interval of [+476, +737] on the mean difference.

**Switch to Average Bidding.**

---

## What the Data Shows

| Variable | Change | Significant | Cohen's d | Verdict |
|----------|--------|-------------|-----------|---------|
| Impression | +18.48% | ✓ YES | 0.96 (large) | More reach |
| Click | −22.21% | ✓ YES | −0.99 (large) | Lower CTR |
| Purchase | +5.67% | ✗ NO | 0.21 (small) | Inconclusive |
| **Earning** | **+31.77%** | **✓ YES** | **2.07 (large)** | **Clear winner** |

### Derived Business Metrics

| Metric | Control | Test | Change |
|--------|---------|------|--------|
| CTR (Click / Impression) | 5.36% | 3.42% | −36.2% |
| Conversion Rate (Purchase / Click) | 11.6% | 15.7% | **+35.3%** |
| Revenue per Click | 0.408 | 0.668 | **+63.7%** |
| Revenue per Impression | 0.0195 | 0.0214 | +9.7% |

**The mechanism:** Average bidding reaches a broader audience (more impressions) with a lower click-through rate, but the users who *do* click convert at a 35% higher rate and spend 64% more per click. It is not failing to attract clicks — it is targeting higher-intent users.

---

## Methodology

**Three-step statistical testing pipeline** applied to each variable:

1. **Shapiro-Wilk** — normality assumption  
2. **Levene** — variance homogeneity (if both groups are normal)  
3. **Student t-test** (equal variance) · **Welch t-test** (unequal variance) · **Mann-Whitney U** (non-normal fallback)

**Beyond p-values:**
- **Cohen's d** — effect size for practical significance
- **95% Confidence Intervals** on the mean difference
- **Derived metrics** — CTR, conversion rate, revenue per click, revenue per impression

Significance level: α = 0.05

---

## Dataset

- **Control Group:** Maximum Bidding — 40 daily observations  
- **Test Group:** Average Bidding — 40 daily observations  
- **Variables:** Impression, Click, Purchase, Earning

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
- `ab_distributions.png` — box plots + KDE for all variables
- `ab_derived_metrics.png` — CTR, conversion rate, revenue efficiency
- `ab_summary_dashboard.png` — % change with CI whiskers + Cohen's d chart
