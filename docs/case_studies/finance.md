# Case Study III — Financial AI

## Crystallizing Confidence in High-Stakes Decisions

A quantitative trading firm uses an LLM to summarize SEC filings and flag risk factors. Accuracy: 88% on common patterns. Problem: financial disasters hide in long-tail events — unusual accounting, footnote disclosures, novel structured products. The model's raw confidence on long-tail events: 0.84 — indistinguishable from its confidence on common patterns.

## The PHANTASM intervention

UC is integrated into the filing analysis pipeline. Every output is crystallized into a reliability tier.

## Tier distribution across 10,000 filings

```
◆ crystal : 72%  — Standard disclosures. Automated action.
◇ solid   : 18%  — Common risk factors. Batch analyst review.
≈ fluid   :  7%  — Unusual footnotes. Mandatory individual review.
~ vapor   :  3%  — Novel instruments. Blocked from automation.
```

The 3% vapor filings — 300 documents — included 4 that preceded significant market events. Under the old system, all 300 looked like confident outputs. Under PHANTASM UC, those 4 were flagged before any position was taken.

## The key reversal

The model's miscalibration was not a flaw to suppress. It was a precision sensor for tail risk. The reliability tier is now a first-class trading signal.
