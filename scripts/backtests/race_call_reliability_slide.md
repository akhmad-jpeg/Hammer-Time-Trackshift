# RELIABILITY: WHAT OUR CONFIDENCE IS WORTH

**We rank attack windows. We don't certify probabilities.**

Every stored race was replayed through the *same* forward projection the Race Call runs (1717 checkpoints, 527 battle segments, 84 race weekends, 2020–2026).
When the model's cumulative confidence reached each level, we scored whether a real on-track pass followed before the next stop:

| Model confidence | Checkpoints | Real passes | Real pass rate | Lift vs base |
|---|---|---|---|---|
| P ≥ 0.30 | 526 | 95 | **18%** | 1.87× |
| P ≥ 0.40 | 512 | 92 | **18%** | 1.86× |
| P ≥ 0.50 | 503 | 90 | **18%** | 1.85× |
| P ≥ 0.60 | 491 | 88 | **18%** | 1.85× |
| P ≥ 0.70 | 484 | 87 | **18%** | 1.86× |
| P ≥ 0.80 | 472 | 86 | **18%** | 1.88× |
| P ≥ 0.90 | 287 | 54 | **19%** | 1.95× |
| P ≥ 0.95 | 161 | 33 | **20%** | 2.12× |

Base rate — an arbitrary in-battle checkpoint — is **10%**.  Confidence at the default call threshold (P ≥ 0.80) is worth **18%** — the operating evidence behind the product's threshold, published rather than asserted.

Read it honestly: the rate is flat through the mid-range (18% at P≥0.50 vs 18% at P≥0.80) and rises only at the very top (20% at P≥0.95) — confidence acts as a **coarse filter**, not a fine-grained dial (2.12× the base rate).  That is what our data supports today.

**Attack-window detection:** precision 38% · recall 65% (TP 240 / FP 394 / FN 132).

Honest by construction:
- windows end at the next pit stop (pit-strategy passes out of scope);
- small-label reality stated up front — the table IS the calibration;
- deterministic: same races → same table (regenerate any time).

\* fewer than 15 scored checkpoints — shown, not hidden.