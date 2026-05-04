# Phase 8 — Curated Example Gallery (Test Set)

Five films selected per the locked rules in ``docs/proposals/phase8_preregistration.md`` Section 4.7. All films come from the held-out 257-film test set.

## High-confidence Greenlight

**Film:** Something Wicked This Way Comes_1983  
**IMDb ID:** tt0086336  
**Genre:** Fantasy  
**Release year:** 1983  
**True ROI > 2x:** False  

**Note:** Highest-probability Greenlight (P=1.000).

### Layered triage output

| Layer | Output |
|---|---|
| 1 — log_roi point prediction | 1.997 |
| 1 — uncalibrated P(ROI > 2x) | 0.904 |
| 2 — **calibrated P(ROI > 2x)** | **1.000** |
| 2 — log_roi 90% interval | [0.00, 3.99] |
| 2 — conformal set @ 0.90 (size) | 1 |
| 3 — recommended action | **Greenlight** |
| 3 — expected cost (Greenlight / Pass / Refer) | $0.00M / $100.00M / $5.0K |

**Decision rationale:**

> Recommended Greenlight: model probability 1.000 of >=2x ROI yields expected loss $0 from Greenlight vs $100.0M from Pass and $5.0K from Refer.

**SHAP rationale (Layer 4):**

> Top features pushing probability up: release_year_parsed (+0.802 log-odds), Genre=Horror (+0.258 log-odds), Embedding PC 12 (+0.187 log-odds). Top features pulling probability down: Network: lead_role_count (-0.069 log-odds), Genre=Action (-0.050 log-odds), Topic 01 proportion (-0.043 log-odds).

---

## High-confidence Pass (substituted)

**Film:** The Last Samurai_2003  
**IMDb ID:** tt0325710  
**Genre:** Drama  
**Release year:** 2003  
**True ROI > 2x:** True  

**Note:** No Pass recommendations on test set (Phase 6 trigger #1 fired in Phase 6 cal); substituting the lowest-probability Refer (P=0.375).

### Layered triage output

| Layer | Output |
|---|---|
| 1 — log_roi point prediction | 0.745 |
| 1 — uncalibrated P(ROI > 2x) | 0.287 |
| 2 — **calibrated P(ROI > 2x)** | **0.375** |
| 2 — log_roi 90% interval | [-1.25, 2.74] |
| 2 — conformal set @ 0.90 (size) | 2 |
| 3 — recommended action | **Refer** |
| 3 — expected cost (Greenlight / Pass / Refer) | $31.25M / $37.50M / $5.0K |

**Decision rationale:**

> Recommended Refer to human reader: at probability 0.375, expected losses from Greenlight ($31.2M) and Pass ($37.5M) both exceed the human-reader cost ($5.0K). Manual review is preferred.

**SHAP rationale (Layer 4):**

> Top features pushing probability up: Genre=Action (+0.193 log-odds), Genre=Thriller (+0.091 log-odds), Embedding PC 04 (+0.088 log-odds). Top features pulling probability down: Genre=Romance (-0.241 log-odds), Genre=Horror (-0.196 log-odds), Network: top1_dialogue_share (-0.135 log-odds).

---

## High-uncertainty Refer near 0.50

**Film:** Pineapple Express_2008  
**IMDb ID:** tt0910936  
**Genre:** Action  
**Release year:** 2008  
**True ROI > 2x:** True  

**Note:** Closest-to-0.5 Refer (P=0.500).

### Layered triage output

| Layer | Output |
|---|---|
| 1 — log_roi point prediction | 0.546 |
| 1 — uncalibrated P(ROI > 2x) | 0.313 |
| 2 — **calibrated P(ROI > 2x)** | **0.500** |
| 2 — log_roi 90% interval | [-1.45, 2.54] |
| 2 — conformal set @ 0.90 (size) | 2 |
| 3 — recommended action | **Refer** |
| 3 — expected cost (Greenlight / Pass / Refer) | $25.00M / $50.00M / $5.0K |

**Decision rationale:**

> Recommended Refer to human reader: at probability 0.500, expected losses from Greenlight ($25.0M) and Pass ($50.0M) both exceed the human-reader cost ($5.0K). Manual review is preferred.

**SHAP rationale (Layer 4):**

> Top features pushing probability up: Embedding PC 12 (+0.157 log-odds), Topic 11 proportion (+0.091 log-odds), Network: lead_role_count (+0.059 log-odds). Top features pulling probability down: release_year_parsed (-0.144 log-odds), Genre=Horror (-0.133 log-odds), Genre=Action (-0.127 log-odds).

---

## Genre-tractable true positive

**Film:** Alien³_1992  
**IMDb ID:** tt0103644  
**Genre:** Science Fiction  
**Release year:** 1992  
**True ROI > 2x:** True  

**Note:** Science Fiction hit correctly recognized (P=0.712; action=Refer).

### Layered triage output

| Layer | Output |
|---|---|
| 1 — log_roi point prediction | 0.990 |
| 1 — uncalibrated P(ROI > 2x) | 0.715 |
| 2 — **calibrated P(ROI > 2x)** | **0.712** |
| 2 — log_roi 90% interval | [-1.01, 2.99] |
| 2 — conformal set @ 0.90 (size) | 1 |
| 3 — recommended action | **Refer** |
| 3 — expected cost (Greenlight / Pass / Refer) | $14.38M / $71.25M / $5.0K |

**Decision rationale:**

> Recommended Refer to human reader: at probability 0.712, expected losses from Greenlight ($14.4M) and Pass ($71.2M) both exceed the human-reader cost ($5.0K). Manual review is preferred.

**SHAP rationale (Layer 4):**

> Top features pushing probability up: Genre=Horror (+0.384 log-odds), Network: lead_role_count (+0.190 log-odds), Topic 01 proportion (+0.164 log-odds). Top features pulling probability down: Genre=Action (-0.093 log-odds), Network: n_significant_characters (-0.086 log-odds), release_year_parsed (-0.053 log-odds).

---

## Genre-intractable defer

**Film:** Swingers_1996  
**IMDb ID:** tt0117802  
**Genre:** Comedy  
**Release year:** 1996  
**True ROI > 2x:** True  

**Note:** Comedy film correctly deferred (P=0.560; near-0.5 uncertainty).

### Layered triage output

| Layer | Output |
|---|---|
| 1 — log_roi point prediction | 0.886 |
| 1 — uncalibrated P(ROI > 2x) | 0.543 |
| 2 — **calibrated P(ROI > 2x)** | **0.560** |
| 2 — log_roi 90% interval | [-1.11, 2.88] |
| 2 — conformal set @ 0.90 (size) | 2 |
| 3 — recommended action | **Refer** |
| 3 — expected cost (Greenlight / Pass / Refer) | $22.00M / $56.00M / $5.0K |

**Decision rationale:**

> Recommended Refer to human reader: at probability 0.560, expected losses from Greenlight ($22.0M) and Pass ($56.0M) both exceed the human-reader cost ($5.0K). Manual review is preferred.

**SHAP rationale (Layer 4):**

> Top features pushing probability up: Network: max_betweenness_centrality (+0.154 log-odds), Embedding PC 05 (+0.140 log-odds), Embedding PC 03 (+0.082 log-odds). Top features pulling probability down: release_year_parsed (-0.120 log-odds), Genre=Action (-0.085 log-odds), Embedding PC 11 (-0.041 log-odds).

---
