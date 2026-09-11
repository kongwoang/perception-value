# Status

Phase 0 complete. Verdict **GO**; see `docs/cvpr_phase0_findings.md`.

Nothing is running in the background.

## Runs behind the findings
| what | run |
|---|---|
| Jetson profile (TensorRT, idle board) | `results/raw/20260911_140226_profile` |
| CHEAP/FULL pair selection | `results/raw/20260911_140836_modesel` |
| Detection over 8008 frames x 4 modes | `results/raw/20260911_142608_detect` |
| Per-object mechanism analysis | `results/raw/20260911_143838_mechanism` |
| Main analysis | `results/raw/20260911_150044_analysis` |
| Sensitivity, 25 configurations | `results/raw/20260911_155416_sensitivity` |

## Not done (deliberately)
- Robustness pairs cheap_384/full_640 and cheap_320/full_960. The detection cache
  already exists for both, so this is `03_build_tables.py` + `04_analysis.py` only.
- Second dataset and second detector family — the two extensions most likely to
  strengthen a submission, per the findings doc.
