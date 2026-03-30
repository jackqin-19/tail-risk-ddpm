# Midterm Notes

## Status Summary

| Milestone | Status |
|-----------|--------|
| Data download & preprocessing | Complete |
| Dataset / DataLoader | Complete |
| Model skeleton (MLP backbone) | Complete |
| Diffusion forward/reverse process | Complete |
| Training loop | Complete |
| DDPM sampler | Complete |
| Evaluation (VaR, ES, KS, Spearman) | Pending |
| Attribution (marginal + Shapley) | Pending |
| Exploratory notebooks | Complete |
| End-to-end run on real data | Complete (A->B->C) |
| Hyperparameter tuning | Pending |
| Report write-up | Pending |

---

## Open Questions

1. **Classifier-free guidance**: Should we implement CFG to strengthen conditioning on the tail label?
2. **Multi-step sequence generation**: Current evaluation focuses on the last day in each window; should we extend to multi-day risk metrics?
3. **Exogenous volatility signal**: Should volatility regime be provided as an explicit conditioning scalar?
4. **Calibration checks**: Should we add rank-histogram / PIT coverage tests?

---

## Key Findings So Far

- A module produces `prices.parquet` with full-history coverage from 2015 onward.
- B module produces balanced-panel datasets (`dataset_train/valid/test.npz`) from common dates.
- With current asset pool, common-date range starts at 2020-11-16 due to asset listing dates.
- C module now trains and samples successfully on current dataset with saved checkpoints and sample artifacts.
- Evaluation findings are not finalized because D module is still pending.

---

## Next Steps

- [x] Run `download_data.py` -> `preprocess.py` end-to-end.
- [x] Run `make_dataset.py` and generate `dataset_*.npz`.
- [x] Implement `model.py`, `diffusion.py`, `train.py`, and `sample.py`.
- [ ] Implement `evaluate.py` and `attribution.py`.
- [ ] Run end-to-end training, sampling, and evaluation.
- [ ] Finalize report section with quantitative tables and figures.
