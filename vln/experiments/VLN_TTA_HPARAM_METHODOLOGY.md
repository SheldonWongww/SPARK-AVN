# VLN TTA hyperparameter methodology

Status: **superseded**.

The former dual-split consistency proposal in this file used canonical order
seed 0, selected with a cross-split no-regression gate, and screened IDEA with
`O=10`. That design is not the active NavTTA protocol and must not be used for
new runs or claims.

The active methodology is
[`VLN_TTA_CONSISTENCY_SEARCH_PROTOCOL.md`](VLN_TTA_CONSISTENCY_SEARCH_PROTOCOL.md)
and the three `*_consistency_search_v2.json` specifications. In summary:

- `val_unseen` is disclosed development/selection data and uses complete
  global-shuffle order seeds 1/2/3;
- the frozen winner alone is evaluated on `val_seen` seeds 1/2/3 for a
  retention report;
- selection is Source-relative mean-minus-sample-standard-deviation, with SPL
  primary for R2R/R2R-CE and RGSPL primary for REVERIE, and has no mandatory
  positive-result gate;
- FSTTA uses the paper band `.95/.7/.9/1.1` and Eq. 6 test-stream variance
  history; FeedTTA uses `sgr_mode=paper_main` with the task-adapted
  target-native argmax protocol (reported as `FeedTTA-argmax`, against ordinary
  argmax Source; sampled Source is needed only for the sampling ablation);
  ATENA reports only its replay-reachable
  high-level policy scope; IDEA always uses `O=50` and remains blocked until
  precomputed Source statistics are SHA256-pinned;
- the v1 JSON files remain unchanged solely for historical result replay.
