# VLN experiment specifications

`tta_hparam_search_v1.json` is the immutable design input for the five-model,
five-method `val_seen` hyperparameter search.  Search jobs use canonical order
seed 0.  After each setting freezes one configuration, that configuration is
run on exactly order seeds 0, 1, and 2.  The canonical seed-0 result is the
future main-table value; the three-order mean is a separate robustness result.

Raw consoles, job directories, checkpoints, and scheduler state belong below
`vln/results/logs/` and are ignored by Git.  Only compact summaries with full
run-manifest provenance may be added to tracked result files.

The scheduler's default complete workflow ends after the five full canonical
`val_seen` finalists per setting and freezes the best configuration.  This
keeps the current hyperparameter search independent of the later robustness
section.  Add `--with-orders` to run the frozen configuration under exactly
order seeds 0, 1, and 2; that option requires matching `--order-seed` support
in the baseline runner and never silently substitutes three ordinary RNG
seeds for three episode permutations.

Typical invocations are:

```text
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --resume
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --status
python3 vln/scripts/run_tta_hparam_search.py all --batch-id SEARCH_ID --watch
```

Each method is completed before the next method starts, while settings are
round-robin scheduled within a stage.  The default limits permit at most three
jobs overall, one job for any one model, and one continuous VLN-CE job.  These
caps, live GPU/RAM launch guards, and the launch stagger can be overridden only
after smoke measurements justify doing so.  A failed attempt is never erased:
`--resume --retry-failed` archives its console, result root, and any formal run
manifest before assigning a new run tag.
