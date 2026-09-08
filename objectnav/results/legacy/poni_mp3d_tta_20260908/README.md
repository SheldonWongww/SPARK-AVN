# Source and five retained PONI MP3D TTA runs

This local legacy archive intentionally contains the complete Source baseline
record, the three completed 2195-episode TTA runs whose primary Success, SPL,
and SoftSPL metrics exceeded it, and the best completed global-continual Tent
and FSTTA runs.

| Method | Run | Success | SPL | SoftSPL | Distance |
| --- | --- | ---: | ---: | ---: | ---: |
| Source | `mp3d_poni_seed_123` | 30.8428% | 11.7916% | 17.5601% | 5.2232 |
| FeedTTA | `20260828_172439_feedtta` | 33.1663% | 13.0658% | 19.4569% | 5.2222 |
| ATENA | `selected_full_20260905_final_atena_lr3e7_wd0` | 31.8907% | 12.2592% | 17.9773% | 5.3041 |
| EAM | `selected_full_20260905_final_eam_always_all_lr1e8` | 31.1162% | 11.9608% | 17.6440% | 5.2850 |
| FSTTA | `selected_full_20260905_final_fstta_slow1e7_n128_fg` | 30.7517% | 11.7797% | 17.3484% | 5.3177 |
| Tent | `selected_full_20260905_final_tent_valid_lr3e8` | 29.6583% | 11.8395% | 17.3058% | 5.2837 |

Each TTA directory under `runs/` contains its complete `stats.json`, run and
console logs, diagnostics, TensorBoard events, run manifest, merged result, and
launcher log. Git tracks the compact `stats.json`, diagnostics, run manifests,
and merged results; raw logs and TensorBoard events remain local.
`runs/Source/mp3d_poni_seed_123/` preserves the original eleven scene-part
result trees locally and tracks their compact statistics and merged result.

No older TTA full run, smoke run, part run, interrupted attempt, or
hyperparameter-search result is retained here. Those originals remain in the
PONI workspace and can be imported later if explicitly requested.

FeedTTA and ATENA consume binary episode feedback. ATENA queried the real
Habitat success value for all 2195 episodes (`query_rate=1.0`). EAM, FSTTA,
and Tent are unsupervised; keep these supervision classes separate in formal
comparisons.

This remains a `legacy` archive because the original run manifests were not
tied to the NavTTA top-level Git commit. Raw logs and TensorBoard contents are
ignored by Git; the file-level SHA256 list is
`objectnav/manifests/poni_tta_archive_20260908.sha256`.
