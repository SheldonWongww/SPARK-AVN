# R2R final benchmark results

These results use order seed 0. Hyperparameters were selected on `val_seen`; `val_unseen` evaluates the frozen configuration once and was not used for reselection.

FeedTTA and ATENA consume binary episode-success feedback. Tent, FSTTA, and EAM are unsupervised TTA.

## Main comparison

Values and deltas are `SR / SPL` in percentage points.

| Model | Method | Supervision | val_seen | val_unseen | val_unseen delta vs Source |
|---|---|---|---:|---:|---:|
| DUET | source | source_no_adaptation | 78.84 / 72.88 | 71.52 / 60.41 | +0.00 / +0.00 |
| DUET | tent | unsupervised_tta | 78.94 / 73.08 | 69.48 / 58.34 | -2.04 / -2.07 |
| DUET | fstta | unsupervised_tta | 79.14 / 73.10 | 70.03 / 58.75 | -1.49 / -1.66 |
| DUET | eam | unsupervised_tta | 80.31 / 74.75 | 68.54 / 59.19 | -2.98 / -1.22 |
| DUET | feedtta | binary_episode_feedback_tta | 79.33 / 73.88 | 72.07 / 60.54 | +0.55 / +0.13 |
| DUET | atena | binary_episode_feedback_tta | 80.41 / 76.30 | 70.50 / 62.32 | -1.02 / +1.91 |
| HAMT | source | source_no_adaptation | 75.61 / 72.18 | 66.24 / 61.51 | +0.00 / +0.00 |
| HAMT | tent | unsupervised_tta | 76.30 / 72.88 | 66.20 / 61.45 | -0.04 / -0.06 |
| HAMT | fstta | unsupervised_tta | 76.40 / 72.97 | 66.11 / 61.27 | -0.13 / -0.24 |
| HAMT | eam | unsupervised_tta | 76.69 / 73.40 | 65.99 / 61.04 | -0.25 / -0.47 |
| HAMT | feedtta | binary_episode_feedback_tta | 76.00 / 72.66 | 66.45 / 61.92 | +0.21 / +0.41 |
| HAMT | atena | binary_episode_feedback_tta | 77.47 / 74.16 | 63.98 / 58.76 | -2.26 / -2.75 |
| GOAT | source | source_no_adaptation | 84.82 / 80.05 | 78.12 / 67.58 | +0.00 / +0.00 |
| GOAT | tent | unsupervised_tta | 84.92 / 80.32 | 78.29 / 68.98 | +0.17 / +1.40 |
| GOAT | fstta | unsupervised_tta | 84.92 / 80.21 | 78.03 / 67.68 | -0.09 / +0.10 |
| GOAT | eam | unsupervised_tta | 85.31 / 80.55 | 77.86 / 67.92 | -0.26 / +0.34 |
| GOAT | feedtta | binary_episode_feedback_tta | 84.92 / 80.11 | 78.03 / 67.52 | -0.09 / -0.06 |
| GOAT | atena | binary_episode_feedback_tta | 84.92 / 80.24 | 78.54 / 68.37 | +0.42 / +0.79 |

## Frozen parameters

Source uses the standard deterministic argmax protocol. TTA parameter objects below are copied verbatim from the `val_seen` registry.

| Model | Method | Parameters |
|---|---|---|
| DUET | source | `{"action_seed":0,"action_selection":"argmax"}` |
| DUET | tent | `{"episodic":false,"last_k_ln":9,"lr":1e-05,"max_grad_norm":0.0,"norm_scope":"last_k_ln","optimizer":"AdamW","update_interval":1,"weight_decay":0.0}` |
| DUET | fstta | `{"a":0.9,"b":1.1,"beta1":0.9,"beta2":0.99,"eigen_eps":1e-06,"episodic":false,"fast_grad_mode":"concordant","last_k_ln":4,"lr_fast":0.0018,"lr_slow":0.0003,"m":8,"max_grad_norm":0.0,"n":4,"norm_scope":"last_k_ln","optimizer":"AdamW","q":0.1,"reset_optimizer_each_episode":true,"reset_slow_optimizer_each_window":false,"rho":0.95,"slow_optimizer":"AdamW","tau":0.7,"use_fast_lr_scaler":true,"use_slow":true,"weight_decay":0.0}` |
| DUET | eam | `{"batch_size":8,"confidence_scale":0.5,"episodic":false,"lr":1e-05,"max_grad_norm":0.0,"memory_size":64,"optimizer":"Adam","update_interval":8,"weight_decay":0.0}` |
| DUET | feedtta | `{"action_selection":"argmax","alpha":0.1,"episodic":false,"gamma":0.8,"lr":5e-06,"max_grad_norm":0.0,"normalize_gradient":false,"optimizer":"Adam","optimizer_eps":1e-05,"p":0.05,"scope_profile":"last_crossmodal","sgr_seed":0,"weight_decay":0.0}` |
| DUET | atena | `{"action_selection":"argmax","episodic":false,"lr_query":1.6e-06,"lr_self":2e-07,"max_grad_norm":0.0,"mix_lambda":0.5,"optimizer":"AdamW","query_threshold":0.0,"self_loss_weight":0.1,"weight_decay":0.01}` |
| HAMT | source | `{"action_seed":0,"action_selection":"argmax"}` |
| HAMT | tent | `{"episodic":false,"lr":3e-06,"max_grad_norm":0.0,"norm_scope":"ln","optimizer":"AdamW","update_interval":1,"weight_decay":0.0}` |
| HAMT | fstta | `{"a":0.9,"b":1.1,"beta1":0.9,"beta2":0.99,"eigen_eps":1e-06,"episodic":false,"fast_grad_mode":"concordant","last_k_ln":4,"lr_fast":0.0008,"lr_slow":0.0002,"m":1,"max_grad_norm":0.0,"n":16,"norm_scope":"last_k_ln","optimizer":"AdamW","q":0.1,"reset_optimizer_each_episode":true,"reset_slow_optimizer_each_window":false,"rho":0.95,"slow_optimizer":"AdamW","tau":0.7,"use_fast_lr_scaler":true,"use_slow":true,"weight_decay":0.0}` |
| HAMT | eam | `{"batch_size":8,"confidence_scale":0.6,"episodic":false,"lr":1e-06,"max_grad_norm":0.0,"memory_size":32,"optimizer":"Adam","update_interval":8,"weight_decay":0.0}` |
| HAMT | feedtta | `{"action_selection":"argmax","alpha":0.1,"episodic":false,"gamma":0.9,"lr":2e-06,"max_grad_norm":0.0,"normalize_gradient":false,"optimizer":"Adam","optimizer_eps":1e-05,"p":0.05,"scope_profile":"last_crossmodal","sgr_seed":0,"weight_decay":0.0}` |
| HAMT | atena | `{"action_selection":"argmax","episodic":false,"lr_query":3.2e-06,"lr_self":4e-07,"max_grad_norm":0.0,"mix_lambda":0.25,"optimizer":"AdamW","query_threshold":0.1,"self_loss_weight":0.1,"weight_decay":0.01}` |
| GOAT | source | `{"action_seed":0,"action_selection":"argmax"}` |
| GOAT | tent | `{"episodic":false,"lr":1e-05,"max_grad_norm":0.0,"norm_scope":"ln","optimizer":"AdamW","update_interval":1,"weight_decay":0.0}` |
| GOAT | fstta | `{"a":0.9,"b":1.1,"beta1":0.9,"beta2":0.99,"eigen_eps":1e-06,"episodic":false,"fast_grad_mode":"concordant","last_k_ln":4,"lr_fast":0.0018,"lr_slow":0.001,"m":3,"max_grad_norm":0.0,"n":8,"norm_scope":"last_k_ln","optimizer":"AdamW","q":0.1,"reset_optimizer_each_episode":true,"reset_slow_optimizer_each_window":false,"rho":0.95,"slow_optimizer":"AdamW","tau":0.7,"use_fast_lr_scaler":true,"use_slow":true,"weight_decay":0.0}` |
| GOAT | eam | `{"batch_size":8,"confidence_scale":0.3,"episodic":false,"lr":3e-06,"max_grad_norm":0.0,"memory_size":32,"optimizer":"Adam","update_interval":4,"weight_decay":0.0}` |
| GOAT | feedtta | `{"action_selection":"argmax","alpha":-0.1,"episodic":false,"gamma":0.8,"lr":1e-06,"max_grad_norm":0.0,"normalize_gradient":false,"optimizer":"Adam","optimizer_eps":1e-05,"p":0.1,"scope_profile":"action_head","sgr_seed":0,"weight_decay":0.0}` |
| GOAT | atena | `{"action_selection":"argmax","episodic":false,"lr_query":3.75e-07,"lr_self":3.515625e-08,"max_grad_norm":0.0,"mix_lambda":0.0,"optimizer":"AdamW","query_threshold":0.15,"self_loss_weight":0.1,"weight_decay":0.01}` |

## Provenance

The machine-readable companion file contains the complete input digests, dataset/order/checkpoint identities, and immutable formal-run identities. The table below identifies every formal run.

| Model | Method | Split | Run tag | Git commit | Formal manifest SHA256 |
|---|---|---|---|---|---|
| DUET | source | val_seen | `vln-r2r-modelwise-cartesian-v2-seed0-source-cartesian-0000-duet-r2r-3865cf90bc` | `a258ac5b1d604cbdaecd64fe0ac184241e8384c1` | `e7adfa7bcab1c9b0c19b4f37b08c9c38de75cb247716c629504941612a87d9bc` |
| DUET | source | val_unseen | `grouped-source-20260810T080743Z` | `0c6e38bb3cb1babbd575c09c97f9439571a43230` | `7014badabce28cfe11d38bdac35202706bfb7e5bdf4d91deab8e1f0b35da632f` |
| DUET | tent | val_seen | `vln-r2r-targeted-gap-expand-v2-seed0-refine-00-tent-0012-duet-r2r-df01c3be0b` | `ef4f3e37fca3b3057cb2a2c365352b70633bd22b` | `af4d7032eeec5e9051ecc6ffc09402370f2d70b217b2b4e6c20044d4c56bd609` |
| DUET | tent | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-00-duet-00-tent-c9af29e0fb` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `aa777fd538d5386f5a03a61abac3e1daddf84a03fb24614def64d36546c32cfd` |
| DUET | fstta | val_seen | `vln-r2r-fstta-feedtta-postfix-v1-seed0-refine-00-fstta-0010-duet-r2r-0d9c5cb9c3` | `b2e37dd387fd8df4543dcfcbac7c79592f88d739` | `1cc85244bb6e69b870813c24114cba8c53d655ef537e9b73534583257e7605e5` |
| DUET | fstta | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-00-duet-01-fstta-b73f4bfa5e` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `a1018397e2a70975982d3e4919018edd5d8e6246a7daf427d66989903736164b` |
| DUET | eam | val_seen | `vln-r2r-eam-atena-formal-confirm-v1-seed0-refine-00-eam-0000-duet-r2r-57ab8b8bf5` | `ed61007ba0cd76a39f9a9dcbbd5f70dfcfffef36` | `61d7725c9fb6db68c7912b19226909573ac0835a3b3ce20614009b9d0ee5399f` |
| DUET | eam | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-00-duet-02-eam-9a323c5931` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `dabdf960135b881d00dfb2075ce6418f107452d3889d48437c8ad93d0b40131c` |
| DUET | feedtta | val_seen | `vln-r2r-fstta-feedtta-postfix-v1-seed0-refine-01-feedtta-0004-duet-r2r-effec71c87` | `b2e37dd387fd8df4543dcfcbac7c79592f88d739` | `e9ab74ff173f04edaae26a21b38b9123b58ada37241b248ffbb53da246ea43b3` |
| DUET | feedtta | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-00-duet-03-feedtta-16b79b24b1` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `efb639ee95ce257a92754ddfe4cb83bd1ac7731f62704a56fabf4c839f7f612a` |
| DUET | atena | val_seen | `vln-r2r-eam-atena-formal-confirm-v1-seed0-refine-01-atena-0000-duet-r2r-8d3372dc24` | `ed61007ba0cd76a39f9a9dcbbd5f70dfcfffef36` | `4a3399d6a1a252d5a0a3418e186ca05e106799ac3cf7e2dcfc146dc1f053ec09` |
| DUET | atena | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-00-duet-04-atena-02afc15d26` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `0c37d8f7a4362ed79903fbf8f400c55aa6e21a7ef862c1987bb32159f67c461d` |
| HAMT | source | val_seen | `vln-r2r-modelwise-cartesian-v2-seed0-source-cartesian-0000-hamt-r2r-3865cf90bc` | `a258ac5b1d604cbdaecd64fe0ac184241e8384c1` | `6c30dfc92530ba96d4bef2f854fc494ec84967eec89393b3ca790b24408c159c` |
| HAMT | source | val_unseen | `hamt-r2r-e2e-source-20260810T152328Z` | `f6175d64487db902ca86a4ab455e48ffc11bafe2` | `89b08dd6df22a9a5aaa4434783af6781d5c4d09bd8dfbe571dd2ced2cf78d3c3` |
| HAMT | tent | val_seen | `vln-r2r-low-lr-refine-prod-nosource-20260820-refine-04-tent-0007-hamt-r2r-84fd45bb28` | `a7fa28efc34970527461f03f60b959d1bff0be10` | `daab8d26155cf0fc6316f2492b5bfbc9032283287ba55d36a10110510941a8ae` |
| HAMT | tent | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-01-hamt-00-tent-316ff74aa8` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `3e0e0a9309086d5d6f845303d9c3e659e4d0572cde715e01d7b8cb2e12901f41` |
| HAMT | fstta | val_seen | `vln-r2r-fstta-feedtta-postfix-v1-seed0-refine-02-fstta-0006-hamt-r2r-65408b9d22` | `b2e37dd387fd8df4543dcfcbac7c79592f88d739` | `44f4fcaf0f02b07b88ff96695aa8168219a8a277d0eeeda54376f8a37f47122a` |
| HAMT | fstta | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-01-hamt-01-fstta-4998f37857` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `ff76c603296378fe8cf0a5166a7bdc92c416d500cc941689f2523a91a6ce0aa4` |
| HAMT | eam | val_seen | `vln-r2r-eam-atena-formal-confirm-v1-seed0-refine-02-eam-0000-hamt-r2r-c88e328c7a` | `ed61007ba0cd76a39f9a9dcbbd5f70dfcfffef36` | `bd750d1e33a3fb11de0da1fd03aa07c118f773e222f051dc2200d6c794ed390a` |
| HAMT | eam | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-01-hamt-02-eam-6c6781aeca` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `34525075dd3bbdc725dbf04f2a8ea8c95df503ab6c045488a85f387818f2574b` |
| HAMT | feedtta | val_seen | `vln-r2r-fstta-feedtta-postfix-v1-seed0-refine-03-feedtta-0001-hamt-r2r-4f02ae8102` | `b2e37dd387fd8df4543dcfcbac7c79592f88d739` | `78dba723a2a71778c7bef78e19f810777d1baf37b0b51ec759c48d79b976ecd4` |
| HAMT | feedtta | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-01-hamt-03-feedtta-9f225afdaf` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `b68c10e780f3b4a61438871760e240f57c3951f59d67fc1424e3a0dcdca4c981` |
| HAMT | atena | val_seen | `vln-r2r-eam-atena-formal-confirm-v1-seed0-refine-03-atena-0000-hamt-r2r-8dfe151e97` | `ed61007ba0cd76a39f9a9dcbbd5f70dfcfffef36` | `d501f3b5831071591ebf7c8f981f66899f9cc367215b4a6b18d948ac95abce33` |
| HAMT | atena | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-01-hamt-04-atena-441dd582be` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `66b145aaa2a23727f84424db734b2a6209c806c8c215ff7c9c78e804812bb9e6` |
| GOAT | source | val_seen | `vln-r2r-modelwise-cartesian-v2-seed0-source-cartesian-0000-goat-r2r-3865cf90bc` | `a258ac5b1d604cbdaecd64fe0ac184241e8384c1` | `6dcb148b179ea45bbef99ccc34ab0109be035b78dfe5e4fa5e48329eaed65b96` |
| GOAT | source | val_unseen | `grouped-source-20260810T080743Z` | `0c6e38bb3cb1babbd575c09c97f9439571a43230` | `06206c9dee71d94b92aaaf0c7092e49869173bfc455d8c928387ed96718c4bcd` |
| GOAT | tent | val_seen | `vln-r2r-low-lr-refine-prod-nosource-20260820-refine-08-tent-0008-goat-r2r-a2b3725323` | `a7fa28efc34970527461f03f60b959d1bff0be10` | `f2fdafba2492e446b1605c0fb07e5192274675124a9dcb820f861642c6ea5e4e` |
| GOAT | tent | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-02-goat-00-tent-eb3052bd14` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `cea2123ea714d27f0f21a32b802a6937955a06258f1896404288900b7cdc2583` |
| GOAT | fstta | val_seen | `vln-r2r-fstta-feedtta-postfix-v1-seed0-refine-04-fstta-0004-goat-r2r-ac37ab8779` | `b2e37dd387fd8df4543dcfcbac7c79592f88d739` | `df7316d7e83960a5c32b518b9f73a292ccfba4b33b82a33527ff663185b348f2` |
| GOAT | fstta | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-02-goat-01-fstta-850c82e8c9` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `153ba3e08a1996e28751c524e5f90f16bb6009df17af73c14893258937964c1c` |
| GOAT | eam | val_seen | `vln-r2r-eam-atena-formal-confirm-v1-seed0-refine-04-eam-0000-goat-r2r-01a184b229` | `ed61007ba0cd76a39f9a9dcbbd5f70dfcfffef36` | `c3c1ba40611a511976c382c238529de3810ca748fd8f717184612d2e5f7ee6d4` |
| GOAT | eam | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-02-goat-02-eam-1d15221d90` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `bfcf76c8415fb5fdedf2963c8d0bc40dad12682aa6ff4370b7a8e8499330793c` |
| GOAT | feedtta | val_seen | `vln-r2r-targeted-gap-expand-v2-seed0-refine-01-feedtta-0031-goat-r2r-cdc9c472c5` | `ef4f3e37fca3b3057cb2a2c365352b70633bd22b` | `1e3c4afa23a38543e955a6684097dbed88ae384e05b09a1ee8e742a8c723c5ab` |
| GOAT | feedtta | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-02-goat-03-feedtta-926573d63f` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `79eb3c7d6be226de8b8cd2ef75567eae7295685ab0bd01dd481c96e78b4e0ad8` |
| GOAT | atena | val_seen | `vln-r2r-targeted-gap-expand-v2-seed0-refine-02-atena-0022-goat-r2r-4aa391d47e` | `ef4f3e37fca3b3057cb2a2c365352b70633bd22b` | `9c42cbb78e22adfdabe93f875aa79ab8ea51c15fb8f3b723e4b04b7b12aecd46` |
| GOAT | atena | val_unseen | `vln-r2r-val-unseen-frozen-eval-v1-seed0-frozen-02-goat-04-atena-ceebf08428` | `aef150fe0130a8a34bced1f296ee0b987f4ade6f` | `e9aca6d4efcf27d6f1a1f58949cc9eaac6ce584daef7f647fdffc117b8fb3199` |
