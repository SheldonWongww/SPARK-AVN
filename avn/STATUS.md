# AVN status

Current phase: implementation recovery and reproducibility validation.

- Source checkpoints for SMT+Audio and ENMuS are locally available but require provenance manifests.
- Tent, FSTTA, EAM, and FeedTTA prototypes exist.
- ATENA has a paper/official-aligned argmax prototype.  The active AVN campaign
  deliberately selects the reviewed task-native sampled-action variant and
  enables its run-specific preflight gate without changing the shared default.
- The active SMT+Audio development campaign is
  `experiments/smt_audio_four_method_val_search_v1.json`: 118 validation jobs
  for EAM, FeedTTA, the explicitly named ATENA-AVN(sample) port, and IDEA.
  Single-source precedes multi-source independently within each fixed GPU lane.
- AVN uses its native sampled-action protocol throughout this campaign.  The
  historical full ATENA argmax grid remains available only as a superseded
  protocol; it is not part of the active campaign.
- No TTA result is considered formal until a run manifest and reproducible summary are produced.
- Legacy ENMuS metrics are retained under `results/legacy/` and excluded from the main table.
