# Vision-Language Navigation

VLN remains isolated from AVN and ObjectNav. The planned model coverage is:

- Discrete R2R/REVERIE: DUET, HAMT, and GOAT.
- Continuous R2R-CE: ETPNav and BEVBert.
- StreamVLN is retained as a continuous-action candidate pending benchmark and
  evaluation-protocol validation.

Active, Git-tracked source snapshots live under `baselines/`. Pinned,
read-only upstream working trees also live under
`references/repos/navigation/vln/` and remain ignored by the top-level Git
repository. URLs, commits, permission basis, and asset links are recorded in
`manifests/upstream_repositories.json`. Reproduce the upstream reference
checkouts without downloading datasets or checkpoints with:

```bash
python3 vln/scripts/fetch_upstreams.py
```

Research modifications must be made in `baselines/`, not inside the read-only
reference working trees. Dataset and checkpoint hashes will be added to the
manifest after those assets are selected and downloaded. The active export
uses the workspace LF policy and omits three upstream-tracked Python bytecode
cache files as well as MP3D connectivity metadata bundled by ETPNav and
BEVBert.
