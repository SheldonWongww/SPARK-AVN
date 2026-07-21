# AVN data

Large files in this directory are intentionally ignored by Git. The default root can be overridden with `NAVTTA_AVN_DATA`.

Expected layout:

- `datasets/train/single_source/`: single-source training and validation definitions
- `datasets/train/multi_source/`: multi-source training and validation definitions
- `datasets/tta_test/`: fixed single-source and multi-source TTA test definitions
- `scene_datasets/`: Matterport3D scene assets; keep the downloaded directory name unchanged
- `scene_observations/`: cached MP3D observations; keep the downloaded directory name unchanged
- `metadata/`: SoundSpaces navigation graphs and scene metadata
- `sounds/`: source, distractor, and noise audio
- `binaural_rirs/`: SoundSpaces room impulse responses; intentionally empty on machines without the several-hundred-GB RIR dataset
- `manifests/`: tracked versions, URLs, licenses, checksums, and availability

The baseline-local `data/` directories expose these assets through symlinks so
SMT+Audio and ENMuS share one physical copy. Dataset directories and their
contents must not be committed to Git.

After placing the same named directories under `avn/data/` on a new machine,
recreate the baseline-local links with:

```bash
python3 avn/scripts/link_local_data.py
```
