# Upstream provenance

- Project: SoundSpaces
- URL: https://github.com/facebookresearch/sound-spaces.git
- Base commit: `287184fd7067a0385558492716355c54875500ee`
- Historical working tree: `references/repos/navigation/avn/sound-spaces`

This active copy retains the pre-refactor AVN training, evaluation, dataset-bridge, and TTA integration changes. Runtime data, checkpoints, archives, nested Git metadata, and caches were excluded during extraction. Shared TTA algorithms now live in `navtta_core`; compatibility imports are retained for older module paths.
