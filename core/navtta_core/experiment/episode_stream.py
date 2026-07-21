"""Utilities for constructing a reproducible online TTA episode stream."""

import os
import random
from collections import defaultdict
from typing import Any, DefaultDict, List, Sequence, Tuple


def _scene_key(episode: Any) -> str:
    """Return a stable scene identifier independent of the dataset root."""
    return os.path.splitext(os.path.basename(str(episode.scene_id)))[0]


def _episode_key(episode: Any) -> Tuple[int, Any, str]:
    """Sort numeric episode ids numerically and all other ids lexically."""
    episode_id = str(episode.episode_id)
    try:
        return 0, int(episode_id), episode_id
    except ValueError:
        return 1, episode_id, episode_id


def build_tta_episode_stream(
    episodes: Sequence[Any],
    episodes_per_scene: int,
    expected_scenes: int,
    seed: int,
    global_shuffle: bool = True,
) -> List[Any]:
    """Select a fixed per-scene set, then optionally permute the global stream.

    Episode selection never depends on ``seed``.  For each scene, episodes are
    sorted by their stable episode id and the first ``episodes_per_scene`` are
    selected.  The seed is used only to permute the resulting global stream.
    """
    if episodes_per_scene <= 0:
        raise ValueError("episodes_per_scene must be positive")
    if expected_scenes <= 0:
        raise ValueError("expected_scenes must be positive")

    episodes_by_scene: DefaultDict[str, List[Any]] = defaultdict(list)
    for episode in episodes:
        episodes_by_scene[_scene_key(episode)].append(episode)

    actual_scenes = len(episodes_by_scene)
    if actual_scenes != expected_scenes:
        raise ValueError(
            "TTA stream expected {} scenes, but loaded {}: {}".format(
                expected_scenes,
                actual_scenes,
                ", ".join(sorted(episodes_by_scene)),
            )
        )

    selected: List[Any] = []
    for scene in sorted(episodes_by_scene):
        scene_episodes = sorted(episodes_by_scene[scene], key=_episode_key)
        if len(scene_episodes) < episodes_per_scene:
            raise ValueError(
                "TTA stream requires {} episodes for scene '{}', but only "
                "{} were loaded".format(
                    episodes_per_scene, scene, len(scene_episodes)
                )
            )
        selected.extend(scene_episodes[:episodes_per_scene])

    if global_shuffle:
        random.Random(int(seed)).shuffle(selected)

    return selected
