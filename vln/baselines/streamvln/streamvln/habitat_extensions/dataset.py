"""R2R-VLNCE loader that also supports goal-free leaderboard splits.

The public R2R-VLNCE test annotations intentionally omit goals and reference
paths.  Habitat's :class:`VLNEpisode` schema nevertheless requires both
fields, even when no goal-dependent measures are enabled.  This registered
dataset fills only those missing structural fields with the start position so
that an agent can generate a trajectory for submission.  The placeholders
must never be used to compute test metrics.
"""

import json

from habitat.core.registry import registry
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1


@registry.register_dataset(name="R2RVLN-NavTTA-v1")
class SubmissionVLNDatasetV1(VLNDatasetV1):
    """Load public validation data unchanged and goal-free test data safely."""

    def from_json(self, json_str, scenes_dir=None):
        document = json.loads(json_str)
        for episode in document["episodes"]:
            start = list(episode["start_position"])
            if not episode.get("goals"):
                episode["goals"] = [{"position": start}]
            if not episode.get("reference_path"):
                episode["reference_path"] = [start]
        super().from_json(json.dumps(document), scenes_dir=scenes_dir)
