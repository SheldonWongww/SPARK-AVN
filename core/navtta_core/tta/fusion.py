#!/usr/bin/env python3
"""Reusable IDEA fusion bindings and offline source-statistics artifacts.

Evaluation is deliberately fail closed: source moments must come from a
digest-pinned artifact collected on source-training trajectories.  Target
observations are never used to initialise or update the source anchor.
"""
import contextlib
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .idea import IDEAFusionProtocol


SOURCE_STATS_SCHEMA = "navtta.idea.source_statistics"
SOURCE_STATS_VERSION = 1
SOURCE_STATS_SCOPE = "all_steps_all_valid_tokens"
SOURCE_STATS_ESTIMATOR = "global_sum_sumsq_count_sample_std"
SOURCE_STATS_EPS = 1e-6
_REQUIRED_PROVENANCE = (
    "checkpoint_sha256",
    "dataset",
    "dataset_version",
    "split",
)


def file_sha256(path, chunk_size=1024 * 1024):
    """Return the SHA256 digest of a local artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _padding_to_valid(mask):
    """Convert a Transformer padding mask to a boolean valid-token mask."""
    if mask.dtype == torch.bool:
        return ~mask
    # PyTorch float key-padding masks use 0 for valid and -inf/large negative
    # for padding.  Treat any nonzero value as padding so malformed masks fail
    # conservatively rather than leaking padded tokens into Eq. (4.1).
    return mask == 0


def _normalise_tokens_and_mask(tokens, valid_mask=None):
    """Return flattened valid ``[tokens, C]`` values for 2-D/3-D layouts."""
    if not torch.is_tensor(tokens) or tokens.ndim not in (2, 3):
        raise ValueError("IDEA tokens must have shape [N,C] or [B,N,C]")
    if tokens.ndim == 2:
        if valid_mask is None:
            valid_mask = torch.ones(
                tokens.shape[0], dtype=torch.bool, device=tokens.device
            )
        if tuple(valid_mask.shape) != (tokens.shape[0],):
            raise ValueError("IDEA valid mask must match the token dimension")
        return tokens[valid_mask.to(device=tokens.device, dtype=torch.bool)]

    if valid_mask is None:
        valid_mask = torch.ones(
            tokens.shape[:2], dtype=torch.bool, device=tokens.device
        )
    if tuple(valid_mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError("IDEA valid mask must match [batch, tokens]")
    return tokens[valid_mask.to(device=tokens.device, dtype=torch.bool)]


def pool_tokens_to_stats(tokens, valid_mask=None):
    """Pool all valid tokens globally into a mean and sample standard deviation.

    ``tokens`` may be ``[N,C]`` or ``[B,N,C]``.  Unlike an average of per-step
    statistics, this function treats every valid token as one observation.  The
    same sum/sumsq/count contract is used by the offline artifact collector.
    """
    values = _normalise_tokens_and_mask(tokens, valid_mask)
    if values.shape[0] == 0:
        raise ValueError("cannot pool IDEA statistics over zero valid tokens")
    mu = values.mean(dim=0)
    if values.shape[0] < 2:
        variance = torch.zeros_like(mu)
    else:
        variance = values.var(dim=0, unbiased=True).clamp_min(0.0)
    # IDEA Appendix Eq. (41) adds epsilon *inside* the square root.  Keeping
    # this identical for online and offline moments is important: otherwise a
    # constant feature dimension has sigma=0 in the source anchor but
    # sigma=1e-3 online (or vice versa), creating an artificial domain shift.
    sigma = torch.sqrt(variance + SOURCE_STATS_EPS)
    return mu, sigma


class SourceStatisticsAccumulator:
    """Offline global-moment collector for source-training trajectories.

    Call :meth:`begin_trajectory` once per source trajectory and :meth:`update`
    for every navigation step.  Each update receives raw per-layer activations,
    not already-pooled per-step statistics.  Consequently the saved artifact
    contains exact global ``sum``, ``sumsq`` and ``count`` values over all valid
    tokens from all steps.

    This class is intentionally *not* a target-stream warmup accumulator.
    Runtime protocols only consume :class:`SourceStatisticsArtifact` instances.
    """

    def __init__(self, num_layers, feature_dim=None):
        self.num_layers = int(num_layers)
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        if self.num_layers < 1:
            raise ValueError("source-statistics collector needs at least one layer")
        self._sums = None
        self._sumsq = None
        self._counts = [0 for _ in range(self.num_layers)]
        self._trajectory_ids = set()

    @property
    def ready(self):
        return self._sums is not None and all(count > 0 for count in self._counts)

    @property
    def trajectory_count(self):
        return len(self._trajectory_ids)

    def begin_trajectory(self, trajectory_id):
        identifier = str(trajectory_id)
        if not identifier:
            raise ValueError("source trajectory id cannot be empty")
        if identifier in self._trajectory_ids:
            raise ValueError("duplicate source trajectory id: {}".format(identifier))
        self._trajectory_ids.add(identifier)

    def update(self, layer_tokens, valid_masks=None):
        if len(layer_tokens) != self.num_layers:
            raise ValueError(
                "source collector received {} layers, expected {}".format(
                    len(layer_tokens), self.num_layers
                )
            )
        if valid_masks is None:
            valid_masks = [None] * self.num_layers
        elif torch.is_tensor(valid_masks):
            valid_masks = [valid_masks] * self.num_layers
        if len(valid_masks) != self.num_layers:
            raise ValueError("source collector valid-mask count must match layers")

        values_by_layer = [
            _normalise_tokens_and_mask(tokens, mask).detach().to(
                device="cpu", dtype=torch.float64
            )
            for tokens, mask in zip(layer_tokens, valid_masks)
        ]
        if any(values.shape[0] == 0 for values in values_by_layer):
            raise ValueError("source collector step has a layer with no valid tokens")
        widths = {int(values.shape[1]) for values in values_by_layer}
        if len(widths) != 1:
            raise ValueError("source collector layers have inconsistent feature widths")
        width = widths.pop()
        if self.feature_dim is None:
            self.feature_dim = width
        if width != self.feature_dim:
            raise ValueError(
                "source collector feature width {} != expected {}".format(
                    width, self.feature_dim
                )
            )
        if self._sums is None:
            self._sums = [torch.zeros(width, dtype=torch.float64)
                          for _ in range(self.num_layers)]
            self._sumsq = [torch.zeros(width, dtype=torch.float64)
                           for _ in range(self.num_layers)]
        for index, values in enumerate(values_by_layer):
            self._sums[index] += values.sum(dim=0)
            self._sumsq[index] += values.square().sum(dim=0)
            self._counts[index] += int(values.shape[0])

    def payload(self, provenance, expected_trajectory_count=128):
        if not self.ready:
            raise RuntimeError("source-statistics collector has no complete moments")
        if self.trajectory_count == 0:
            raise RuntimeError(
                "source-statistics collection requires recorded trajectory ids"
            )
        if (
            expected_trajectory_count is not None
            and self.trajectory_count != int(expected_trajectory_count)
        ):
            raise ValueError(
                "source-statistics artifact requires {} trajectories, got {}"
                .format(expected_trajectory_count, self.trajectory_count)
            )
        provenance = dict(provenance or {})
        missing = [key for key in _REQUIRED_PROVENANCE if not provenance.get(key)]
        if missing:
            raise ValueError(
                "source-statistics provenance is missing: {}".format(
                    ", ".join(missing)
                )
            )
        if not _is_sha256(provenance["checkpoint_sha256"]):
            raise ValueError("checkpoint_sha256 must be a 64-character SHA256")
        if "train" not in str(provenance["split"]).lower():
            raise ValueError("source-statistics split must be a training split")
        trajectory_ids = sorted(self._trajectory_ids)
        ids_bytes = json.dumps(
            trajectory_ids, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        provenance["trajectory_count"] = self.trajectory_count
        provenance["trajectory_ids"] = trajectory_ids
        provenance["trajectory_ids_sha256"] = hashlib.sha256(ids_bytes).hexdigest()
        provenance["collection_scope"] = SOURCE_STATS_SCOPE
        return {
            "schema": SOURCE_STATS_SCHEMA,
            "version": SOURCE_STATS_VERSION,
            "feature_dim": self.feature_dim,
            "num_layers": self.num_layers,
            "moment_estimator": SOURCE_STATS_ESTIMATOR,
            "provenance": provenance,
            "layers": [
                {
                    "count": self._counts[index],
                    "sum": self._sums[index].tolist(),
                    "sumsq": self._sumsq[index].tolist(),
                }
                for index in range(self.num_layers)
            ],
        }

    def save(self, path, provenance, expected_trajectory_count=128):
        """Write a ``.json`` or ``.pt`` artifact and return its file SHA256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payload(provenance, expected_trajectory_count)
        if path.suffix.lower() == ".json":
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        elif path.suffix.lower() in (".pt", ".pth"):
            torch.save(payload, path)
        else:
            raise ValueError("source-statistics artifact must end in .json or .pt")
        return file_sha256(path)


class SourceStatisticsArtifact:
    """Validated, digest-pinned offline source moments used during evaluation."""

    def __init__(self, payload, sha256, expected_feature_dim=None,
                 expected_num_layers=None, expected_trajectory_count=128,
                 expected_provenance=None):
        if payload.get("schema") != SOURCE_STATS_SCHEMA:
            raise ValueError("invalid IDEA source-statistics schema")
        if int(payload.get("version", -1)) != SOURCE_STATS_VERSION:
            raise ValueError("unsupported IDEA source-statistics version")
        if payload.get("moment_estimator") != SOURCE_STATS_ESTIMATOR:
            raise ValueError("invalid IDEA global-moment estimator")
        self.feature_dim = int(payload.get("feature_dim", 0))
        self.num_layers = int(payload.get("num_layers", 0))
        if self.feature_dim < 1 or self.num_layers < 1:
            raise ValueError("invalid source-statistics dimensions")
        if expected_feature_dim is not None and self.feature_dim != int(expected_feature_dim):
            raise ValueError("source-statistics feature_dim does not match protocol")
        if expected_num_layers is not None and self.num_layers != int(expected_num_layers):
            raise ValueError("source-statistics num_layers does not match protocol")

        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("source-statistics provenance must be an object")
        missing = [key for key in _REQUIRED_PROVENANCE if not provenance.get(key)]
        if missing:
            raise ValueError(
                "source-statistics provenance is missing: {}".format(
                    ", ".join(missing)
                )
            )
        for key in ("checkpoint_sha256", "trajectory_ids_sha256"):
            if not _is_sha256(provenance.get(key)):
                raise ValueError("{} must be a 64-character SHA256".format(key))
        if provenance.get("collection_scope") != SOURCE_STATS_SCOPE:
            raise ValueError(
                "source statistics were not pooled over all valid source tokens"
            )
        if "train" not in str(provenance["split"]).lower():
            raise ValueError("source-statistics artifact is not from a training split")
        trajectory_count = int(provenance.get("trajectory_count", -1))
        trajectory_ids = provenance.get("trajectory_ids")
        if (
            not isinstance(trajectory_ids, list)
            or len(trajectory_ids) != trajectory_count
            or len(set(map(str, trajectory_ids))) != trajectory_count
        ):
            raise ValueError("source-statistics trajectory ids are incomplete")
        ids_bytes = json.dumps(
            sorted(map(str, trajectory_ids)),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(ids_bytes).hexdigest() != provenance["trajectory_ids_sha256"]:
            raise ValueError("source-statistics trajectory-id digest mismatch")
        if (
            expected_trajectory_count is not None
            and trajectory_count != int(expected_trajectory_count)
        ):
            raise ValueError(
                "source-statistics trajectory_count {} != required {}".format(
                    trajectory_count, expected_trajectory_count
                )
            )
        for key, expected_value in dict(expected_provenance or {}).items():
            if provenance.get(key) != expected_value:
                raise ValueError(
                    "source-statistics provenance mismatch for {}: {!r} != {!r}"
                    .format(key, provenance.get(key), expected_value)
                )

        layers = payload.get("layers")
        if not isinstance(layers, list) or len(layers) != self.num_layers:
            raise ValueError("source-statistics layer count is invalid")
        self._moments = []
        for layer in layers:
            count = int(layer.get("count", 0))
            total = torch.as_tensor(layer.get("sum"), dtype=torch.float64)
            total_sq = torch.as_tensor(layer.get("sumsq"), dtype=torch.float64)
            if count < 1 or total.shape != (self.feature_dim,) or total_sq.shape != total.shape:
                raise ValueError("invalid source-statistics layer moments")
            if not bool(torch.isfinite(total).all() and torch.isfinite(total_sq).all()):
                raise ValueError("source-statistics moments must be finite")
            self._moments.append((count, total, total_sq))
        self.provenance = dict(provenance)
        self.sha256 = str(sha256)

    @classmethod
    def load(cls, path, expected_sha256, **expected):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                "required offline IDEA source-statistics artifact is missing: {}"
                .format(path)
            )
        if not _is_sha256(expected_sha256):
            raise ValueError("IDEA source-statistics expected SHA256 is required")
        actual = file_sha256(path)
        if actual.lower() != str(expected_sha256).lower():
            raise ValueError(
                "IDEA source-statistics SHA256 mismatch: expected {}, got {}"
                .format(expected_sha256, actual)
            )
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in (".pt", ".pth"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - older supported torch
                payload = torch.load(path, map_location="cpu")
        else:
            raise ValueError("source-statistics artifact must end in .json or .pt")
        return cls(payload, actual, **expected)

    def statistics(self, device, dtype):
        stats = []
        for count, total, total_sq in self._moments:
            mean = total / count
            if count < 2:
                variance = torch.zeros_like(mean)
            else:
                centered = (total_sq - total.square() / count).clamp_min(0.0)
                variance = centered / (count - 1)
            stats.append((
                mean.to(device=device, dtype=dtype),
                torch.sqrt(variance + SOURCE_STATS_EPS).to(
                    device=device, dtype=dtype
                ),
            ))
        return stats


def load_source_statistics_artifact(path, expected_sha256, **expected):
    return SourceStatisticsArtifact.load(path, expected_sha256, **expected)


def build_multiscale_memory_key_padding_mask(
    src_key_padding_mask, pool_ratios=(4, 8, 16, 32)
):
    """Build ENMuS' ``[base, /4, /8, /16, /32]`` decoder memory mask.

    Input uses PyTorch's convention (``True`` means padding).  A pooled token is
    valid only when every source token in its pooling window is valid, matching
    ENMuS ``floor(avg_pool1d(valid))`` exactly.
    """
    if src_key_padding_mask.ndim != 2:
        raise ValueError("multiscale source padding mask must have shape [B,S]")
    ratios = tuple(int(value) for value in pool_ratios)
    if not ratios or ratios[0] < 1:
        raise ValueError("pool_ratios must be nonempty positive integers")
    valid = (~src_key_padding_mask.bool()).to(dtype=torch.float32)
    masks = [valid]
    previous_ratio = 1
    for ratio in ratios:
        if ratio % previous_ratio:
            raise ValueError("pool_ratios must be successively divisible")
        factor = ratio // previous_ratio
        if masks[-1].shape[1] % factor:
            raise ValueError(
                "source length is not divisible by multiscale pool ratio {}"
                .format(ratio)
            )
        pooled = torch.floor(F.avg_pool1d(masks[-1], factor, factor))
        masks.append(pooled)
        previous_ratio = ratio
    return ~torch.cat(masks, dim=1).bool()


def collect_source_statistics(
    protocol,
    trajectories,
    output_path,
    provenance,
    expected_trajectory_count=128,
):
    """Run the task-agnostic offline source-statistics collection loop.

    ``protocol`` must have been constructed in explicit source-collection mode.
    ``trajectories`` yields ``(trajectory_id, steps)`` pairs, where every item in
    ``steps`` is the same immutable ``policy_inputs`` snapshot accepted by the
    protocol at evaluation time.  Task runners remain responsible for producing
    those snapshots from the source-training split; this helper owns exact
    trajectory accounting, global moment accumulation, serialization and digest.
    """
    if not bool(getattr(protocol, "source_collection", False)):
        raise ValueError("offline collection requires a source_collection protocol")
    accumulator = SourceStatisticsAccumulator(
        protocol.num_layers, protocol.feature_dim
    )
    for trajectory_id, steps in trajectories:
        accumulator.begin_trajectory(trajectory_id)
        step_count = 0
        for policy_inputs in steps:
            protocol.collect_source_step(policy_inputs, accumulator)
            step_count += 1
        if step_count == 0:
            raise ValueError(
                "source trajectory {!r} contains no navigation steps".format(
                    trajectory_id
                )
            )
    return accumulator.save(
        output_path,
        provenance,
        expected_trajectory_count=expected_trajectory_count,
    )


class SourceStatisticsCollectionSession:
    """Online driver used by task evaluators to collect exactly N source episodes.

    Collection mode is deliberately separate from IDEA evaluation.  It observes
    prompt-free policy snapshots while the frozen source policy chooses native
    actions, records real task episode/instruction ids, and writes the artifact
    only after ``expected_trajectory_count`` completed trajectories.
    """

    def __init__(self, protocol, output_path, provenance,
                 expected_trajectory_count=128):
        if not bool(getattr(protocol, "source_collection", False)):
            raise ValueError("collection session requires source_collection=True")
        self.protocol = protocol
        self.output_path = str(output_path)
        if not self.output_path:
            raise ValueError("source-statistics collection output path is required")
        if Path(self.output_path).suffix.lower() not in (".json", ".pt", ".pth"):
            raise ValueError("source-statistics output must end in .json or .pt")
        if Path(self.output_path).exists():
            raise FileExistsError(
                "refusing to overwrite source-statistics artifact: {}".format(
                    self.output_path
                )
            )
        self.provenance = dict(provenance or {})
        missing = [key for key in _REQUIRED_PROVENANCE
                   if not self.provenance.get(key)]
        if missing:
            raise ValueError(
                "source-statistics provenance is missing: {}".format(
                    ", ".join(missing)
                )
            )
        if not _is_sha256(self.provenance["checkpoint_sha256"]):
            raise ValueError("source checkpoint SHA256 is invalid")
        if "train" not in str(self.provenance["split"]).lower():
            raise ValueError("source-statistics collection requires a training split")
        self.expected_trajectory_count = int(expected_trajectory_count)
        if self.expected_trajectory_count != 128:
            raise ValueError("canonical IDEA collection requires exactly 128 trajectories")
        self.accumulator = SourceStatisticsAccumulator(
            protocol.num_layers, protocol.feature_dim
        )
        self.current_trajectory = None
        self.current_step_count = 0
        self.artifact_sha256 = None

    @property
    def complete(self):
        return self.artifact_sha256 is not None

    def reset(self):
        if Path(self.output_path).exists():
            raise FileExistsError(
                "refusing to reset collection over an existing artifact: {}"
                .format(self.output_path)
            )
        self.accumulator = SourceStatisticsAccumulator(
            self.protocol.num_layers, self.protocol.feature_dim
        )
        self.current_trajectory = None
        self.current_step_count = 0
        self.artifact_sha256 = None

    def begin_trajectory(self, trajectory_id):
        if self.current_trajectory is not None:
            raise RuntimeError("source trajectory started before the prior one ended")
        if self.complete:
            return False
        self.accumulator.begin_trajectory(trajectory_id)
        self.current_trajectory = str(trajectory_id)
        self.current_step_count = 0
        return True

    def observe_step(self, policy_inputs):
        if self.complete:
            return
        if self.current_trajectory is None:
            raise RuntimeError("source step observed outside a trajectory")
        self.protocol.collect_source_step(policy_inputs, self.accumulator)
        self.current_step_count += 1

    def end_trajectory(self):
        if self.complete:
            return self.artifact_sha256
        if self.current_trajectory is None:
            raise RuntimeError("source trajectory ended before it started")
        if self.current_step_count < 1:
            raise ValueError("source trajectory contains no navigation steps")
        self.current_trajectory = None
        self.current_step_count = 0
        if self.accumulator.trajectory_count == self.expected_trajectory_count:
            self.artifact_sha256 = self.accumulator.save(
                self.output_path,
                self.provenance,
                expected_trajectory_count=self.expected_trajectory_count,
            )
        return self.artifact_sha256

    def diagnostics(self):
        return {
            "mode": "idea_source_statistics_collection",
            "collection_scope": SOURCE_STATS_SCOPE,
            "trajectory_count": self.accumulator.trajectory_count,
            "expected_trajectory_count": self.expected_trajectory_count,
            "complete": self.complete,
            "output_path": self.output_path,
            "artifact_sha256": self.artifact_sha256,
            "provenance": dict(self.provenance),
        }


class TransformerFusionProtocol(IDEAFusionProtocol):
    """IDEA protocol for a seq-first ``torch.nn.Transformer`` fusion core.

    Evaluation construction requires ``source_stats_path`` and
    ``source_stats_sha256``.  Offline collection is an explicit, separate mode
    selected with ``source_collection=True`` and can only call
    :meth:`collect_source_step`, never :meth:`source_statistics`.

    ``pad_to_multiple=32`` plus
    ``memory_key_padding_mask_builder=build_multiscale_memory_key_padding_mask``
    supports ENMuS/MSMT.  The extended decoder mask is rebuilt from the prompt-
    and padding-aware source mask instead of incorrectly prepending columns to
    the already pooled memory mask.
    """

    _POSITIONAL = {
        "src": 0,
        "tgt": 1,
        "src_mask": 2,
        "tgt_mask": 3,
        "memory_mask": 4,
        "src_key_padding_mask": 5,
        "tgt_key_padding_mask": 6,
        "memory_key_padding_mask": 7,
    }

    def __init__(
        self,
        transformer,
        forward_logits,
        feature_dim,
        num_layers=None,
        source_stats_path=None,
        source_stats_sha256=None,
        expected_source_trajectories=128,
        expected_source_provenance=None,
        source_collection=False,
        pad_to_multiple=None,
        memory_key_padding_mask_builder=None,
    ):
        if bool(getattr(transformer, "batch_first", False)):
            raise ValueError("IDEA TransformerFusionProtocol requires seq-first input")
        self.transformer = transformer
        self.forward_logits = forward_logits
        self._feature_dim = int(feature_dim)
        encoder_layers = list(transformer.encoder.layers)
        if not encoder_layers:
            raise ValueError("transformer encoder has no layers to align")
        self._num_layers = len(encoder_layers) if num_layers is None else int(num_layers)
        if self._num_layers < 1 or self._num_layers > len(encoder_layers):
            raise ValueError("requested IDEA align-layer count is invalid")
        self._aligned_layers = encoder_layers[-self._num_layers:]
        self.source_collection = bool(source_collection)
        if self.source_collection:
            if source_stats_path is not None or source_stats_sha256 is not None:
                raise ValueError("source collection mode cannot consume an artifact")
            self._source = None
        else:
            if source_stats_path is None or source_stats_sha256 is None:
                raise ValueError(
                    "IDEA evaluation requires source_stats_path and "
                    "source_stats_sha256; target warmup is forbidden"
                )
            self._source = SourceStatisticsArtifact.load(
                source_stats_path,
                source_stats_sha256,
                expected_feature_dim=self._feature_dim,
                expected_num_layers=self._num_layers,
                expected_trajectory_count=expected_source_trajectories,
                expected_provenance=expected_source_provenance,
            )
        self.pad_to_multiple = (
            None if pad_to_multiple is None else int(pad_to_multiple)
        )
        if self.pad_to_multiple is not None and self.pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be positive")
        self.memory_key_padding_mask_builder = memory_key_padding_mask_builder
        self._active_prompt = None
        self._fisher_probe = False
        self._num_prompt = 0
        self._captured = {}
        self._stats_valid_mask = None

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def source_statistics_metadata(self):
        if self._source is None:
            return {"mode": "offline_collection"}
        return {
            "mode": "offline_artifact",
            "schema": SOURCE_STATS_SCHEMA,
            "moment_estimator": SOURCE_STATS_ESTIMATOR,
            "sha256": self._source.sha256,
            "trajectory_count": self._source.provenance["trajectory_count"],
            "provenance": dict(self._source.provenance),
        }

    @classmethod
    def for_source_collection(cls, transformer, forward_logits, feature_dim,
                              num_layers=None, **kwargs):
        return cls(
            transformer,
            forward_logits,
            feature_dim,
            num_layers=num_layers,
            source_collection=True,
            **kwargs,
        )

    @classmethod
    def _get(cls, args, kwargs, name):
        index = cls._POSITIONAL[name]
        return args[index] if index < len(args) else kwargs.get(name)

    @classmethod
    def _set(cls, args, kwargs, name, value):
        index = cls._POSITIONAL[name]
        if index < len(args):
            args[index] = value
        else:
            kwargs[name] = value

    def _prompt_pre_hook(self, module, args, kwargs):
        args = list(args)
        src = self._get(args, kwargs, "src")
        if src is None or src.ndim != 3:
            raise ValueError("IDEA transformer source must be [S,B,C]")
        if self._fisher_probe:
            src = src.detach().requires_grad_(True)
        original_length, batch, width = src.shape
        if width != self._feature_dim:
            raise ValueError("IDEA transformer feature width mismatch")

        padding_mask = self._get(args, kwargs, "src_key_padding_mask")
        if padding_mask is None:
            original_padding = torch.zeros(
                batch, original_length, dtype=torch.bool, device=src.device
            )
            mask_dtype = torch.bool
        else:
            if tuple(padding_mask.shape) != (batch, original_length):
                raise ValueError("src_key_padding_mask must have shape [B,S]")
            if padding_mask.dtype != torch.bool and not padding_mask.is_floating_point():
                raise ValueError("src_key_padding_mask must be boolean or floating")
            original_padding = ~_padding_to_valid(padding_mask)
            mask_dtype = padding_mask.dtype

        length = 0 if self._active_prompt is None else self._num_prompt
        if length:
            prompt = self._active_prompt.to(device=src.device, dtype=src.dtype)
            prompt_tokens = prompt.unsqueeze(1).expand(length, batch, width)
            src = torch.cat([prompt_tokens, src], dim=0)
        synthetic = 0
        if self.pad_to_multiple is not None:
            synthetic = (-src.shape[0]) % self.pad_to_multiple
            if synthetic:
                src = torch.cat([
                    src,
                    torch.zeros(synthetic, batch, width,
                                dtype=src.dtype, device=src.device),
                ], dim=0)

        prompt_padding = torch.zeros(batch, length, dtype=torch.bool, device=src.device)
        synthetic_padding = torch.ones(
            batch, synthetic, dtype=torch.bool, device=src.device
        )
        effective_padding = torch.cat(
            [prompt_padding, original_padding.bool(), synthetic_padding], dim=1
        )
        self._stats_valid_mask = torch.cat([
            torch.zeros_like(prompt_padding),
            ~original_padding.bool(),
            torch.zeros_like(synthetic_padding),
        ], dim=1)
        self._set(args, kwargs, "src", src)

        changed_length = bool(length or synthetic)
        if changed_length and self._get(args, kwargs, "src_mask") is not None:
            raise ValueError("IDEA cannot safely extend a non-null src_mask")
        if changed_length and self._get(args, kwargs, "memory_mask") is not None:
            raise ValueError("IDEA cannot safely extend a non-null memory_mask")

        if padding_mask is not None or changed_length:
            if mask_dtype == torch.bool:
                updated_src_mask = effective_padding
            else:
                updated_src_mask = torch.zeros(
                    effective_padding.shape, dtype=mask_dtype, device=src.device
                ).masked_fill(effective_padding, float("-inf"))
            self._set(args, kwargs, "src_key_padding_mask", updated_src_mask)

        memory_mask = self._get(args, kwargs, "memory_key_padding_mask")
        if self.memory_key_padding_mask_builder is not None:
            rebuilt = self.memory_key_padding_mask_builder(effective_padding)
            self._set(args, kwargs, "memory_key_padding_mask", rebuilt)
        elif changed_length:
            if memory_mask is not None and memory_mask.shape[1] != original_length:
                raise ValueError(
                    "nonstandard decoder memory mask requires an explicit builder"
                )
            self._set(args, kwargs, "memory_key_padding_mask", effective_padding)
        elif padding_mask is not None and memory_mask is None:
            # Standard Transformer decoder memory has the same sequence layout
            # as the encoder source; preserve padding semantics even when the
            # caller supplied only src_key_padding_mask.
            self._set(args, kwargs, "memory_key_padding_mask", updated_src_mask)
        return tuple(args), kwargs

    def _make_layer_hook(self, index):
        def hook(module, inputs, output):
            self._captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    @contextlib.contextmanager
    def _instrumented(self, prompt, fisher_probe=False):
        self._active_prompt = prompt
        self._fisher_probe = bool(fisher_probe)
        self._num_prompt = 0 if prompt is None else int(prompt.shape[0])
        self._captured = {}
        self._stats_valid_mask = None
        handles = [self.transformer.register_forward_pre_hook(
            self._prompt_pre_hook, with_kwargs=True
        )]
        handles.extend(
            layer.register_forward_hook(self._make_layer_hook(index))
            for index, layer in enumerate(self._aligned_layers)
        )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._active_prompt = None
            self._fisher_probe = False

    def _checked_captures(self):
        if len(self._captured) != self._num_layers:
            raise RuntimeError(
                "captured {} aligned layers, expected {}".format(
                    len(self._captured), self._num_layers
                )
            )
        return [self._captured[index] for index in range(self._num_layers)]

    def _batch_first_captures(self):
        values = []
        for tokens in self._checked_captures():
            if tokens.ndim != 3:
                raise ValueError("transformer aligned activations must be [S,B,C]")
            values.append(tokens.transpose(0, 1))
        return values

    def source_statistics(self, device, dtype):
        if self._source is None:
            raise RuntimeError("source collection mode has no evaluation anchor")
        return self._source.statistics(device, dtype)

    def fused_forward(self, policy_inputs, prompt):
        if self.source_collection:
            raise RuntimeError("collection protocol cannot run IDEA evaluation")
        with self._instrumented(prompt):
            logits = self.forward_logits(policy_inputs)
            captures = self._batch_first_captures()
            stats = [
                pool_tokens_to_stats(tokens, self._stats_valid_mask)
                for tokens in captures
            ]
        return stats, logits

    def fisher_forward(self, policy_inputs):
        if self.source_collection:
            raise RuntimeError("collection protocol cannot run IDEA evaluation")
        with self._instrumented(None, fisher_probe=True):
            logits = self.forward_logits(policy_inputs)
            features = self._checked_captures()
            # Encoder activations are sequence-first [S,B,C], whereas the
            # transformer's padding bookkeeping is batch-first [B,S].  Return
            # masks in the exact leading layout of each activation so the core
            # Fisher trace cannot accidentally include padded/synthetic rows.
            valid = self._stats_valid_mask.transpose(0, 1)
        if not all(feature.requires_grad for feature in features):
            raise RuntimeError("IDEA Fisher activations are not gradient-connected")
        return features, logits, [valid] * self._num_layers

    def collect_source_step(self, policy_inputs, accumulator):
        """Accumulate one offline source step into ``accumulator``."""
        if not self.source_collection:
            raise RuntimeError("collect_source_step requires source_collection=True")
        with torch.no_grad(), self._instrumented(None):
            self.forward_logits(policy_inputs)
            captures = self._batch_first_captures()
            accumulator.update(captures, self._stats_valid_mask)


__all__ = [
    "SOURCE_STATS_SCHEMA",
    "SOURCE_STATS_SCOPE",
    "SOURCE_STATS_ESTIMATOR",
    "SOURCE_STATS_EPS",
    "SourceStatisticsAccumulator",
    "SourceStatisticsArtifact",
    "SourceStatisticsCollectionSession",
    "TransformerFusionProtocol",
    "build_multiscale_memory_key_padding_mask",
    "collect_source_statistics",
    "file_sha256",
    "load_source_statistics_artifact",
    "pool_tokens_to_stats",
]
