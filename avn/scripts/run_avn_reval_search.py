#!/usr/bin/env python3
"""Manifested two-model AVN revaluation and four-round validation search.

The scheduler itself uses only the standard library. Each evaluation receives
its model's Python environment through PATH and runs in a fresh process group.
Results are eligible on technical validity alone; negative Source deltas remain
in the ranking. No historical campaign result is adopted or overwritten.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "avn/experiments/avn_reval_search_v1.json"
SCHEMA = "navtta.avn.reval_search.v1"
BATCH_SCHEMA = "navtta.avn.reval_search.batch.v1"
MODELS = ("smt_audio", "enmus")
SETTINGS = ("single_source", "multi_source")
SEARCH_METHODS = ("feedtta", "atena")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
METRICS = ("reward", "distance_to_goal", "normalized_distance_to_goal",
           "success", "spl", "softspl", "na", "sna", "sws")
POINT_KEYS = {
    "source": {},
    "tent": {"lr": "TTA.LR", "scope": "TTA.NORM_SCOPE",
             "update_interval": "TTA.UPDATE_INTERVAL"},
    "fstta": {"fast_lr": "TTA.LR", "fast_window": "TTA.FSTTA.M",
              "slow_lr": "TTA.FSTTA.LR_SLOW", "slow_window": "TTA.FSTTA.N"},
    "eam": {"lr": "TTA.EAM.LR", "update_interval": "TTA.EAM.UPDATE_INTERVAL"},
    "feedtta": {"lr": "TTA.FEEDTTA.LR", "gamma": "TTA.FEEDTTA.GAMMA",
                "p": "TTA.FEEDTTA.P", "alpha": "TTA.FEEDTTA.ALPHA"},
    "atena": {"lr_query": "TTA.ATENA.LR_QUERY", "lr_self": "TTA.ATENA.LR_SELF",
              "query_threshold": "TTA.ATENA.QUERY_THRESHOLD",
              "mix_lambda": "TTA.ATENA.MIX_LAMBDA",
              "self_loss_weight": "TTA.ATENA.SELF_LOSS_WEIGHT"},
}
METHOD_DEFAULTS = {
    "source": {},
    "tent": {"TTA.OPTIMIZER": "Adam", "TTA.LAST_K_LN": 4,
             "TTA.MAX_UPDATES_PER_EPISODE": -1},
    "fstta": {"TTA.NORM_SCOPE": "last_k_ln", "TTA.LAST_K_LN": 4,
              "TTA.FSTTA.USE_SLOW": True, "TTA.FSTTA.FAST_GRAD_MODE": "concordant",
              "TTA.FSTTA.OPTIMIZER": "AdamW", "TTA.FSTTA.SLOW_OPTIMIZER": "AdamW",
              "TTA.FSTTA.RESET_VAR_HIST_EACH_EPISODE": False},
    "eam": {"TTA.EAM.PARAM_SCOPE": "module_prefixes",
            "TTA.EAM.TRAINABLE_PREFIXES": ["net.smt_state_encoder.transformer", "action_distribution"],
            "TTA.EAM.CONFIDENCE_SCALE": 0.4, "TTA.EAM.MEMORY_SIZE": 32,
            "TTA.EAM.BATCH_SIZE": 8, "TTA.EAM.OPTIMIZER": "Adam",
            "TTA.EAM.MAX_GRAD_NORM": 0.0},
    "feedtta": {"TTA.FEEDTTA.PARAM_SCOPE": "module_prefixes",
                "TTA.FEEDTTA.TRAINABLE_PREFIXES": ["net.smt_state_encoder", "action_distribution"],
                "TTA.FEEDTTA.OPTIMIZER": "Adam", "TTA.FEEDTTA.EPS": "1e-5",
                "TTA.FEEDTTA.SGR_MODE": "paper_main", "TTA.FEEDTTA.SGR_SEED": 0,
                "TTA.FEEDTTA.NORMALIZE_GRADIENT": False,
                "TTA.FEEDTTA.ACTION_SELECTION_PROTOCOL": "sample_from_policy",
                "TTA.FEEDTTA.MAX_GRAD_NORM": 0.0},
    "atena": {"TTA.ATENA.PREFLIGHT_APPROVED": True, "TTA.ATENA.PARAM_SCOPE": "all",
              "TTA.ATENA.ACTION_SELECTION_PROTOCOL": "sample_from_policy",
              "TTA.ATENA.TASK_UPDATE_SCOPE": "replay_reachable_actor_navigation_policy",
              "TTA.ATENA.OPTIMIZER": "AdamW", "TTA.ATENA.WEIGHT_DECAY": 0.01,
              "TTA.ATENA.MAX_GRAD_NORM": 0.0},
}


class CampaignError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def content_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        require(key not in output, "duplicate JSON key: {}".format(key))
        output[key] = value
    return output


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object)
    except (OSError, ValueError) as error:
        raise CampaignError("cannot read {}: {}".format(path, error)) from error


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + "\n")


def capture(command, *, cwd=ROOT, env=None, timeout=60):
    try:
        completed = subprocess.run(command, cwd=str(cwd), env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError("command failed: {}: {}".format(command[0], error)) from error
    require(completed.returncode == 0,
            "{}: {}".format(command[0], completed.stderr.strip() or completed.stdout.strip()))
    return completed.stdout.strip()


@dataclass(frozen=True)
class Job:
    stage: str
    model: str
    method: str
    source_setting: str
    candidate_id: str
    point: dict
    gpu: str
    round: int
    episodes: int

    @property
    def key(self):
        return "/".join((self.stage, self.model, self.method,
                         self.source_setting, self.candidate_id))


def model_paths(model, root=ROOT):
    require(model in MODELS, "unknown model: {}".format(model))
    baseline = root / "avn/baselines" / model
    config = ("ss_baselines/savi/config/tta_avn/{}/smt_audio_tta_test.yaml"
              if model == "smt_audio" else "sen_baselines/enmus/config/{}/enmus_tta_test.yaml")
    names = (("single_best_val.pth", "multi_best_val.pth") if model == "smt_audio"
             else ("single_source_best_val.pth", "multi_source_best_val.pth"))
    checkpoints = {setting: root / "avn/checkpoints/source" / model / filename
                   for setting, filename in zip(SETTINGS, names)}
    auxiliary = {}
    if model == "enmus":
        directory = baseline / "data/pretrained_weights/semantic_audionav/enmus"
        auxiliary = {name: directory / filename for name, filename in (
            ("audio_encoder", "audio_encoder_best_val.pth"),
            ("visual_encoder", "visual_encoder_best_val.pth"),
            ("seld_encoder", "seld_crnn_best_val.h5"))}
    return {"baseline": baseline, "config": config, "checkpoints": checkpoints,
            "auxiliary": auxiliary, "runner": root / "avn/scripts" / ("eval_" + model + ".sh")}


def lane_map(spec):
    return {(lane["model"], lane["method"]): str(lane["gpu"])
            for lane in spec["scheduler"]["lanes"]}


def validate_spec(spec):
    require(spec.get("schema") == SCHEMA, "unsupported campaign schema")
    for key, value in (("seed", 0), ("audio_seed", 0), ("episodes", 2000)):
        require(type(spec.get(key)) is int and spec[key] == value,
                "{} must be {}".format(key, value))
    scheduler = spec.get("scheduler", {})
    require(scheduler.get("rounds") == 4, "search requires four rounds")
    require(scheduler.get("candidates_per_cell_per_round", 3) == 3,
            "each round must contain three candidates per cell")
    lanes = scheduler.get("lanes", [])
    expected_lanes = {(model, method) for model in MODELS for method in SEARCH_METHODS}
    require(len(lanes) == 4 and {(item.get("model"), item.get("method")) for item in lanes}
            == expected_lanes, "four model/method lanes are required")
    require(len({str(item.get("gpu")) for item in lanes}) == 4
            and all(str(item.get("gpu", "")).isdigit() for item in lanes),
            "lanes must use four distinct numeric GPU IDs")
    cells = spec.get("search", {}).get("cells", [])
    expected_cells = {(m, t, s) for m, t in expected_lanes for s in SETTINGS}
    require(len(cells) == 8 and {(c.get("model"), c.get("method"), c.get("source_setting"))
                                for c in cells} == expected_cells,
            "search must contain the eight model/method/source cells")
    for cell in cells:
        candidates = cell.get("candidates", [])
        require(len(candidates) == 12, "each search cell requires exactly 12 candidates")
        ids = [candidate.get("id") for candidate in candidates]
        require(all(isinstance(item, str) and SAFE_ID.fullmatch(item) for item in ids)
                and len(set(ids)) == 12, "candidate IDs must be unique safe names per cell")
    reval = spec.get("revaluation", [])
    expected_reval = {("enmus", method, setting) for method in ("source", "tent", "fstta", "eam")
                      for setting in SETTINGS} | {("smt_audio", "source", setting) for setting in SETTINGS}
    require(len(reval) == 10 and {(c.get("model"), c.get("method"), c.get("source_setting"))
                                 for c in reval} == expected_reval,
            "revaluation must contain ENMuS Source/Tent/FSTTA/EAM and SMT Source controls")
    for job in build_jobs(spec, "all", validated=True):
        expected_overrides(spec, job)
    return spec


def load_spec(path):
    return validate_spec(read_json(path))


def build_jobs(spec, stage="all", *, validated=False):
    if not validated:
        validate_spec(spec)
    require(stage in ("all", "smoke", "reval", "search"), "invalid stage")
    lanes = lane_map(spec)
    jobs = []
    if stage in ("all", "smoke"):
        for model in MODELS:
            for setting in SETTINGS:
                jobs.append(Job("smoke", model, "source", setting, "control", {},
                                lanes[(model, "feedtta")], 0, 20))
        for cell in spec["search"]["cells"]:
            point = dict(cell["candidates"][0])
            jobs.append(Job("smoke", cell["model"], cell["method"], cell["source_setting"],
                            point.pop("id"), point, lanes[(cell["model"], cell["method"])], 0, 20))
    if stage in ("all", "reval"):
        for cell in spec["revaluation"]:
            method = cell["method"]
            lane = "atena" if method in ("tent", "eam") else "feedtta"
            point = dict(cell.get("point", {}))
            identifier = str(point.pop("id", "control" if method == "source" else "reval"))
            require(SAFE_ID.fullmatch(identifier), "unsafe revaluation candidate ID")
            jobs.append(Job("reval", cell["model"], method, cell["source_setting"],
                            identifier, point, lanes[(cell["model"], lane)], 0, spec["episodes"]))
    if stage in ("all", "search"):
        cells = {(c["model"], c["method"], c["source_setting"]): c
                 for c in spec["search"]["cells"]}
        for round_index in range(4):
            for lane in spec["scheduler"]["lanes"]:
                for offset in range(3):
                    for setting in SETTINGS:
                        point = dict(cells[(lane["model"], lane["method"], setting)]
                                     ["candidates"][round_index * 3 + offset])
                        jobs.append(Job("search", lane["model"], lane["method"], setting,
                                        point.pop("id"), point, str(lane["gpu"]), round_index + 1,
                                        spec["episodes"]))
    require(len({job.key for job in jobs}) == len(jobs), "duplicate job key")
    return jobs


def config_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return canonical(value)
    return str(value)


def expected_overrides(spec, job):
    protected = {"TEST_EPISODE_COUNT": str(job.episodes), "NUM_PROCESSES": "1",
                 "EVAL.SPLIT": "val", "EVAL.USE_CKPT_CONFIG": "False",
                 "EVAL.ACTION_SELECTION": "sample", "TTA.EPISODIC": "False", "TTA.STEPS": "1"}
    values = dict(protected)
    values.update({key: config_value(value) for key, value in METHOD_DEFAULTS[job.method].items()})
    protocol = spec.get("protocol", {})
    layers = [protocol.get("fixed_overrides_by_method", {}).get(job.method, {}),
              protocol.get("overrides_by_model", {}).get(job.model, {})]
    allowed = set(POINT_KEYS[job.method]) | {"overrides", "fixed_overrides"}
    require(set(job.point) <= allowed, "unknown point keys for {}: {}".format(job.key, set(job.point) - allowed))
    for key in POINT_KEYS[job.method]:
        require(key in job.point, "{} is missing {}".format(job.key, key))
        value = job.point[key]
        if key == "scope":
            require(value in ("first_ln", "last_ln", "last_k_ln", "ln", "gn", "bn", "all"), "invalid norm scope")
        else:
            number = finite(value, key)
            if key in ("lr", "fast_lr", "slow_lr", "lr_query", "lr_self", "fast_window", "slow_window", "update_interval"):
                require(number > 0, key + " must be positive")
            if key in ("fast_window", "slow_window", "update_interval"):
                require(number.is_integer(), key + " must be an integer")
            if key in ("gamma", "p", "mix_lambda"):
                require(0 <= number <= 1 and (key != "p" or number < 1), "invalid " + key)
            if key in ("query_threshold", "self_loss_weight"):
                require(number >= 0, "invalid " + key)
    layers.append({destination: job.point[key] for key, destination in POINT_KEYS[job.method].items()})
    layers.extend([job.point.get("fixed_overrides", {}), job.point.get("overrides", {})])
    for layer in layers:
        require(isinstance(layer, dict), "config overrides must be KEY: VALUE objects")
        for key, value in layer.items():
            text = config_value(value)
            require(key not in protected or text == protected[key], "cannot override frozen protocol: " + key)
            require(key not in ("SEED", "TASK_CONFIG.SEED", "TTA.METHOD", "EVAL_CKPT_PATH_DIR", "BASE_TASK_CONFIG_PATH")
                    and not key.startswith(("TASK_CONFIG.DATASET.", "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.")),
                    "unsupported identity override: " + key)
            require(not key.startswith("TTA.AUDIT_"), "audit modes are not search candidates")
            values[key] = text
    for method in SEARCH_METHODS:
        if job.method == method:
            require(values["TTA." + method.upper() + ".ACTION_SELECTION_PROTOCOL"] == "sample_from_policy",
                    "feedback methods must optimize executed sampled actions")
    return values


def job_command(spec, job, root=ROOT):
    command = ["bash", str(model_paths(job.model, root)["runner"]), job.source_setting,
               job.method, str(spec["seed"])]
    for key, value in expected_overrides(spec, job).items():
        command.extend((key, value))
    return command


def runtime_files(spec, root=ROOT):
    files = set()
    for directory in ("core/navtta_core", "avn/navtta_avn", "avn/baselines/smt_audio/ss_baselines",
                      "avn/baselines/smt_audio/soundspaces", "avn/baselines/smt_audio/configs",
                      "avn/baselines/enmus/sen_baselines", "avn/baselines/enmus/sen", "avn/baselines/enmus/configs"):
        base = root / directory
        files.update(path for path in base.rglob("*") if path.is_file()
                     and path.suffix in (".py", ".yaml", ".yml") and "__pycache__" not in path.parts)
    for name in ("avn/scripts/run_avn_reval_search.py", "avn/scripts/eval_smt_audio.sh",
                 "avn/scripts/eval_enmus.sh", "avn/scripts/fingerprint_episode_stream.py",
                 "tools/create_run_manifest.py", "tools/finalize_run_manifest.py",
                 "tools/validate_run_manifest.py", "tools/run_manifest_identity.py"):
        files.add(root / name)
    files.update(root / name for name in spec.get("runtime", {}).get("files", []))
    require(all(path.is_file() for path in files), "a runtime dependency is missing")
    return sorted(files)


def fingerprint_stream(dataset, count, seed, root=ROOT):
    module_spec = importlib.util.spec_from_file_location(
        "avn_reval_stream_fingerprint", root / "avn/scripts/fingerprint_episode_stream.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    stream = module.build_stream(dataset, 100, 20, seed)[:count]
    require(len(stream) == count, "incomplete evaluation stream")
    order, content = module.fingerprints(stream)
    ids = [module.scene_key(item) + "/" + str(item["episode_id"]) for item in stream]
    require(len(set(ids)) == count, "duplicate stream episode IDs")
    return {"stream_order_sha256": order, "stream_content_sha256": content, "episode_order": ids}


def interpreter_environment(python, environment=None):
    environment = dict(os.environ if environment is None else environment)
    environment["PATH"] = str(Path(python).absolute().parent) + os.pathsep + environment.get("PATH", "")
    return environment


def interpreter_identity(python, gpus):
    located = shutil.which(str(python))
    require(located is not None, "Python interpreter is unavailable: " + str(python))
    # Do not resolve the symlink before choosing PATH: that loses venv activation.
    python = str(Path(located).absolute())
    environment = interpreter_environment(python)
    python3 = shutil.which("python3", path=environment["PATH"])
    require(python3 is not None, "selected environment does not provide python3")
    prefix = capture([python, "-c", "import sys; print(sys.prefix)"])
    require(capture([python3, "-c", "import sys; print(sys.prefix)"], env=environment) == prefix,
            "PATH python3 does not belong to the requested environment")
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    probe = capture([python3, "-c", "import inspect,json,sys,torch; print(json.dumps({"
                     "'prefix':sys.prefix,'python':sys.version,'torch':torch.__version__,"
                     "'torch_cuda':torch.version.cuda,'cuda':torch.cuda.is_available(),"
                     "'named_modules_remove_duplicate':'remove_duplicate' in inspect.signature(torch.nn.Module.named_modules).parameters,"
                     "'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"],
                    env=environment)
    identity = json.loads(probe)
    require(identity["cuda"] and len(identity["devices"]) == len(gpus),
            "requested GPUs are not available in " + python)
    require(identity["named_modules_remove_duplicate"], "Torch does not support the complete Source state-hash API")
    hash_probe = capture([python3, "-c", "import sys; sys.path.insert(0, {!r}); "
                         "import torch; from navtta_core.tta import module_state_sha256; "
                         "torch.manual_seed(0); model=torch.nn.Linear(2,2); "
                         "model.register_buffer('probe',torch.zeros(1),persistent=False); "
                         "print(module_state_sha256(model))".format(str(ROOT / "core"))], env=environment)
    require(DIGEST.fullmatch(hash_probe), "complete Source state hashing is unsupported by selected environment")
    identity["source_state_hash_probe_sha256"] = hash_probe
    identity.update({"path": python, "python3": str(Path(python3).absolute()),
                     "binary_sha256": sha256_file(python3),
                     "packages": sorted(capture([python3, "-m", "pip", "freeze", "--all"], env=environment).splitlines())})
    return identity


def preflight(spec, spec_path, pythons, root=ROOT):
    commit = capture(["git", "rev-parse", "HEAD"], cwd=root)
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "invalid repository commit")
    files = runtime_files(spec, root)
    identity = {"git_commit": commit, "spec_sha256": sha256_file(spec_path), "spec_path": str(spec_path.resolve()),
                "runtime_files": {str(path.relative_to(root)): sha256_file(path) for path in files},
                "streams": {}, "checkpoints": {}, "auxiliary_checkpoints": {}, "environments": {}}
    identity["runtime_sha256"] = content_digest(identity["runtime_files"])
    for setting in SETTINGS:
        dataset = root / ("avn/data/datasets/tta_test/{}/mp3d/v1/val/val.json.gz".format(setting))
        pinned = spec["data"]["streams"][setting]
        require(sha256_file(dataset) == pinned["dataset_index_sha256"], "dataset index mismatch: " + setting)
        stream = fingerprint_stream(dataset, spec["episodes"], spec["seed"], root)
        for key in ("stream_order_sha256", "stream_content_sha256"):
            require(stream[key] == pinned[key], "{} mismatch: {}".format(setting, key))
        identity["streams"][setting] = {"dataset_index_sha256": pinned["dataset_index_sha256"],
                                        "full": stream, "smoke": fingerprint_stream(dataset, 20, spec["seed"], root)}
    for model in MODELS:
        paths = model_paths(model, root)
        identity["checkpoints"][model] = {}
        for setting, path in paths["checkpoints"].items():
            pinned = spec["checkpoints"][model][setting]
            expected = pinned if isinstance(pinned, str) else pinned["sha256"]
            if isinstance(pinned, dict) and "path" in pinned:
                require((root / pinned["path"]).resolve() == path.resolve(), "checkpoint path differs from eval wrapper")
            require(DIGEST.fullmatch(expected), "invalid checkpoint digest")
            actual = sha256_file(path)
            require(actual == expected, "checkpoint digest mismatch: {}/{}".format(model, setting))
            identity["checkpoints"][model][setting] = actual
        identity["auxiliary_checkpoints"][model] = {name: sha256_file(path)
                                                   for name, path in paths["auxiliary"].items()}
        expected_auxiliary = spec.get("auxiliary_checkpoints", {}).get(model, {})
        require(set(expected_auxiliary) == set(paths["auxiliary"]), "auxiliary checkpoint pins are incomplete: " + model)
        for name, path in paths["auxiliary"].items():
            pinned = expected_auxiliary[name]
            require((root / pinned["path"]).resolve() == path.resolve()
                    and pinned["sha256"] == identity["auxiliary_checkpoints"][model][name],
                    "auxiliary checkpoint mismatch: " + name)
        for setting in SETTINGS:
            require((paths["baseline"] / paths["config"].format(setting)).is_file(), "missing model configuration")
        gpus = [str(lane["gpu"]) for lane in spec["scheduler"]["lanes"] if lane["model"] == model]
        identity["environments"][model] = interpreter_identity(pythons[model], gpus)
    return identity


class BatchLock:
    """An advisory exclusive lock on a stable inode; kernel releases on crash."""
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            self.stream = None
            raise CampaignError("batch is already locked: " + str(self.path)) from error
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(canonical({"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()}))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, *ignored):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def next_attempt_dir(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = directory / ("attempt-{}".format(index))
        try:
            path.mkdir()
            return index, path
        except FileExistsError:
            index += 1


def process_birth(pid):
    try:
        return Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None


def process_alive(pid, birth=None):
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return birth is None or process_birth(pid) == birth


def process_group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def normalized_episode_key(key):
    scene, episode = key.rsplit(",", 1)
    scene = Path(scene.replace("\\", "/")).stem
    return scene + "/" + episode


def finite(value, label):
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise CampaignError("missing/nonnumeric " + label) from error
    require(math.isfinite(number), "nonfinite " + label)
    return number


def close(actual, expected, label):
    require(math.isclose(finite(actual, label), float(expected), rel_tol=1e-10, abs_tol=1e-15),
            label + " mismatch")


def validate_diagnostics(job, diagnostics, overrides):
    require(diagnostics.get("episodes") == job.episodes, "diagnostic episode count mismatch")
    steps = int(diagnostics.get("action_steps", 0))
    require(steps > 0 and diagnostics.get("task_action_selection") == "sample", "invalid action diagnostics")
    updates = int(diagnostics.get("updates", -1))
    require(updates >= 0, "missing update count")
    finite(diagnostics.get("relative_param_drift"), "relative_param_drift")
    names = diagnostics.get("adapted_parameter_names")
    require(isinstance(names, list) and all(isinstance(name, str) for name in names)
            and len(names) == len(set(names)), "invalid adapted parameter names")
    count = diagnostics.get("adapted_parameter_count")
    require(type(count) is int and count >= len(names), "invalid adapted parameter numel")
    if job.method == "source":
        require(updates == 0 and count == 0 and not names, "Source performed adaptation")
        require(diagnostics.get("slow_updates", 0) == 0, "Source performed slow updates")
        close(diagnostics.get("relative_param_drift"), 0, "Source drift")
        before = diagnostics.get("source_model_state_sha256")
        require(diagnostics.get("source_policy_frozen") is True
                and isinstance(before, str) and DIGEST.fullmatch(before)
                and diagnostics.get("final_model_state_sha256") == before, "Source model state was not verified unchanged")
        return
    require(count > 0 and names, "empty adaptation parameter scope")
    require(not any(name.startswith("critic.") for name in names), "value-only critic in adaptation scope")
    for key in ("mean_entropy", "last_entropy"):
        finite(diagnostics.get(key), key)
    if job.method in ("eam", "feedtta"):
        prefix = "TTA." + job.method.upper() + "."
        prefixes = json.loads(overrides[prefix + "TRAINABLE_PREFIXES"])
        require(diagnostics.get("trainable_prefixes") == prefixes, "trainable-prefix mismatch")
        require(diagnostics.get("param_scope") == "module_prefixes", "parameter scope mismatch")
        require(all(any(name == p or name.startswith(p + ".") for p in prefixes) for name in names),
                "adapted parameter lies outside the declared prefixes")
        require(all(any(name == p or name.startswith(p + ".") for name in names) for p in prefixes),
                "a declared trainable prefix has no actual parameter")
        close(diagnostics.get("current_lr"), overrides[prefix + "LR"], "learning rate")
    if job.method == "tent":
        close(diagnostics.get("current_lr"), overrides["TTA.LR"], "Tent LR")
        require(diagnostics.get("update_interval") == int(overrides["TTA.UPDATE_INTERVAL"]), "Tent interval mismatch")
        if overrides.get("TTA.MAX_UPDATES_PER_EPISODE") == "-1":
            require(updates == steps // int(overrides["TTA.UPDATE_INTERVAL"]), "Tent update accounting mismatch")
    elif job.method == "fstta":
        for key, config_key in (("fast_lr", "TTA.LR"), ("slow_lr", "TTA.FSTTA.LR_SLOW"),
                                ("fast_window", "TTA.FSTTA.M"), ("slow_window", "TTA.FSTTA.N")):
            close(diagnostics.get(key), overrides[config_key], key)
        if overrides.get("TTA.FSTTA.USE_SLOW") == "True":
            require(diagnostics.get("completed_slow_windows") == job.episodes // int(overrides["TTA.FSTTA.N"]),
                    "FSTTA slow window accounting mismatch")
    elif job.method == "eam":
        require(diagnostics.get("replay_unit") == "action_step"
                and diagnostics.get("update_timing") == "after_preupdate_action_selection", "EAM replay protocol mismatch")
        close(diagnostics.get("update_interval"), overrides["TTA.EAM.UPDATE_INTERVAL"], "EAM interval")
        batch = int(overrides["TTA.EAM.BATCH_SIZE"])
        interval = int(overrides["TTA.EAM.UPDATE_INTERVAL"])
        attempts = steps // interval - min(steps, batch - 1) // interval
        require(diagnostics.get("short_buffer_behavior") == "warmup_no_update"
                and diagnostics.get("warmup_no_update_steps") == min(steps, batch - 1), "EAM warmup mismatch")
        require(diagnostics.get("update_attempts") == attempts
                and diagnostics.get("replayed_steps") == batch * attempts
                and updates + diagnostics.get("updates_skipped_no_reliable", -1) == attempts, "EAM update accounting mismatch")
    elif job.method == "feedtta":
        require(diagnostics.get("feedback_type") == "binary_episode_success"
                and diagnostics.get("feedback_episodes") == job.episodes, "FeedTTA feedback mismatch")
        require(updates == job.episodes and diagnostics.get("policy_gradient_action") == "task_runner_executed_action",
                "FeedTTA episode update mismatch")
        require(diagnostics.get("sgr_mode") == overrides["TTA.FEEDTTA.SGR_MODE"]
                and diagnostics.get("action_selection_protocol") == "sample_from_policy", "FeedTTA protocol mismatch")
        for key, config_key in (("gamma", "GAMMA"), ("reversal_probability", "P"), ("reversal_scale", "ALPHA")):
            close(diagnostics.get(key), overrides["TTA.FEEDTTA." + config_key], key)
    elif job.method == "atena":
        require(updates == job.episodes and diagnostics.get("replayed_steps") == steps, "ATENA replay count mismatch")
        require(diagnostics.get("action_selection_protocol") == "sample_from_policy"
                and diagnostics.get("pseudo_expert_action") == "executed_task_native_sample", "ATENA action protocol mismatch")
        require(diagnostics.get("replay_reachability_validated") is True
                and diagnostics.get("optimizer_policy_scope_matches_reachable") is True,
                "ATENA replay scope was not verified")
        require(names == diagnostics.get("replay_reachable_parameter_names")
                == diagnostics.get("optimizer_policy_parameter_names"), "ATENA reachable parameter names differ")
        require(diagnostics.get("replay_reachable_parameter_count") == len(names), "ATENA tensor count mismatch")
        candidates = diagnostics.get("replay_reachability_candidate_parameter_names", [])
        unreachable = diagnostics.get("replay_unreachable_parameter_names", [])
        require(set(names).isdisjoint(unreachable) and set(names) | set(unreachable) == set(candidates),
                "ATENA reachable/unreachable scope partition mismatch")
        require(diagnostics.get("preexisting_frozen_policy_parameter_count") == 0
                and diagnostics.get("requires_task_trainability_wiring") is False,
                "ATENA task policy was not made trainable before replay")
        require(diagnostics.get("replay_determinism_validated_episodes") == job.episodes,
                "ATENA deterministic replay was not checked for every episode")
        require(finite(diagnostics.get("max_replay_feature_abs_error"), "replay error") <= 1e-5,
                "ATENA replay error exceeded tolerance")
        queries = diagnostics.get("queries", -1)
        require(type(queries) is int and 0 <= queries <= job.episodes
                and queries + diagnostics.get("self_label_episodes", -1) == job.episodes
                and diagnostics.get("feedback_observed_episodes") == queries, "ATENA query accounting mismatch")
        close(diagnostics.get("query_rate"), queries / job.episodes, "query rate")
        for key in POINT_KEYS["atena"]:
            close(diagnostics.get(key), overrides[POINT_KEYS["atena"][key]], key)


def validate_audio_evidence(run_dir, job, stream, overrides, source_result=None, checkpoint_sha256=None):
    if job.model != "enmus":
        return {}
    tb = run_dir / "raw/model/tb"
    protocol_path, audio_path = tb / "eval_protocol_0.json", tb / "audio_schedule_0.json"
    protocol, audio = read_json(protocol_path), read_json(audio_path)
    require(isinstance(protocol, dict) and isinstance(audio, dict), "ENMuS sidecars must be objects")
    require(protocol.get("schema") == "navtta.avn.enmus_eval_protocol.v1"
            and audio.get("schema") == "navtta.avn.audio_schedule.v1", "ENMuS sidecar schema mismatch")
    require(protocol.get("profile") == overrides["EVAL.PROTOCOL_PROFILE"]
            and audio.get("profile") == protocol["profile"], "ENMuS protocol profile mismatch")
    # These local helpers use only stdlib on this path; importing them does not
    # import the simulator, NumPy, Torch, or either model environment.
    task_package = str(ROOT / "avn")
    if task_package not in sys.path:
        sys.path.insert(0, task_package)
    from navtta_avn.enmus_eval_protocol import build_eval_protocol
    from navtta_avn.audio_schedule import schedule_seed
    resolved = protocol.get("resolved_config")
    require(isinstance(resolved, dict), "missing resolved ENMuS configuration")
    reconstructed = build_eval_protocol(resolved)
    require(reconstructed["semantic_config"] == protocol.get("semantic_config")
            and reconstructed["profile"] == protocol["profile"], "sidecar semantics disagree with resolved model configuration")
    require(resolved.get("TEST_EPISODE_COUNT") == job.episodes and protocol.get("observed_action_count") == 4,
            "resolved episode count/action space mismatch")
    checkpoint_load = protocol.get("checkpoint_load", {})
    loaded_digest = checkpoint_load.get("checkpoint_sha256")
    require(checkpoint_load.get("strict") is True and checkpoint_load.get("missing_keys") == []
            and checkpoint_load.get("unexpected_keys") == [] and isinstance(loaded_digest, str)
            and DIGEST.fullmatch(loaded_digest) and checkpoint_load.get("state_dict_tensors", 0) > 0,
            "ENMuS checkpoint was not loaded strictly")
    if checkpoint_sha256 is not None:
        require(loaded_digest == checkpoint_sha256, "ENMuS loaded checkpoint digest mismatch")
    semantic = protocol.get("semantic_digest")
    require(isinstance(semantic, str) and DIGEST.fullmatch(semantic)
            and content_digest(protocol.get("semantic_config")) == semantic, "ENMuS semantic digest mismatch")
    require(audio.get("semantic_digest") == semantic
            and audio.get("mode") == overrides["TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE"]
            and audio.get("schedule_seed") == int(overrides["TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED"]),
            "audio scheduling protocol mismatch")
    require(protocol.get("source_setting") == audio.get("source_setting") == job.source_setting,
            "ENMuS sidecar source setting mismatch")
    require(audio.get("complete") is True and audio.get("episode_count") == job.episodes
            and audio.get("episode_order") == stream["episode_order"],
            "audio episode count/order mismatch")
    require(audio.get("episode_order_sha256") == content_digest(stream["episode_order"]), "audio episode-order digest mismatch")
    records = audio.get("episodes")
    require(isinstance(records, dict) and set(records) == set(stream["episode_order"]), "audio episodes mismatch")
    for key, record in records.items():
        require(isinstance(record, dict), "invalid audio episode record")
        require(str(record.get("scene_id")) + "/" + str(record.get("episode_id")) == key, "audio episode identity mismatch")
        roles = record.get("roles", {})
        require(isinstance(roles, dict) and roles and "target" in roles, "audio role evidence missing")
        if job.source_setting == "multi_source":
            require("distractor" in roles, "multi-source audio lacks distractor evidence")
        for name, role in roles.items():
            require(isinstance(role, dict) and isinstance(role.get("sha256"), str) and DIGEST.fullmatch(role["sha256"])
                    and isinstance(role.get("length"), int) and role["length"] > 0
                    and 0 <= role.get("active_steps", -1) <= role["length"], "invalid actual audio schedule evidence")
            require(role.get("rng_seed") == schedule_seed(audio["schedule_seed"], job.source_setting,
                    record["scene_id"], record["episode_id"], name), "audio role RNG seed mismatch")
        require(content_digest({name: role["sha256"] for name, role in roles.items()}) == record.get("schedule_sha256"),
                "audio per-episode digest mismatch")
    if source_result is not None:
        source_audio = read_json(source_result["audio_evidence"]["audio_path"])
        require(source_audio.get("semantic_digest") == semantic, "matched Source ENMuS semantics differ")
        require(source_audio.get("mode") == audio["mode"] and source_audio.get("schedule_seed") == audio["schedule_seed"],
                "matched Source audio RNG protocol differs")
        for key in stream["episode_order"]:
            require(source_audio.get("episodes", {}).get(key, {}).get("schedule_sha256") == records[key]["schedule_sha256"],
                    "actual audio schedule differs from matched Source: " + key)
    return {"protocol_path": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
            "audio_path": str(audio_path), "audio_sha256": sha256_file(audio_path), "semantic_digest": semantic}


def validate_run(manifest_path, job, run_tag, spec, identity, source_result=None, root=ROOT):
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest_path.relative_to((root / "avn/results/runs").resolve())
    except ValueError as error:
        raise CampaignError("manifest is outside task run storage") from error
    stream = identity["streams"][job.source_setting]["smoke" if job.stage == "smoke" else "full"]
    command = [sys.executable, str(root / "tools/validate_run_manifest.py"), "--manifest", str(manifest_path),
               "--run-tag", run_tag, "--model", job.model, "--method", job.method,
               "--source-setting", job.source_setting, "--seed", str(spec["seed"]),
               "--git-commit", identity["git_commit"], "--checkpoint-sha256", identity["checkpoints"][job.model][job.source_setting],
               "--stream-order-sha256", stream["stream_order_sha256"], "--stream-content-sha256", stream["stream_content_sha256"],
               "--require-immutable-identity", "--require-result-artifacts"]
    capture(command, cwd=root)
    manifest = read_json(manifest_path)
    paths = model_paths(job.model, root)
    require(manifest.get("config") == paths["config"].format(job.source_setting), "manifest configuration path mismatch")
    require(manifest.get("hardware", {}).get("cuda_visible_devices") == job.gpu, "manifest GPU identity mismatch")
    require(manifest.get("dataset", {}).get("index_sha256") == identity["streams"][job.source_setting]["dataset_index_sha256"],
            "manifest dataset index mismatch")
    actual_auxiliary = {item["name"]: item["sha256"] for item in manifest.get("auxiliary_checkpoints", [])}
    require(actual_auxiliary == identity["auxiliary_checkpoints"][job.model], "auxiliary checkpoint identity mismatch")
    raw = manifest.get("config_overrides")
    require(isinstance(raw, list) and len(raw) % 2 == 0 and len(set(raw[::2])) == len(raw) // 2,
            "invalid/duplicate manifest configuration overrides")
    overrides = expected_overrides(spec, job)
    require(dict(zip(raw[::2], map(str, raw[1::2]))) == overrides, "manifest overrides differ from frozen candidate")
    tb = manifest_path.parent / "raw/model/tb"
    stats_path, diagnostics_path = tb / "val_stats_0.json", tb / "tta_diagnostics_0.json"
    required_artifacts = {stats_path.resolve(), diagnostics_path.resolve()}
    if job.model == "enmus":
        required_artifacts.update({(tb / "eval_protocol_0.json").resolve(), (tb / "audio_schedule_0.json").resolve()})
    actual_artifacts = {Path(item["path"]).resolve() for item in manifest["result_artifacts"]}
    require(required_artifacts <= actual_artifacts, "manifest did not register every required result artifact")
    stats, diagnostics = read_json(stats_path), read_json(diagnostics_path)
    require(isinstance(stats, dict) and len(stats) == job.episodes, "episode metrics are incomplete")
    require(all(isinstance(record, dict) for record in stats.values()), "episode metric records must be objects")
    require(isinstance(diagnostics, dict), "diagnostics must be an object")
    try:
        ids = [normalized_episode_key(key) for key in stats]
    except (ValueError, AttributeError) as error:
        raise CampaignError("invalid episode metric identity") from error
    require(ids == stream["episode_order"], "episode metrics do not follow the pinned complete stream")
    metrics = {key: sum(finite(record.get(key), key) for record in stats.values()) / job.episodes for key in METRICS}
    validate_diagnostics(job, diagnostics, overrides)
    action_counts = diagnostics.get("action_counts")
    require(isinstance(action_counts, list) and len(action_counts) == 4
            and all(type(value) is int and value >= 0 for value in action_counts), "invalid actual action counts")
    action_steps = diagnostics["action_steps"]
    require(sum(action_counts) == action_steps == sum(record["na"] for record in stats.values()),
            "action diagnostics disagree with episode metrics")
    if job.method == "fstta":
        window = int(overrides["TTA.FSTTA.M"])
        require(diagnostics.get("fast_optimizer_attempts") == sum(int(record["na"]) // window for record in stats.values())
                and diagnostics.get("updates") == diagnostics.get("fast_optimizer_attempts"), "FSTTA fast update accounting mismatch")
    if job.method == "feedtta":
        require(diagnostics.get("successful_feedback_episodes") == sum(record["success"] >= 0.5 for record in stats.values()),
                "FeedTTA feedback does not match completed episode outcomes")
    audio_evidence = validate_audio_evidence(manifest_path.parent, job, stream, overrides, source_result,
                                             identity["checkpoints"][job.model][job.source_setting])
    if audio_evidence:
        require(diagnostics.get("eval_semantic_digest") == audio_evidence["semantic_digest"],
                "diagnostic semantic identity disagrees with ENMuS sidecar")
    return {"job": asdict(job), "certified": True, "validated_at": utc_now(),
            "identity_sha256": content_digest(identity), "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path), "run_tag": run_tag,
            "stats": str(stats_path), "stats_sha256": sha256_file(stats_path),
            "diagnostics": str(diagnostics_path), "diagnostics_sha256": sha256_file(diagnostics_path),
            "metrics": metrics, "config_overrides": overrides, "diagnostic_summary": {key: diagnostics[key] for key in (
                "updates", "relative_param_drift", "query_rate", "adapted_parameter_count",
                "replay_reachability_validated") if key in diagnostics},
            "audio_evidence": audio_evidence,
            "matched_source": None if source_result is None else {
                "manifest": source_result["manifest"], "manifest_sha256": source_result["manifest_sha256"]}}


def verify_result_files(result):
    require(result.get("certified") is True, "result is not certified")
    for key in ("manifest", "stats", "diagnostics"):
        require(sha256_file(result[key]) == result[key + "_sha256"], "saved result artifact changed: " + key)
    evidence = result.get("audio_evidence", {})
    for key in ("protocol", "audio"):
        if key + "_path" in evidence:
            require(sha256_file(evidence[key + "_path"]) == evidence[key + "_sha256"], "saved audio evidence changed")


def require_source(result, job, identity, batch_id=None):
    require(result is not None and result.get("certified") is True, "missing certified matched Source for " + job.key)
    source = result.get("job", {})
    require(source.get("method") == "source" and source.get("model") == job.model
            and source.get("source_setting") == job.source_setting and source.get("episodes") == job.episodes
            and source.get("stage") == ("smoke" if job.stage == "smoke" else "reval"),
            "Source does not match model/setting/stage/complete episode count")
    require(result.get("identity_sha256") == content_digest(identity), "Source belongs to a different batch identity")
    if batch_id is not None:
        require(result.get("batch_id") == batch_id, "Source belongs to another batch")


def attempt_tag(batch_id, job, number):
    return "arv1-{}-{}-a{}".format(content_digest(batch_id)[:12], content_digest(job.key)[:12], number)


def rank_results(results):
    return sorted(results, key=lambda result: (-result["metrics"]["spl"], -result["metrics"]["success"],
                   -result["metrics"]["softspl"], result["diagnostic_summary"].get("relative_param_drift", 0.0),
                   result["job"]["candidate_id"]))


@dataclass
class Worker:
    process: object
    output: object
    job: Job
    attempt: dict
    source: object


class Campaign:
    def __init__(self, spec, identity, batch_id, *, resume=False, retry_failed=False, root=ROOT):
        require(SAFE_ID.fullmatch(batch_id), "unsafe batch ID")
        self.spec, self.identity, self.batch_id, self.root = spec, identity, batch_id, root
        self.directory = root / "avn/results/analysis/reval_search" / batch_id
        self.log_directory = root / "avn/results/logs/reval_search" / batch_id
        self.path = self.directory / "batch.json"
        self.jobs = build_jobs(spec, "all")
        self.by_key = {job.key: job for job in self.jobs}
        self.retry_failed, self.workers, self.stop_requested = retry_failed, [], False
        self.results = {}
        if self.path.exists():
            require(resume, "batch already exists; use --resume")
            self.state = read_json(self.path)
            require(self.state.get("schema") == BATCH_SCHEMA and self.state.get("identity") == identity,
                    "resume identity changed (spec/runtime/environment/checkpoints/stream/commit)")
            require(self.state.get("plan") == [asdict(job) for job in self.jobs], "resume job plan changed")
        else:
            require(not resume, "cannot resume an unknown batch")
            self.state = {"schema": BATCH_SCHEMA, "batch_id": batch_id, "created_at": utc_now(),
                          "identity": identity, "plan": [asdict(job) for job in self.jobs],
                          "jobs": {}, "events": [], "status": "initialized"}
            self.save()
        # Preserve compact reports for stages outside this invocation. Requested
        # stages are additionally revalidated through the manifest validator.
        for key, entry in self.state["jobs"].items():
            latest = entry.get("attempts", [{}])[-1]
            if latest.get("status") == "completed":
                try:
                    result = read_json(latest["result"])
                    verify_result_files(result)
                    require(result.get("batch_id") == batch_id
                            and result.get("identity_sha256") == content_digest(identity), "cached batch identity mismatch")
                    self.results[key] = result
                except (CampaignError, OSError, KeyError) as error:
                    latest.update({"status": "invalidated", "error": str(error)})
        self.prune_dependencies()

    def save(self):
        self.state["updated_at"] = utc_now()
        write_json(self.path, self.state)

    def prune_dependencies(self):
        """A failed Source invalidates its dependent cache without touching artifacts."""
        for key, result in list(self.results.items()):
            job = self.by_key[key]
            if job.method == "source":
                continue
            try:
                source = self.source_for(job)
                require((result.get("matched_source") or {}).get("manifest_sha256") == source["manifest_sha256"],
                        "result references a replaced Source attempt")
            except CampaignError as error:
                self.results.pop(key, None)
                latest = self.state["jobs"][key]["attempts"][-1]
                if latest.get("status") == "completed":
                    latest.update({"status": "invalidated", "error": str(error)})

    def source_for(self, job):
        if job.method == "source":
            return None
        stage = "smoke" if job.stage == "smoke" else "reval"
        matches = [candidate for candidate in self.jobs if candidate.stage == stage and candidate.method == "source"
                   and candidate.model == job.model and candidate.source_setting == job.source_setting]
        require(len(matches) == 1, "ambiguous matched Source")
        source = self.results.get(matches[0].key)
        require_source(source, job, self.identity, self.batch_id)
        return source

    def discover_manifest(self, run_tag):
        matches = []
        for path in (self.root / "avn/results/runs").glob("*-{}-*/manifest.json".format(run_tag)):
            if read_json(path).get("run_tag") == run_tag:
                matches.append(path)
        require(len(matches) == 1, "expected exactly one manifest for {}, found {}".format(run_tag, len(matches)))
        return matches[0]

    def certify(self, job, attempt, source):
        result = validate_run(self.discover_manifest(attempt["run_tag"]), job, attempt["run_tag"],
                              self.spec, self.identity, source, self.root)
        result["batch_id"] = self.batch_id
        result_path = Path(attempt["directory"]) / "result.json"
        write_json(result_path, result)
        attempt.update({"status": "completed", "result": str(result_path), "finished_at": utc_now()})
        self.results[job.key] = result
        return result

    def prepare(self, jobs):
        """Revalidate completed work and recover dead-launcher completion before retry."""
        pending = []
        failed = False
        for job in jobs:
            source = self.source_for(job)
            entry = self.state["jobs"].get(job.key)
            if entry is None or not entry.get("attempts"):
                pending.append(job)
                continue
            latest = entry["attempts"][-1]
            require(latest.get("run_tag") == attempt_tag(self.batch_id, job, latest.get("number")),
                    "attempt tag does not belong to this batch/job")
            expected_directory = self.log_directory / job.key / ("attempt-{}".format(latest["number"]))
            require(Path(latest["directory"]).resolve() == expected_directory.resolve(), "attempt directory does not belong to this batch")
            if latest["status"] == "running":
                require(latest.get("host") == socket.gethostname(), "cannot recover running work from another host")
                require(not process_alive(latest.get("pid"), latest.get("birth")),
                        "previous attempt is still alive: " + job.key)
                if isinstance(latest.get("pid"), int) and latest["pid"] > 0:
                    require(not process_group_alive(latest["pid"]), "previous attempt process group is still alive: " + job.key)
                try:
                    self.certify(job, latest, source)
                except (CampaignError, OSError, KeyError, ValueError, TypeError, AttributeError) as error:
                    latest.update({"status": "interrupted", "error": str(error)})
            if latest["status"] == "completed":
                try:
                    saved = read_json(latest["result"])
                    verify_result_files(saved)
                    expected_source = None if source is None else source["manifest_sha256"]
                    recorded_source = (saved.get("matched_source") or {}).get("manifest_sha256")
                    require(recorded_source == expected_source, "saved result references another Source attempt")
                    result = validate_run(saved["manifest"], job, latest["run_tag"], self.spec,
                                          self.identity, source, self.root)
                    result["batch_id"] = self.batch_id
                    self.results[job.key] = result
                    continue
                except (CampaignError, OSError, KeyError, ValueError, TypeError, AttributeError) as error:
                    latest.update({"status": "invalidated", "error": str(error)})
                    self.results.pop(job.key, None)
            if self.retry_failed:
                pending.append(job)
            else:
                failed = True
        self.prune_dependencies()
        self.save()
        return pending, failed

    def environment(self, job, run_tag):
        python = self.identity["environments"][job.model]["path"]
        environment = interpreter_environment(python)
        stream = self.identity["streams"][job.source_setting]["smoke" if job.stage == "smoke" else "full"]
        environment.update({"CUDA_VISIBLE_DEVICES": job.gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "TF_FORCE_GPU_ALLOW_GROWTH": "true", "NAVTTA_RUN_TAG": run_tag,
                            "NAVTTA_STREAM_ORDER_SHA256": stream["stream_order_sha256"],
                            "NAVTTA_STREAM_CONTENT_SHA256": stream["stream_content_sha256"]})
        environment.pop("NAVTTA_EVAL_SPLIT", None)
        return environment

    def launch(self, job):
        source = self.source_for(job)
        number, directory = next_attempt_dir(self.log_directory / job.key)
        run_tag = attempt_tag(self.batch_id, job, number)
        require(not list((self.root / "avn/results/runs").glob("*-{}-*/manifest.json".format(run_tag))),
                "new attempt tag already has a run manifest")
        command = job_command(self.spec, job, self.root)
        attempt = {"number": number, "directory": str(directory), "run_tag": run_tag,
                   "status": "running", "host": socket.gethostname(), "started_at": utc_now(),
                   "pid": None, "birth": None}
        entry = self.state["jobs"].setdefault(job.key, {"attempts": []})
        entry["attempts"].append(attempt)
        write_json(directory / "command.json", {"argv": command, "gpu": job.gpu,
                   "python": self.identity["environments"][job.model]["path"],
                   "run_tag": run_tag, "matched_source": None if source is None else source["manifest"]})
        output = (directory / "console.log").open("x", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=str(self.root), env=self.environment(job, run_tag),
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            output.close()
            attempt.update({"status": "failed", "error": str(error), "finished_at": utc_now()})
            self.save()
            return False
        worker = Worker(process, output, job, attempt, source)
        self.workers.append(worker)
        attempt.update({"pid": process.pid, "birth": process_birth(process.pid)})
        self.save()
        print("started {} attempt {} on GPU {}".format(job.key, number, job.gpu), flush=True)
        return True

    def finish(self, worker):
        worker.output.close()
        attempt = worker.attempt
        attempt["exit_code"] = worker.process.returncode
        try:
            require(worker.process.returncode == 0, "evaluation exited {}".format(worker.process.returncode))
            self.certify(worker.job, attempt, worker.source)
        except (CampaignError, OSError, KeyError, ValueError, TypeError, AttributeError) as error:
            attempt.update({"status": "failed", "error": str(error), "finished_at": utc_now()})
        print("{} {}{}".format(attempt["status"], worker.job.key,
                              ": " + attempt["error"] if "error" in attempt else ""), flush=True)
        self.workers.remove(worker)
        self.save()
        self.report()

    def request_stop(self, signum, frame):
        self.stop_requested = True

    def terminate_workers(self):
        # Only process groups created by this scheduler instance are signalled.
        # A dead shell leader does not imply its simulator children are dead.
        for worker in self.workers:
            try:
                os.killpg(worker.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while any(process_group_alive(worker.process.pid) for worker in self.workers) and time.monotonic() < deadline:
            time.sleep(0.1)
        for worker in list(self.workers):
            if process_group_alive(worker.process.pid):
                try:
                    os.killpg(worker.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            worker.process.wait()
            worker.output.close()
            worker.attempt.update({"status": "interrupted", "finished_at": utc_now(),
                                   "exit_code": worker.process.returncode, "error": "scheduler interrupted"})
            self.workers.remove(worker)
        self.save()

    def check_runtime(self):
        require(capture(["git", "rev-parse", "HEAD"], cwd=self.root) == self.identity["git_commit"],
                "repository commit changed during batch")
        require(sha256_file(self.identity["spec_path"]) == self.identity["spec_sha256"], "spec changed during batch")
        current = {str(path.relative_to(self.root)): sha256_file(path) for path in runtime_files(self.spec, self.root)}
        require(current == self.identity["runtime_files"], "runtime files changed during batch")

    def run_group(self, jobs, label, poll_seconds=1):
        self.check_runtime()
        pending, prior_failures = self.prepare(jobs)
        self.state["events"].append({"event": "barrier_started", "label": label, "at": utc_now(), "jobs": len(jobs)})
        self.save()
        while pending or self.workers:
            if self.stop_requested:
                self.terminate_workers()
                return False
            for job in list(pending):
                if self.stop_requested:
                    break
                if sum(worker.job.gpu == job.gpu for worker in self.workers) < 6:
                    pending.remove(job)
                    self.launch(job)
            for worker in list(self.workers):
                if worker.process.poll() is not None:
                    self.finish(worker)
            if pending or self.workers:
                time.sleep(poll_seconds)
        complete = not prior_failures and all(job.key in self.results for job in jobs)
        self.state["events"].append({"event": "barrier_finished", "label": label, "at": utc_now(), "success": complete})
        self.save()
        self.report()
        return complete

    def run(self, stage, poll_seconds=1):
        previous = {sig: signal.signal(sig, self.request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.state["status"] = "running"
        try:
            stages = ("smoke", "reval", "search") if stage == "all" else (stage,)
            if stage == "search":
                controls = [job for job in self.jobs if job.stage == "reval" and job.method == "source"]
                pending, failed = self.prepare(controls)
                require(not pending and not failed, "search requires four completed certified Source controls in this batch; run --stage reval")
            for current in stages:
                stage_jobs = [job for job in self.jobs if job.stage == current]
                groups = ([("{}-source".format(current), [job for job in stage_jobs if job.method == "source"]),
                           (current, [job for job in stage_jobs if job.method != "source"])]
                          if current != "search" else [
                              ("search-round-{}".format(index), [job for job in stage_jobs if job.round == index])
                              for index in range(1, 5)])
                for label, jobs in groups:
                    if jobs and not self.run_group(jobs, label, poll_seconds):
                        self.state["status"] = "interrupted" if self.stop_requested else "failed"
                        self.save()
                        return 130 if self.stop_requested else 1
            self.state["status"] = "completed-" + stage
            self.save()
            self.report()
            return 0
        except BaseException:
            self.terminate_workers()
            self.state["status"] = "failed"
            self.save()
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def report(self):
        self.prune_dependencies()
        columns = ["stage", "round", "model", "method", "source_setting", "candidate_id", "gpu", "status", "attempt",
                   "episodes", "spl", "success", "softspl", "delta_spl", "delta_success", "delta_softspl",
                   "relative_param_drift", "query_rate", "manifest", "error"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for job in self.jobs:
            entry = self.state["jobs"].get(job.key, {})
            latest = entry.get("attempts", [{}])[-1]
            row = {key: getattr(job, key) for key in columns if hasattr(job, key)}
            row.update({"status": latest.get("status", "pending"), "attempt": latest.get("number", ""), "error": latest.get("error", "")})
            result = self.results.get(job.key)
            if result:
                row.update({key: result["metrics"][key] for key in ("spl", "success", "softspl")})
                row.update({key: result["diagnostic_summary"].get(key, "") for key in ("relative_param_drift", "query_rate")})
                row["manifest"] = result["manifest"]
                if job.method != "source":
                    source = self.source_for(job)
                    for metric in ("spl", "success", "softspl"):
                        row["delta_" + metric] = result["metrics"][metric] - source["metrics"][metric]
            writer.writerow(row)
        atomic_write(self.directory / "metrics.csv", output.getvalue())
        selections = []
        for model in MODELS:
            for method in SEARCH_METHODS:
                for setting in SETTINGS:
                    candidates = [result for result in self.results.values() if result["job"]["stage"] == "search"
                                  and (result["job"]["model"], result["job"]["method"], result["job"]["source_setting"])
                                  == (model, method, setting)]
                    ranked = rank_results(candidates)
                    selections.append({"model": model, "method": method, "source_setting": setting,
                                       "certified_candidates": len(ranked), "complete": len(ranked) == 12,
                                       "winner": ranked[0] if ranked else None})
        write_json(self.directory / "selected_configs.json", {"schema": "navtta.avn.reval_search.selection.v1",
                   "batch_id": self.batch_id, "ranking": ["spl_desc", "success_desc", "softspl_desc", "drift_asc", "id_asc"],
                   "partial_rankings_are_provisional": True, "cells": selections})
        counts = {}
        for job in self.jobs:
            state = self.state["jobs"].get(job.key, {}).get("attempts", [{}])[-1].get("status", "pending")
            counts[state] = counts.get(state, 0) + 1
        lines = ["AVN revaluation/search batch `{}`".format(self.batch_id), "", "Status: " + self.state["status"], "",
                 "Job counts: " + canonical(counts), "",
                 "Ranking uses SPL, SR, SoftSPL, lower drift, then candidate ID. Negative Source deltas are retained.",
                 "ATENA query rate is reported without using it as a score or excluding valid trials.", "",
                 "| Revaluation model | Method | Source | Status | SPL | SR | SoftSPL |",
                 "|---|---|---|---|---:|---:|---:|"]
        for job in self.jobs:
            if job.stage != "reval":
                continue
            result = self.results.get(job.key)
            status = self.state["jobs"].get(job.key, {}).get("attempts", [{}])[-1].get("status", "pending")
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                job.model, job.method, job.source_setting, status,
                result["metrics"]["spl"] if result else "—", result["metrics"]["success"] if result else "—",
                result["metrics"]["softspl"] if result else "—"))
        lines.extend(["",
                 "| Model | Method | Source | Certified / 12 | Best candidate | SPL | SR | Query rate |",
                 "|---|---|---|---:|---|---:|---:|---:|"])
        for cell in selections:
            winner = cell["winner"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cell["model"], cell["method"], cell["source_setting"], cell["certified_candidates"],
                winner["job"]["candidate_id"] if winner else "—", winner["metrics"]["spl"] if winner else "—",
                winner["metrics"]["success"] if winner else "—", winner["diagnostic_summary"].get("query_rate", "—") if winner else "—"))
        atomic_write(self.directory / "summary.md", "\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--stage", choices=("all", "reval", "search", "smoke"), default="all")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--smt-python", default=sys.executable)
    parser.add_argument("--enmus-python", default=sys.executable)
    for flag in ("dry-run", "preflight-only", "resume", "retry-failed", "status"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    require(args.poll_seconds > 0 and args.poll_seconds <= 60, "poll interval must be in (0,60]")
    require(not args.retry_failed or args.resume, "--retry-failed requires --resume")
    require(not (args.resume or args.status) or args.batch_id, "--resume/--status requires --batch-id")
    if not args.batch_id:
        args.batch_id = "avn-reval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    require(SAFE_ID.fullmatch(args.batch_id), "unsafe batch ID")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.status:
            state = read_json(ROOT / "avn/results/analysis/reval_search" / args.batch_id / "batch.json")
            counts = {}
            for entry in state["jobs"].values():
                value = entry["attempts"][-1]["status"]
                counts[value] = counts.get(value, 0) + 1
            print(canonical({"batch_id": args.batch_id, "status": state["status"], "jobs": counts,
                             "updated_at": state["updated_at"]}))
            return 0
        spec = load_spec(args.spec)
        jobs = build_jobs(spec, args.stage)
        if args.dry_run:
            print(json.dumps({"batch_id": args.batch_id, "stage": args.stage, "job_count": len(jobs),
                              "plan": [{**asdict(job), "command": job_command(spec, job),
                                        "python": args.smt_python if job.model == "smt_audio" else args.enmus_python}
                                       for job in jobs]}, indent=2))
            return 0
        identity = preflight(spec, args.spec, {"smt_audio": args.smt_python, "enmus": args.enmus_python})
        if args.preflight_only:
            print(canonical({"preflight": "passed", "identity_sha256": content_digest(identity), "job_count": len(jobs)}))
            return 0
        directory = ROOT / "avn/results/analysis/reval_search" / args.batch_id
        with BatchLock(directory / ".lock"):
            campaign = Campaign(spec, identity, args.batch_id, resume=args.resume, retry_failed=args.retry_failed)
            return campaign.run(args.stage, args.poll_seconds)
    except (CampaignError, OSError, KeyError, ValueError) as error:
        print("AVN campaign error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
