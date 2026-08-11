import hashlib
import json
import os
import tempfile
import unittest

from navtta_core.experiment.episode_order import (
    SEEDED_ORDER_ALGORITHM,
    SEEDED_ORDER_DOMAIN_SEPARATOR,
    SEEDED_ORDER_POLICY,
    build_episode_order_manifest,
    canonical_episode_records,
    canonicalize_eval_splits,
    configure_exact_episode_env,
    derive_seeded_episode_order_manifest,
    iter_exact_batches,
    load_episode_order_manifest,
    prefix_episode_order_manifest,
    reorder_episodes,
    resolve_episode_order_manifest_path,
    run_exact_agent_epoch,
    select_allowed_episodes_in_order,
    sha256_file,
    validate_episode_order_manifest,
    verify_manifest_dataset,
)


class EpisodeOrderTest(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            {"instr_id": "10_0", "scan": "scene-b"},
            {"instr_id": "2_0", "scan": "scene-b"},
            {
                "instr_id": "3_0",
                "scene_id": "data/scene-a/scene-a.glb",
            },
        ]
        self.dataset_digest = "a" * 64

    def _manifest(self):
        return build_episode_order_manifest(
            self.episodes,
            benchmark="synthetic-r2r",
            split="val_seen",
            dataset_path="vln/data/synthetic.json",
            dataset_sha256=self.dataset_digest,
            source_id_field="instr_id",
        )

    def test_canonical_order_is_scene_then_natural_episode_id(self):
        records = canonical_episode_records(self.episodes)
        self.assertEqual(
            records,
            [
                {"episode_id": "3_0", "scene_id": "scene-a"},
                {"episode_id": "2_0", "scene_id": "scene-b"},
                {"episode_id": "10_0", "scene_id": "scene-b"},
            ],
        )

    def test_manifest_round_trip_and_exact_reorder(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "order.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            loaded = load_episode_order_manifest(path, expected_split="val_seen")
        ordered = reorder_episodes(self.episodes, loaded)
        self.assertEqual([item["instr_id"] for item in ordered], ["3_0", "2_0", "10_0"])

    def test_seeded_manifest_is_domain_separated_and_reproducible(self):
        parent = self._manifest()
        parent_file_sha = hashlib.sha256(b"parent manifest bytes").hexdigest()
        first = derive_seeded_episode_order_manifest(
            parent,
            order_seed=1,
            parent_manifest_path="vln/manifests/episode_order/example/val_seen.json",
            parent_manifest_sha256=parent_file_sha,
        )
        repeated = derive_seeded_episode_order_manifest(
            parent,
            order_seed=1,
            parent_manifest_path="vln/manifests/episode_order/example/val_seen.json",
            parent_manifest_sha256=parent_file_sha,
        )
        second = derive_seeded_episode_order_manifest(
            parent,
            order_seed=2,
            parent_manifest_path="vln/manifests/episode_order/example/val_seen.json",
            parent_manifest_sha256=parent_file_sha,
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first["order_sha256"], second["order_sha256"])
        self.assertEqual(first["order_policy"], SEEDED_ORDER_POLICY)
        self.assertEqual(first["derivation"], {
            "algorithm": SEEDED_ORDER_ALGORITHM,
            "domain_separator": SEEDED_ORDER_DOMAIN_SEPARATOR,
            "order_seed": 1,
            "parent_manifest_path": (
                "vln/manifests/episode_order/example/val_seen.json"
            ),
            "parent_manifest_sha256": parent_file_sha,
            "parent_order_sha256": parent["order_sha256"],
        })

    def test_seeded_manifest_rejects_algorithm_and_order_tampering(self):
        derived = derive_seeded_episode_order_manifest(
            self._manifest(),
            order_seed=1,
            parent_manifest_path="parent.json",
            parent_manifest_sha256="b" * 64,
        )
        derived["derivation"]["algorithm"] = "random.shuffle"
        with self.assertRaisesRegex(ValueError, "algorithm"):
            validate_episode_order_manifest(derived)

        derived["derivation"]["algorithm"] = SEEDED_ORDER_ALGORITHM
        derived["derivation"]["order_seed"] = True
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_episode_order_manifest(derived)

        derived["derivation"]["order_seed"] = 1
        derived["episodes"] = list(reversed(derived["episodes"]))
        with self.assertRaisesRegex(ValueError, "declared order policy"):
            validate_episode_order_manifest(derived)

    def test_seeded_manifest_requires_an_exact_integer_seed(self):
        for seed in (True, 1.0, "1"):
            with self.subTest(seed=seed), self.assertRaisesRegex(
                    ValueError, "must be an integer"):
                derive_seeded_episode_order_manifest(
                    self._manifest(),
                    order_seed=seed,
                    parent_manifest_path="parent.json",
                    parent_manifest_sha256="b" * 64,
                )

    def test_manifest_checks_expected_benchmark(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "order.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(ValueError, "expected benchmark"):
                load_episode_order_manifest(
                    path,
                    expected_split="val_seen",
                    expected_benchmark="different-benchmark",
                )

    def test_manifest_rejects_order_that_disagrees_with_policy(self):
        manifest = self._manifest()
        manifest["episodes"] = list(reversed(manifest["episodes"]))
        payload = json.dumps(
            manifest["episodes"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        manifest["order_sha256"] = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "order.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(ValueError, "declared order policy"):
                load_episode_order_manifest(path, expected_split="val_seen")

    def test_reorder_rejects_missing_or_extra_episode(self):
        with self.assertRaisesRegex(ValueError, "manifest/dataset mismatch"):
            reorder_episodes(self.episodes[:-1], self._manifest())

    def test_manifest_rejects_duplicate_scene_and_episode_key(self):
        episodes = self.episodes + [dict(self.episodes[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate canonical episode"):
            build_episode_order_manifest(
                episodes,
                benchmark="synthetic-r2r",
                split="val_seen",
                dataset_path="vln/data/synthetic.json",
                dataset_sha256=self.dataset_digest,
                source_id_field="instr_id",
            )

    def test_exact_batches_never_wrap_pad(self):
        batches = list(iter_exact_batches(list(range(5)), batch_size=2))
        self.assertEqual(batches, [[0, 1], [2, 3], [4]])
        self.assertEqual([item for batch in batches for item in batch], list(range(5)))

    def test_split_order_is_fixed_and_validated(self):
        self.assertEqual(
            canonicalize_eval_splits(["test", "val_seen", "val_unseen"]),
            ["val_seen", "val_unseen", "test"],
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            canonicalize_eval_splits(["val_seen", "val_seen"])

    def test_dataset_digest_is_checked(self):
        with tempfile.NamedTemporaryFile() as stream:
            stream.write(b"dataset bytes")
            stream.flush()
            manifest = self._manifest()
            manifest["dataset"]["sha256"] = sha256_file(stream.name)
            verify_manifest_dataset(manifest, stream.name)
            with open(stream.name, "ab") as changed:
                changed.write(b" changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_manifest_dataset(manifest, stream.name)

    def test_manifest_path_accepts_directory_or_split_template(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                resolve_episode_order_manifest_path(directory, "val_seen"),
                os.path.join(directory, "val_seen.json"),
            )
        self.assertEqual(
            resolve_episode_order_manifest_path(
                "/orders/{split}.json", "val_unseen"
            ),
            "/orders/val_unseen.json",
        )

    def test_exact_agent_epoch_consumes_no_wrap_duplicate(self):
        class FakeEnv(object):
            def __init__(self, data):
                self.data = data
                self.batch_size = 1
                self.ix = 0
                self.batch = []

            def reset_epoch(self, shuffle=False):
                self.ix = 0

            def next(self):
                self.batch = self.data[self.ix : self.ix + 1]
                self.ix += 1
                return self.batch[0]

        class FakeAgent(object):
            def __init__(self, env):
                self.env = env
                self.results = {}
                self.tta_adapter = type(
                    "Adapter",
                    (),
                    {
                        "reset": lambda adapter: setattr(
                            adapter, "resets", getattr(adapter, "resets", 0) + 1
                        )
                    },
                )()

            def rollout(self):
                item = self.env.next()
                return [{"instr_id": item["instr_id"], "path": []}]

        env = FakeEnv(list(reversed(self.episodes)))
        configure_exact_episode_env(env, self._manifest())
        agent = FakeAgent(env)
        self.assertTrue(run_exact_agent_epoch(agent, agent.rollout))
        self.assertEqual(list(agent.results), ["3_0", "2_0", "10_0"])
        self.assertEqual(env.ix, 3)
        self.assertEqual(agent.tta_adapter.resets, 1)

    def test_exact_env_rejects_batching_that_would_hide_global_order(self):
        class FakeEnv(object):
            batch_size = 2
            data = []

            def reset_epoch(self, shuffle=False):
                pass

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            configure_exact_episode_env(FakeEnv(), self._manifest())

    def test_smoke_prefix_is_self_consistent_and_limits_environment(self):
        manifest = prefix_episode_order_manifest(self._manifest(), 1)
        self.assertEqual(manifest["episode_count"], 1)
        self.assertEqual(manifest["episodes"], self._manifest()["episodes"][:1])

        class FakeEnv(object):
            batch_size = 1
            data = list(reversed(self.episodes))
            ix = 0

            def reset_epoch(self, shuffle=False):
                self.ix = 0

        env = FakeEnv()
        old_value = os.environ.get("NAVTTA_SMOKE_EPISODES")
        os.environ["NAVTTA_SMOKE_EPISODES"] = "1"
        try:
            configure_exact_episode_env(env, self._manifest())
        finally:
            if old_value is None:
                os.environ.pop("NAVTTA_SMOKE_EPISODES", None)
            else:
                os.environ["NAVTTA_SMOKE_EPISODES"] = old_value
        self.assertEqual(len(env.data), 1)
        self.assertEqual(env._navtta_episode_order, self._manifest()["episodes"][:1])

    def test_allowed_episode_filter_honors_ids_across_type_and_scene_partition(self):
        episodes = [
            {"episode_id": 3, "scene_id": "a.glb"},
            {"episode_id": 1, "scene_id": "a.glb"},
        ]
        ordered = select_allowed_episodes_in_order(episodes, ["1", "2", "3"])
        self.assertEqual([item["episode_id"] for item in ordered], [1, 3])


if __name__ == "__main__":
    unittest.main()
