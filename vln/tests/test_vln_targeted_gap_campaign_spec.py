import hashlib
import json
from pathlib import Path
import re
import unittest
from xml.etree import ElementTree
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v1.json"
PLAN_PATH = REPO_ROOT / "vln/experiments/VLN_TARGETED_GAP_CAMPAIGN_V1_PLAN.md"
WORKBOOK_PATH = REPO_ROOT / "docs/NavTTA_benchmark_results.xlsx"

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xlsx_rows(path):
    """Read the few scalar XLSX fields needed for the gap audit with stdlib."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iterfind(".//{%s}t" % _MAIN_NS))
                for item in root.findall("{%s}si" % _MAIN_NS)
            ]

        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("{%s}Relationship" % _PACKAGE_REL_NS)
        }
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        result = {}
        for sheet in workbook.findall(".//{%s}sheet" % _MAIN_NS):
            name = sheet.attrib["name"]
            if name not in {"REVERIE", "R2R", "R2R-CE"}:
                continue
            target = targets[sheet.attrib["{%s}id" % _REL_NS]].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheet_xml = ElementTree.fromstring(archive.read(target))
            rows = {}
            for cell in sheet_xml.findall(".//{%s}c" % _MAIN_NS):
                match = re.fullmatch(r"([A-Z]+)([0-9]+)", cell.attrib["r"])
                column, row_number = match.group(1), int(match.group(2))
                value_node = cell.find("{%s}v" % _MAIN_NS)
                value = None if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iterfind(".//{%s}t" % _MAIN_NS)
                    )
                rows.setdefault(row_number, {})[column] = value
            result[name] = rows
    return result


def _workbook_gap_inventory(path):
    sheets = _xlsx_rows(path)
    metric_columns = {
        "REVERIE": tuple("BCDEFGHIJKLM"),
        "R2R": tuple("BCDEFGHI"),
        "R2R-CE": tuple("BCDEFGHIJK"),
    }
    benchmark_ids = {"REVERIE": "reverie", "R2R": "r2r", "R2R-CE": "r2r-ce"}
    counts = {
        "candidate_method_rows_before_exclusions": 0,
        "excluded_streamvln_method_rows": 0,
        "excluded_ours_rows_after_streamvln": 0,
        "excluded_populated_prior_result_rows": 0,
        "eligible_blank_cells": 0,
    }
    eligible = []
    for sheet_name in ("REVERIE", "R2R", "R2R-CE"):
        model = None
        for row_number, row in sorted(sheets[sheet_name].items()):
            label = row.get("A")
            if row_number < 3 or not isinstance(label, str):
                continue
            if not label.startswith("+"):
                model = label.split("（", 1)[0].split("(", 1)[0].strip().lower()
                continue
            counts["candidate_method_rows_before_exclusions"] += 1
            method = label[1:].strip().lower()
            if model == "streamvln":
                counts["excluded_streamvln_method_rows"] += 1
                continue
            if method == "ours":
                counts["excluded_ours_rows_after_streamvln"] += 1
                continue
            if any(row.get(column) not in (None, "") for column in metric_columns[sheet_name]):
                counts["excluded_populated_prior_result_rows"] += 1
                continue
            counts["eligible_blank_cells"] += 1
            eligible.append(
                {
                    "sheet": sheet_name,
                    "row": row_number,
                    "setting": "{}-{}".format(model, benchmark_ids[sheet_name]),
                    "method": method,
                }
            )
    return counts, eligible


class TargetedGapCampaignSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_plan_or_successor_pins_v1_digest(self):
        digest = _sha256(SPEC_PATH)
        if self.spec["status"] == "active":
            self.assertIn(
                "SHA256 `{}`".format(digest),
                PLAN_PATH.read_text(encoding="utf-8"),
            )
        else:
            successor_path = REPO_ROOT / (
                "vln/experiments/{}.json".format(self.spec["superseded_by"])
            )
            successor = json.loads(successor_path.read_text(encoding="utf-8"))
            self.assertEqual(successor["supersedes"][0]["sha256"], digest)

    def test_lifecycle_and_exact_cell_matrix(self):
        spec = self.spec
        self.assertEqual(spec["schema"], "navtta.vln_targeted_gap_campaign.v1")
        self.assertEqual(spec["status"], "active")
        self.assertEqual(spec["supersedes"], [])
        self.assertIsNone(spec["superseded_by"])
        self.assertEqual(spec["scope"]["cell_count"], 16)
        self.assertEqual(spec["scope"]["hidden_test_transfer_jobs"], 5)
        self.assertNotIn("vln", spec["scope"]["excluded"])

        expected = [
            ("hamt-reverie", "eam"),
            ("duet-reverie", "eam"),
            ("goat-reverie", "eam"),
            ("goat-reverie", "feedtta"),
            ("goat-reverie", "idea"),
            ("hamt-r2r", "tent"),
            ("hamt-r2r", "fstta"),
            ("hamt-r2r", "feedtta"),
            ("hamt-r2r", "atena"),
            ("hamt-r2r", "idea"),
            ("goat-r2r", "tent"),
            ("goat-r2r", "eam"),
            ("goat-r2r", "feedtta"),
            ("goat-r2r", "idea"),
            ("etpnav-r2r-ce", "eam"),
            ("bevbert-r2r-ce", "eam"),
        ]
        cells = spec["cells"]
        self.assertEqual([(c["setting"], c["method"]) for c in cells], expected)
        self.assertEqual([c["queue_id"] for c in cells], list(range(16)))
        self.assertEqual(len({c["cell_id"] for c in cells}), 16)

    def test_exact_cell_matrix_is_derived_from_pinned_workbook(self):
        inventory = self.spec["gap_inventory"]
        workbook = inventory["workbook"]
        self.assertEqual(workbook["path"], "docs/NavTTA_benchmark_results.xlsx")
        self.assertTrue(WORKBOOK_PATH.is_file())
        self.assertEqual(_sha256(WORKBOOK_PATH), workbook["sha256"])
        self.assertEqual(
            inventory["metric_columns"],
            {"REVERIE": "B:M", "R2R": "B:I", "R2R-CE": "B:K"},
        )

        counts, eligible = _workbook_gap_inventory(WORKBOOK_PATH)
        self.assertEqual(counts, inventory["counts"])
        self.assertEqual(eligible, inventory["eligible_cells"])
        self.assertEqual(counts["eligible_blank_cells"], 16)
        self.assertEqual(
            [(item["setting"], item["method"]) for item in eligible],
            [(cell["setting"], cell["method"]) for cell in self.spec["cells"]],
        )

    def test_exact_four_queues_per_gpu(self):
        schedule = self.spec["schedule"]
        self.assertEqual(schedule["gpu_ids"], [0, 1, 2, 3])
        self.assertEqual(schedule["queues_per_gpu"], 4)
        self.assertEqual(schedule["max_active_jobs_per_cell"], 1)
        self.assertFalse(schedule["work_stealing"])

        cells = self.spec["cells"]
        observed = {str(gpu): [] for gpu in range(4)}
        for cell in cells:
            self.assertEqual(cell["gpu_slot"], cell["queue_id"] % 4)
            observed[str(cell["gpu_slot"])].append(cell["queue_id"])
        self.assertEqual(observed, schedule["gpu_queues"])
        self.assertTrue(all(len(queue_ids) == 4 for queue_ids in observed.values()))

        grids = self.spec["candidate_grids"]
        jobs = {
            str(gpu): sum(
                grids[cell["grid"]]["candidate_count"]
                for cell in cells
                if cell["gpu_slot"] == gpu
            )
            for gpu in range(4)
        }
        self.assertEqual(jobs, schedule["development_jobs_by_gpu"])
        self.assertEqual(jobs, {"0": 15, "1": 15, "2": 13, "3": 12})

    def test_grids_are_explicit_provenanced_and_capped(self):
        spec = self.spec
        grids = spec["candidate_grids"]
        used_grids = {cell["grid"] for cell in spec["cells"]}
        self.assertEqual(used_grids, set(grids))

        for name, grid in grids.items():
            candidates = grid["candidates"]
            self.assertEqual(grid["candidate_count"], len(candidates), name)
            self.assertGreaterEqual(len(candidates), 1, name)
            self.assertLessEqual(len(candidates), 10, name)
            self.assertEqual(
                len({candidate["id"] for candidate in candidates}),
                len(candidates),
                name,
            )
            self.assertTrue(
                all(candidate.get("parameters") for candidate in candidates), name
            )
            self.assertTrue(
                all(candidate.get("provenance") for candidate in candidates), name
            )
            source = grid["source_spec"]
            path = REPO_ROOT / source["path"]
            self.assertTrue(path.is_file(), source["path"])
            self.assertEqual(_sha256(path), source["sha256"], source["path"])
            source_spec = json.loads(path.read_text(encoding="utf-8"))
            source_method = source_spec["methods"][grid["method"]]
            source_candidates = {
                candidate["id"]: candidate
                for candidate in source_method["candidates"]
            }
            self.assertEqual(
                grid.get("required_diagnostics"),
                source_method.get("required_diagnostics"),
                name,
            )
            for key, value in source_method["base"].items():
                self.assertEqual(grid["base"].get(key), value, (name, key))
            for candidate in candidates:
                self.assertEqual(
                    candidate["provenance"],
                    "{}:{}:{}".format(
                        path.stem, grid["method"], candidate["id"]
                    ),
                )
                self.assertEqual(
                    candidate["parameters"],
                    source_candidates[candidate["id"]]["parameters"],
                    (name, candidate["id"]),
                )
            self.assertEqual(
                sum(candidate.get("role") == "paper_anchor" for candidate in candidates),
                1,
                name,
            )

        job_count = sum(
            grids[cell["grid"]]["candidate_count"] for cell in spec["cells"]
        )
        self.assertEqual(job_count, 55)
        self.assertEqual(spec["scope"]["development_candidate_jobs"], 55)
        self.assertEqual(spec["selection"]["candidate_count"], 55)
        self.assertEqual(
            max(grid["candidate_count"] for grid in grids.values()), 5
        )

        learning_rate_primary_grids = 0
        for grid in grids.values():
            candidates = grid["candidates"]
            parameter_keys = set().union(
                *(candidate["parameters"].keys() for candidate in candidates)
            )
            varying = {
                key
                for key in parameter_keys
                if len(
                    {
                        candidate["parameters"].get(key)
                        for candidate in candidates
                    }
                )
                > 1
            }
            if varying and all(key.startswith("lr") for key in varying):
                learning_rate_primary_grids += 1
        self.assertGreater(learning_rate_primary_grids, len(grids) / 2)

        atena = grids["r2r_atena_capped_v1"]
        self.assertEqual(atena["candidate_count"], 5)
        self.assertEqual(atena["predeclared_exclusion"]["candidate_id"], "delta_0p2")
        self.assertNotIn(
            "delta_0p2", {candidate["id"] for candidate in atena["candidates"]}
        )

    def test_split_roles_seeds_horizons_and_freeze_barriers(self):
        protocol = self.spec["protocol"]
        phases = protocol["phase_order"]
        self.assertEqual(
            [phase["phase"] for phase in phases],
            [
                "development_search",
                "campaign_freeze",
                "frozen_evaluation",
                "optional_hidden_test_transfer",
            ],
        )
        self.assertEqual(phases[0]["split"], "val_unseen")
        self.assertEqual(phases[0]["split_role"], "development")
        self.assertEqual(phases[2]["split"], "val_seen")
        self.assertEqual(phases[2]["split_role"], "evaluation")
        for index in (0, 2, 3):
            self.assertEqual(phases[index]["episode_order_seed"], 0)
            self.assertEqual(phases[index]["model_seed"], 0)
        self.assertEqual(protocol["randomness"]["model_seeds"], [0])
        self.assertEqual(protocol["randomness"]["episode_order_seeds"], [0])
        self.assertEqual(
            protocol["randomness"]["action_rng"],
            "not_consumed_by_the_frozen_argmax_action_protocol",
        )
        self.assertEqual(
            protocol["randomness"]["augmentation_rng"],
            "evaluation_augmentation_is_disabled",
        )
        self.assertIn(
            "does not claim independent hash-derived RNG streams",
            protocol["randomness"]["note"],
        )
        self.assertFalse(protocol["state"]["state_transfer_between_splits"])
        self.assertIn(
            "all_16_winners_frozen_before_any_val_seen_job",
            protocol["global_barriers"],
        )

        horizons = protocol["horizons"]
        self.assertEqual(horizons["reverie"]["development"]["value"], 3521)
        self.assertEqual(horizons["reverie"]["evaluation"]["value"], 1423)
        self.assertEqual(horizons["r2r"]["development"]["value"], 2349)
        self.assertEqual(horizons["r2r"]["evaluation"]["value"], 1021)
        self.assertEqual(horizons["r2r-ce"]["development"]["value"], 1839)
        self.assertEqual(horizons["r2r-ce"]["evaluation"]["value"], 778)

        freeze = self.spec["freeze"]
        self.assertTrue(freeze["before_evaluation"])
        self.assertIn("gap_inventory", freeze["immutable_fields"])
        self.assertIn("all_16_winner_config_sha256_values", freeze["required_bindings"])
        self.assertIn("candidate_addition", freeze["forbidden_overrides"])
        self.assertTrue(
            self.spec["leakage_controls"]["posthoc_grid_expansion_forbidden"]
        )
        self.assertIn(
            "val_seen_metrics",
            self.spec["leakage_controls"]["forbidden_selection_inputs"],
        )

        rankings = self.spec["selection"]["ranking_by_benchmark"]
        self.assertEqual(
            rankings["reverie"],
            [
                "highest_RGSPL",
                "highest_RGS",
                "highest_SPL",
                "highest_SR",
                "lower_parameter_drift",
                "fewer_accepted_updates",
                "lexicographically_smallest_candidate_id",
            ],
        )
        expected_navigation_ranking = [
            "highest_SPL",
            "highest_SR",
            "lower_parameter_drift",
            "fewer_accepted_updates",
            "lexicographically_smallest_candidate_id",
        ]
        self.assertEqual(rankings["r2r"], expected_navigation_ranking)
        self.assertEqual(rankings["r2r-ce"], expected_navigation_ranking)

    def test_stream_manifests_and_native_ce_version_are_pinned(self):
        streams = self.spec["data_bindings"]["streams"]
        for name, binding in streams.items():
            path = REPO_ROOT / binding["path"]
            self.assertTrue(path.is_file(), name)
            self.assertEqual(_sha256(path), binding["sha256"], name)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["episode_count"], binding["episodes"], name)
            self.assertEqual(document["order_sha256"], binding["order_sha256"], name)
            self.assertEqual(document["dataset"]["sha256"], binding["dataset_sha256"], name)

        ce_cells = [cell for cell in self.spec["cells"] if cell["benchmark"] == "r2r-ce"]
        self.assertEqual(len(ce_cells), 2)
        self.assertTrue(all(cell["data_version"] == "v1.2-native" for cell in ce_cells))
        self.assertEqual(
            self.spec["source_control"]["bindings"]["r2r_ce_v1_2_val_unseen"]["mode"],
            "rerun_required",
        )
        self.assertEqual(
            self.spec["source_control"]["bindings"]["r2r_ce_v1_2_val_seen"]["mode"],
            "rerun_required",
        )
        self.assertEqual(
            set(self.spec["source_control"]["forbidden_reuse"]),
            {
                "vln/manifests/r2r_ce_reused_source_controls.json",
                "vln/manifests/r2r_ce_val_unseen_reused_source_controls.json",
            },
        )
        promotion = self.spec["source_control"]["native_v1_2_promotion_gate"]
        self.assertEqual(
            promotion["status"], "blocked_pending_authenticated_source_ledgers"
        )
        self.assertEqual(
            set(promotion["required_setting_split_pairs"]),
            {
                "etpnav-r2r-ce:val_unseen",
                "bevbert-r2r-ce:val_unseen",
                "etpnav-r2r-ce:val_seen",
                "bevbert-r2r-ce:val_seen",
            },
        )
        self.assertIn("successor spec", promotion["promotion_action"])
        self.assertEqual(
            promotion["candidate_ledger_schema"],
            "navtta.vln_r2r_ce_v1_2_source_controls_candidate.v1",
        )
        self.assertEqual(
            promotion["suggested_tracked_bindings"],
            {
                "r2r_ce_v1_2_val_unseen": (
                    "vln/manifests/r2r_ce_v1_2_val_unseen_source_controls.json"
                ),
                "r2r_ce_v1_2_val_seen": (
                    "vln/manifests/r2r_ce_v1_2_val_seen_source_controls.json"
                ),
            },
        )
        self.assertEqual(self.spec["launch_readiness"]["status"], "blocked")

    def test_reused_source_bindings_are_content_addressed(self):
        bindings = self.spec["source_control"]["bindings"]
        for name in (
            "reverie_val_unseen",
            "reverie_val_seen",
            "r2r_val_unseen",
            "r2r_val_seen",
        ):
            binding = bindings[name]
            self.assertEqual(binding["mode"], "reuse")
            path = REPO_ROOT / binding["path"]
            self.assertTrue(path.is_file(), name)
            self.assertEqual(_sha256(path), binding["sha256"], name)

    def test_idea_source_statistics_bindings_keep_reviewed_provenance(self):
        bindings = self.spec["data_bindings"]["idea_source_statistics"]
        self.assertEqual(set(bindings), {"goat-reverie", "hamt-r2r", "goat-r2r"})
        sources = {
            "goat-reverie": "vln/experiments/reverie_consistency_search_v2.json",
            "hamt-r2r": "vln/experiments/r2r_consistency_search_v2.json",
            "goat-r2r": "vln/experiments/r2r_consistency_search_v2.json",
        }
        for setting, source_path in sources.items():
            source = json.loads((REPO_ROOT / source_path).read_text(encoding="utf-8"))
            reviewed = source["methods"]["idea"]["source_statistics"][setting]
            binding = bindings[setting]
            for field in ("path", "sha256", "trajectory_count"):
                self.assertEqual(binding[field], reviewed[field], (setting, field))
            self.assertEqual(
                reviewed["checkpoint_sha256"],
                self.spec["data_bindings"]["checkpoints"][setting],
            )
            local_path = REPO_ROOT / binding["path"]
            if local_path.is_file():
                self.assertEqual(_sha256(local_path), binding["sha256"], setting)
        self.assertTrue(
            any(
                "IDEA source-training-statistics" in blocker
                for blocker in self.spec["launch_readiness"]["blockers"]
            )
        )

    def test_supervision_is_per_cell_and_idea_is_not_llm(self):
        expected_feedback = {
            "goat-reverie-feedtta",
            "hamt-r2r-feedtta",
            "hamt-r2r-atena",
            "goat-r2r-feedtta",
        }
        observed_feedback = {
            cell["cell_id"]
            for cell in self.spec["cells"]
            if cell["supervision"] == "feedback_supervised"
        }
        self.assertEqual(observed_feedback, expected_feedback)
        self.assertTrue(
            all(
                cell["supervision"] == "unsupervised"
                for cell in self.spec["cells"]
                if cell["method"] == "idea"
            )
        )
        unsupervised = self.spec["supervision_contracts"]["unsupervised"]
        self.assertIn("idea", unsupervised["methods"])
        self.assertEqual(unsupervised["provider"], "none")

    def test_hidden_test_is_distinct_contract_ready_feedtta_llm_proxy(self):
        transfer = self.spec["hidden_test_transfer"]
        provider = transfer["provider"]
        self.assertEqual(transfer["status"], "ready")
        self.assertEqual(transfer["submission_job_count"], 5)
        self.assertEqual(len(transfer["submission_cells"]), 5)
        self.assertEqual(
            {item["source_cell_id"] for item in transfer["submission_cells"]},
            {
                "hamt-reverie-eam",
                "duet-reverie-eam",
                "goat-reverie-eam",
                "goat-reverie-feedtta",
                "goat-reverie-idea",
            },
        )
        ordinary = [
            item
            for item in transfer["submission_cells"]
            if item["reported_method_label"] != "FeedTTA-LLM"
        ]
        self.assertEqual(len(ordinary), 4)
        self.assertTrue(all(item["supervision"] == "unsupervised" for item in ordinary))
        proxy = [
            item
            for item in transfer["submission_cells"]
            if item["reported_method_label"] == "FeedTTA-LLM"
        ]
        self.assertEqual(len(proxy), 1)
        self.assertEqual(proxy[0]["feedback_provider"], "qwen2_vl_2b_v1")
        self.assertEqual(transfer["base_cell_id"], "goat-reverie-feedtta")
        self.assertEqual(transfer["reported_method_label"], "FeedTTA-LLM")
        self.assertEqual(transfer["supervision"], "pseudo_label")
        self.assertFalse(transfer["counts_as_search_cell"])
        self.assertIn("FeedTTA", transfer["not_equivalent_to"])
        self.assertIn("IDEA", transfer["not_equivalent_to"])
        self.assertEqual(provider["model_id"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(
            provider["revision"], "895c3a49bc3fa70a340399125c650a463535e71c"
        )
        self.assertEqual(
            provider["weights_sha256"],
            "4faeb74ee719f9c35d7fa254d7cc2d7131e4b4ada7d0340615c66b22d5e16fc5",
        )
        self.assertEqual(
            provider["model_bundle_sha256"],
            "cf7dd27d27987b7ae71458529e6a72e3dcc3db6e91fb7298577e03f88e238a7e",
        )
        self.assertEqual(
            provider["prompt_bundle_sha256"],
            "9d608193a2cd444688c6f507973ab8da68940f4b11439d329195cfd2804e0ed9",
        )
        self.assertEqual(
            [item["prompt_sha256"] for item in provider["pipeline"]],
            [
                "62d6978269100d5129fa2ac612e83e033603bcba50009ab711124fee1c22f0e4",
                "f023385c8c153b87aa20b28ea446ab5ef6a55e632d21fb1f13a48289a7f4d81f",
            ],
        )
        self.assertEqual(provider["budgets"]["binary_labels_maximum"], 6292)
        self.assertEqual(provider["budgets"]["provider_requests_maximum"], 12584)
        self.assertEqual(provider["budgets"]["requests_per_episode"], 2)
        self.assertEqual(provider["pipeline"][1]["output"], "binary_pseudo_success")
        self.assertIn(
            "36_view_final_panorama_at_submitted_reranked_endpoint",
            provider["pipeline"][1]["input"],
        )
        for forbidden in (
            "gt_trajs",
            "_eval_item",
            "distance_to_goal",
            "object_ground_truth",
            "leaderboard_feedback",
        ):
            self.assertIn(forbidden, provider["forbidden_inputs"])
        self.assertEqual(
            provider["required_record"],
            [
                "model_id",
                "revision",
                "weights_sha256",
                "prompt_sha256_values",
                "request_sha256_values",
                "response_sha256_values",
                "parsed_label",
                "latency_seconds_values",
                "cache_key",
                "cache_hit",
                "token_count",
                "cost",
            ],
        )
        runtime = provider["runtime"]
        contract = runtime["contract"]
        contract_path = REPO_ROOT / contract["path"]
        self.assertTrue(contract_path.is_file())
        self.assertEqual(_sha256(contract_path), contract["sha256"])
        self.assertEqual(runtime["contract_sha256"], contract["sha256"])
        self.assertEqual(
            contract["sha256"],
            "f48aa543d41992e647007af9cc4c2e3d4d926923baf6deb452a64b956234dc4d",
        )
        model_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(model_contract["model_id"], provider["model_id"])
        self.assertEqual(model_contract["revision"], provider["revision"])
        self.assertEqual(model_contract["weights_sha256"], provider["weights_sha256"])
        self.assertEqual(
            model_contract["bundle_sha256"], provider["model_bundle_sha256"]
        )
        preflight = transfer["runtime_preflight"]
        self.assertEqual(
            preflight["required_for_submission_id"], proxy[0]["submission_id"]
        )
        self.assertEqual(
            preflight["schema"], "navtta.reverie_llm_feedback_preflight.v1"
        )
        self.assertEqual(
            preflight["render_evidence_schema"],
            "navtta.reverie_mattersim_render_preflight.v1",
        )
        self.assertEqual(
            preflight["path_env"], "NAVTTA_REVERIE_LLM_PREFLIGHT"
        )
        self.assertEqual(
            preflight["render_build_env"],
            "NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD",
        )
        self.assertTrue(preflight["same_git_commit_required"])
        self.assertEqual(
            set(preflight["required_artifacts"]),
            {
                "model_verify", "render_evidence", "rendered_panorama",
                "provider_smoke", "provider_transcript",
            },
        )
        self.assertEqual(
            runtime["model_path"],
            "/data1/wxy/exp_data/NavTTA/vln/models/Qwen2-VL-2B-Instruct",
        )
        self.assertEqual(runtime["service_url"], "http://127.0.0.1:8765")
        self.assertEqual(transfer["local_test_metrics"], "forbidden")


if __name__ == "__main__":
    unittest.main()
