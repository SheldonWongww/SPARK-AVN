# /data1/wxy/code/ENMuS/sen/datasets/sound_eventnav_dataset.py
import gzip
import json
import os
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from habitat.core.simulator import AgentState
from sen.tasks.nav import ObjectViewLocation, SoundEventNavEpisode

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    ShortestPathPoint,
)
from habitat.core.utils import DatasetFloatJSONEncoder
from sen.tasks.nav import SoundEventGoal
from navtta_core.experiment.episode_stream import build_tta_episode_stream

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"

@registry.register_dataset(name="SoundEventNav")
class SoundEventNavDataset(Dataset):


    episodes: List[SoundEventNavEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, List[SoundEventGoal]]

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        assert SoundEventNavDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = SoundEventNavDataset(cfg)
        return SoundEventNavDataset._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes
    
    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = dict()
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = SoundEventNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset
    
    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            self.episodes[i].goals = self.goals_by_category[
                self.episodes[i].goals_key
            ]

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = SoundEventNavDataset._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path,
                dataset_dir=dataset_dir,
            )

        last_episode_cnt = 0
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            with gzip.open(scene_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=scene_filename)

            num_episode = len(self.episodes) - last_episode_cnt
            last_episode_cnt = len(self.episodes)
            logging.info('Sampled {} from {}'.format(num_episode, scene))

        episodes_per_scene = int(
            getattr(config, "TTA_EPISODES_PER_SCENE", -1)
        )
        source_manifest_path = str(
            getattr(config, "IDEA_SOURCE_EPISODE_MANIFEST", "") or ""
        )
        source_manifest_sha256 = str(
            getattr(config, "IDEA_SOURCE_EPISODE_MANIFEST_SHA256", "") or ""
        )
        if bool(source_manifest_path) != bool(source_manifest_sha256):
            raise ValueError(
                "IDEA source manifest path and SHA256 must be supplied together"
            )
        if source_manifest_path and episodes_per_scene > 0:
            raise ValueError(
                "IDEA source selection and target TTA stream sampling are "
                "mutually exclusive"
            )
        # ``get_scenes_to_load`` constructs a temporary dataset with an empty
        # CONTENT_SCENES list.  Apply the pinned order only to the real dataset
        # instance after its scene content has been loaded.
        if scenes and source_manifest_path:
            from navtta_avn.idea_source import (
                load_source_manifest,
                select_manifest_episodes,
            )

            if str(config.SPLIT).lower() != "train":
                raise ValueError(
                    "IDEA source episode manifests are valid only on train"
                )
            data_path = str(config.DATA_PATH).lower()
            if "multi_source" in data_path:
                source_setting = "multi_source"
            elif "single_source" in data_path:
                source_setting = "single_source"
            else:
                raise ValueError(
                    "cannot infer ENMuS source setting from DATA_PATH"
                )
            manifest, _ = load_source_manifest(
                source_manifest_path,
                source_manifest_sha256,
                model="enmus",
                source_setting=source_setting,
            )
            self.episodes = select_manifest_episodes(self.episodes, manifest)
            logging.info(
                "Loaded digest-pinned IDEA source order with %d episodes "
                "(%s).",
                len(self.episodes),
                manifest["episode_order_sha256"],
            )
        if scenes and episodes_per_scene > 0:
            expected_scenes = int(config.TTA_EXPECTED_SCENES)
            episode_seed = int(config.TTA_EPISODE_SEED)
            global_shuffle = bool(config.TTA_GLOBAL_SHUFFLE)
            self.episodes = build_tta_episode_stream(
                self.episodes,
                episodes_per_scene=episodes_per_scene,
                expected_scenes=expected_scenes,
                seed=episode_seed,
                global_shuffle=global_shuffle,
            )
            logging.info(
                "Built TTA stream with %d scenes x %d episodes "
                "(seed=%d, global_shuffle=%s).",
                expected_scenes,
                episodes_per_scene,
                episode_seed,
                global_shuffle,
            )

    def filter_by_ids(self, scene_ids):
        episodes_to_keep = list()

        for episode in self.episodes:
            for scene_id in scene_ids:
                scene, ep_id = scene_id.split(',')
                if scene in episode.scene_id and ep_id == episode.episode_id:
                    episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes(self, scene):
        episodes_to_keep = list()

        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[3]
            if scene == episode_scene:
                episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = SoundEventNavEpisode(**episode)
            # a temporal workaround to set scene_dataset_config attribute
            episode.scene_dataset_config = self._config.SCENES_DIR.split('/')[-1]

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = self.__deserialize_goal(goal)
            
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, action in enumerate(path):
                        path[p_index] = ShortestPathPoint(
                            [np.inf, np.inf, np.inf], 
                            [np.inf, np.inf, np.inf], 
                            action,
                        )

            if hasattr(self._config, 'CONTINUOUS') and self._config.CONTINUOUS:
                episode.goals[0].position[1] += 0.1
                for view_point in episode.goals[0].view_points:
                    view_point.agent_state.position[1] += 0.1

            self.episodes.append(episode)
            # episode_cnt += 1

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> SoundEventGoal:
        g = SoundEventGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            # modify the generation of view points
            # view_location = ObjectViewLocation(view, iou=0)
            view_location = ObjectViewLocation(AgentState(position=serialized_goal["view_points"][vidx]), iou=0)
            g.view_points[vidx] = view_location

        return g
