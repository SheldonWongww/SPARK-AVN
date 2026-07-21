"""
TTA Adapter Module for Audio-Visual Navigation
================================================
Bridges ENMuS-format datasets (SoundEventNav) to work with ss_baselines models
(av_nav, av_wan, savi) using the SoundSpacesSim simulator.

ENMuS episodes have extra fields (interval_mean, interval_upper_limit, etc.)
and store distractor position as 3D coords instead of graph node index.
This adapter handles these format differences.
"""

import os
import json
import gzip
import logging
from typing import Any, Dict, List, Optional, Type

import attr
import numpy as np

from navtta_core.experiment.episode_stream import build_tta_episode_stream

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.tasks.nav.nav import NavigationTask, NavigationEpisode, NavigationGoal
from habitat.core.utils import not_none_validator

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


def optional_int(value):
    return int(value) if value is not None else None


@attr.s(auto_attribs=True, kw_only=True)
class TTAAudioNavEpisode(NavigationEpisode):
    """Episode class that supports both ENMuS and original SemanticAudioNav fields."""
    object_category: str
    sound_id: str
    offset: attr.ib(converter=int)
    duration: attr.ib(converter=int)

    # ENMuS-specific fields (interval timing for intermittent sounds)
    interval_mean: Optional[int] = attr.ib(default=None, converter=optional_int)
    interval_upper_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    interval_lower_limit: Optional[int] = attr.ib(default=None, converter=optional_int)

    # Distractor fields - ENMuS format (dict with position) or original (position_index)
    distractor_sound_id: Optional[str] = attr.ib(default=None)
    distractor: Optional[Dict] = attr.ib(default=None)  # ENMuS: {"position": [x,y,z], ...}
    distractor_position_index: Optional[int] = attr.ib(default=None)  # original format
    distractor_offset: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_duration: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_interval_mean: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_interval_upper_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_interval_lower_limit: Optional[int] = attr.ib(default=None, converter=optional_int)

    # Noise fields (ENMuS only)
    noise_sound_id: Optional[str] = attr.ib(default=None)
    noise_duration: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_offset: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_interval_mean: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_interval_upper_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_interval_lower_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_positions: Optional[List[List[float]]] = attr.ib(default=None)

    @property
    def goals_key(self) -> str:
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class TTAAudioGoal(NavigationGoal):
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_dataset(name="TTAAudioNav")
class TTAAudioNavDataset(Dataset):
    """Dataset class that loads ENMuS-format episodes for use with ss_baselines models."""
    episodes: List[TTAAudioNavEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config, **kwargs) -> List[str]:
        assert TTAAudioNavDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = TTAAudioNavDataset(cfg)
        return TTAAudioNavDataset._get_scenes_from_folder(
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

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = TTAAudioNavDataset._get_scenes_from_folder(
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
            logging.debug('Sampled {} from {}'.format(num_episode, scene))

        episodes_per_scene = int(
            getattr(config, "TTA_EPISODES_PER_SCENE", -1)
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
        else:
            logging.info(
                "Sampled %d episodes from %d scenes.",
                len(self.episodes),
                len(scenes),
            )

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> TTAAudioGoal:
        g = TTAAudioGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(view, iou=0)
            view_location.agent_state = AgentState(view_location.agent_state)
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return

        for i, episode in enumerate(deserialized["episodes"]):
            # Filter out any fields not in TTAAudioNavEpisode to be safe
            episode = TTAAudioNavEpisode(**episode)
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
                    for p_index, point in enumerate(path):
                        if isinstance(point, dict):
                            path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)


@registry.register_task(name="TTAAudioNav")
class TTAAudioNavigationTask(NavigationTask):
    """Task that bridges ENMuS episode format to SoundSpacesSim configuration.

    Handles:
    - Standard fields (sound_id, offset, duration, goal_position)
    - Distractor position conversion (3D coords -> stored for sim to convert to index)
    """

    def overwrite_sim_config(
            self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)


def merge_sim_episode_config(
        sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
            episode.start_position is not None
            and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.GOAL_POSITION = episode.goals[0].position
        agent_cfg.SOUND_ID = episode.sound_id
        agent_cfg.OFFSET = episode.offset
        agent_cfg.DURATION = episode.duration

        # Handle distractor - ENMuS format (dict with position)
        if hasattr(episode, 'distractor_sound_id') and episode.distractor_sound_id is not None:
            agent_cfg.DISTRACTOR_SOUND_ID = episode.distractor_sound_id
            if hasattr(episode, 'distractor') and episode.distractor is not None:
                # ENMuS format: distractor is a dict with "position"
                agent_cfg.DISTRACTOR_POSITION = episode.distractor["position"]
            elif hasattr(episode, 'distractor_position_index') and episode.distractor_position_index is not None:
                agent_cfg.DISTRACTOR_POSITION_INDEX = episode.distractor_position_index

        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config
