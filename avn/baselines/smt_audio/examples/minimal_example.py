'''
演示 SoundSpaces 中最基本的声音渲染流程
展示了如何在一个指定的3D场景中，设置一个声源和一个听者，然后计算并保存从声源传播到听者的室内脉冲响应（Room Impulse Response, RIR）
'''

import quaternion

import habitat_sim.sim
import numpy as np
from scipy.io import wavfile

# 配置并初始化 Habitat-Sim, 加载场景
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()

cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

# 配置并添加音频传感器
audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
audio_sensor_spec.enableMaterials = True # make sure _semantic.ply file is in the scene folder
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
audio_sensor_spec.channelLayout.channelCount = 1
audio_sensor_spec.position = [0.0, 1.5, 0.0]
audio_sensor_spec.acousticsConfig.sampleRate = 16000
audio_sensor_spec.acousticsConfig.indirect = True
sim.add_sensor(audio_sensor_spec)

# 设置声学环境
audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
audio_sensor.setAudioSourceTransform(np.array([-8.56, 1.5, 0.50])) # 声源位置
audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
agent = sim.get_agent(0)
new_state = sim.get_agent(0).get_state()
new_state.position = np.array([-10.57, 0, -0.25]) # 听者位置
new_state.sensor_states = {}
agent.set_state(new_state, True)

# 执行声音渲染并保存
obs = np.array(sim.get_sensor_observations()["audio_sensor"])
wavfile.write('data/output.wav', 16000, obs.T)

