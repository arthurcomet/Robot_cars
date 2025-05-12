import json
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

class RobocarClient:
    def __init__(self, port=5004):
        self.port = port
        self.env = None
        self.engine_config_channel = EngineConfigurationChannel()
        self.config_path = "config.json"
        self.agents = []

    def create_config(self):
        config = {
            "agents": [
                {"fov": 180, "nbRay": 10},
                {"fov": 48, "nbRay": 36}
            ]
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def connect(self):
        try:
            self.create_config()
            self.engine_config_channel.set_configuration_parameters(
                width=1280, height=720, quality_level=5,
                time_scale=1.0, target_frame_rate=60, capture_frame_rate=60
            )
            self.env = UnityEnvironment(
                file_name="./RacingSimulator.x86_64",
                side_channels=[self.engine_config_channel],
                base_port=self.port,
                no_graphics=False,
                worker_id=0,
                additional_args=[
                    "--config-path", self.config_path,
                    "--screen-fullscreen", "0",
                    "--screen-width", "1280",
                    "--screen-height", "720"
                ]
            )
            self.env.reset()
            behavior_names = list(self.env.behavior_specs.keys())
            if behavior_names:
                self.primary_behavior = behavior_names[0]
                return True
            return False
        except Exception as e:
            print(f"Erreur de connexion: {e}")
            return False

    def get_observations(self):
        if self.env:
            decision_steps, _ = self.env.get_steps(self.primary_behavior)
            if len(decision_steps) > 0:
                return decision_steps[0].obs
        return None

    def send_actions(self, actions):
        if self.env:
            action_tuple = ActionTuple(continuous=actions)
            self.env.set_actions(self.primary_behavior, action_tuple)
            self.env.step()

    def close(self):
        if self.env:
            self.env.close()
