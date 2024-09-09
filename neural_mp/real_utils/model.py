import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from huggingface_hub import PyTorchModelHubMixin
from robomimic.algo import algo_factory
from robomimic.algo.algo import RolloutPolicy
from robomimic.config import config_factory
from robomimic.models.base_nets import DDPModelWrapper


class NeuralMPModel(
    RolloutPolicy,
    PyTorchModelHubMixin,
):
    def __init__(self, config):
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        config_robomimic = config_factory(config["algo_name"], dic=config["config"])
        ObsUtils.initialize_obs_utils_with_config(config_robomimic)
        model = algo_factory(
            config["algo_name"],
            config_robomimic,
            config["obs_key_shapes"],
            config["ac_dim"],
            device=device,
        )
        model.nets["policy"] = DDPModelWrapper(model.nets["policy"])
        model.set_eval()

        super().__init__(model)

    def state_dict(self):
        return self.policy.nets.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        self.policy.nets.load_state_dict(state_dict, strict=strict)

    def eval(self):
        self.policy.set_eval()
