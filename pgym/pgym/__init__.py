import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DiscreteVoltVarPowerflow-v0',
    entry_point='pgym.envs:DiscreteVoltVarEnv',
)

register(
    id='ContinuousVoltVarPowerflow-v0',
    entry_point='pgym.envs:ContinuousVoltVarEnv',
)

register(
    id='DContinuousVoltVarPowerflow-v0',
    entry_point='pgym.envs:DContinuousVoltVarEnv',
)

register(
    id='DiscreteBuildingHeatLoad-v0',
    entry_point='pgym.envs:DiscreteHeatLoadEnv',
)

register(
    id='ContinuousBuildingHeatLoad-v0',
    entry_point='pgym.envs:ContinuousHeatLoadEnv',
)
