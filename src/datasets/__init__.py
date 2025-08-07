from .SMD_Dataset import SMDSystem
from .LNOData import LNO_1D_Duffing_c0,LNO_1D_Duffing_c05,LNO_1D_Pendulum_c0,LNO_1D_Pendulum_c05,LNO_1D_Lorenz_rho5,LNO_1D_Lorenz_rho10
from .MackeyGlass import MackeyGlassSystem
# Define aliases explicitly

SMDSystem = SMDSystem


DATASET_REGISTRY = {
    "SMDSystem": SMDSystem,
    "LNO_1D_Duffing_c0": LNO_1D_Duffing_c0,
    "LNO_1D_Duffing_c05": LNO_1D_Duffing_c05,
    "LNO_1D_Pendulum_c0": LNO_1D_Pendulum_c0,
    "LNO_1D_Pendulum_c05": LNO_1D_Pendulum_c05,
    "LNO_1D_Lorenz_rho5": LNO_1D_Lorenz_rho5,
    "LNO_1D_Lorenz_rho10": LNO_1D_Lorenz_rho10,
    "MackeyGlass": MackeyGlassSystem,

}
# Add aliases and registry keys to __all__
__all__ = [DATASET_REGISTRY.keys()]