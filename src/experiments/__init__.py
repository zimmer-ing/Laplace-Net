from .PredictionPerformace import PredictionPerformance

EXPERIMENTS_REGISTRY = {
    "PredictionPerformance": PredictionPerformance
}

# Define __all__ based on EXPERIMENTS_REGISTRY keys
__all__ = list(EXPERIMENTS_REGISTRY.keys())