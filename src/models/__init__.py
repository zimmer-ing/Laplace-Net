from .LSTM_Seq2Seq import LSTMSeq2Seq
from .LaplaceNeuralOperator import LaplaceNeuralOperator
from .LaplaceNet import LaplaceNet as LaplaceNet

# Registry dictionary mapping model names to classes
MODEL_REGISTRY = {
    "LSTMSeq2Seq": LSTMSeq2Seq,
    "LaplaceNeuralOperator": LaplaceNeuralOperator,
    "LaplaceNet": LaplaceNet,
}
# Define __all__ based on MODEL_REGISTRY keys
__all__ = list(MODEL_REGISTRY.keys())

