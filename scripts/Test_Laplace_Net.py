### This script is used to test the LaplaceNet model on the SMDSystem dataset. You can run it to check if the model works correctly.

from src.experiments import PredictionPerformance

if __name__=="__main__":
    dataset = "SMDSystem"
    model = "LaplaceNet"
    exp = PredictionPerformance(dataset, model,console_mode=True,hash='Test')
    exp.final_training()
