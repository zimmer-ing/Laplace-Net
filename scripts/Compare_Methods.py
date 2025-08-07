##
from src.experiments import PredictionPerformance
from pathlib import Path
import json
import Constants as const
if __name__=="__main__":
    primary_seed=42
    seeds=[1,2,3,4,5]
    datasets = ["SMDSystem",
                "LNO_1D_Duffing_c0",
                "LNO_1D_Duffing_c05",
                "LNO_1D_Lorenz_rho5",
                "LNO_1D_Lorenz_rho10",
                "LNO_1D_Pendulum_c0",
                "LNO_1D_Pendulum_c05",
                "MackeyGlass"
                ]

    algorithms=["LaplaceNet","LSTMSeq2Seq","LaplaceNeuralOperator"]


    for dataset_name in datasets:
        path_compare_res = Path(const.RESULTS_PATH, "Compare_Methods", dataset_name)
        path_compare_res.mkdir(parents=True, exist_ok=True)
        for model_name in algorithms:
            print(f"Running {model_name} on {dataset_name} for seed {primary_seed}")
            exp = PredictionPerformance(dataset_name, model_name, seed=primary_seed)
            res=exp.run_experiment()
            with open(Path(path_compare_res, f"{model_name}_{primary_seed}.json"), 'w') as f:
                json.dump(res, f)
            best_hparams=exp.load_best_hparams()
            with open(Path(path_compare_res,f"{model_name}_{primary_seed}_best_hparams.json"), 'w') as f:
                json.dump(best_hparams, f)
            print(f"Finished {model_name} on {dataset_name} for seed {primary_seed}")
            for seed in seeds:
                print(f"Running {model_name} on {dataset_name} for seed {seed}")
                best_hparams = exp.load_best_hparams()
                model= exp.final_training(best_hparams,seed)
                results = exp.performance_measurement(model)
                with open(Path(path_compare_res, f"{model_name}_{seed}.json"), 'w') as f:
                    json.dump(results, f)
                print(f"Finished {model_name} on {dataset_name} for seed {seed}")
            exp.finish()
            del exp