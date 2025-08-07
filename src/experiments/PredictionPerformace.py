# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.

from src.experiments.experiments_base import ExperimentBase
from torch.utils.data import DataLoader
import torch


class PredictionPerformance(ExperimentBase):

    def __init__(self, dataset,model,seed=None,hash=None,console_mode=False):

        assert dataset is not None, "dataset must be provided"
        assert model is not None, "model must be provided"
        self.experiment_type='PredictionPerformance'
        # parent class loads the configuration file, initialises model and dataset etc.
        super().__init__(model,dataset,self.experiment_type,seed,hash,console_mode)

    def run_experiment(self, default_parameters=False):
        self.logger.info("Running Prediction Performance Evaluation Experiment with model: "+
              self.model_name+ " and dataset: "+ self.dataset_name)
        if default_parameters:
            best_hyperparameters = None
        else:
            best_hyperparameters = self.hyperparameter_tuning()
        model=self.final_training(best_hyperparameters)
        results=self.performance_measurement(model)

        return results

    def load_data(self, return_dataloader=True, shuffle_train=True, batch_size=None):
        """
        Loads the data for the experiment.
        Args:
            return_dataloader (bool, optional): If True, return the DataLoader. Defaults to True.
            shuffle_train (bool, optional): If True, shuffle the training data. Defaults to True.
            batch_size (int, optional): The batch size to use for the DataLoader. If None, the default batch size of the experiment is used.
        Returns:
            dict: Dictionary containing the data.
        """
        train_dataset = self.dataset_class(self.config['dataset'], data_split='train', device=self.device)
        val_dataset = self.dataset_class(self.config['dataset'], data_split='val', device=self.device)
        test_dataset = self.dataset_class(self.config['dataset'], data_split='test', device=self.device)

        if return_dataloader:
            if batch_size is None:
                batch_size = self.config['experiment']['batch_size']
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,#For this experiment we do not want to shuffle the data
                                      collate_fn=self.dataset_class.collate_fn)
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=self.dataset_class.collate_fn)
            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=self.dataset_class.collate_fn)
            return {"train": train_loader, "val": val_loader, "test": test_loader}
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    def performance_measurement(self, model):
        """
        Evaluates the performance of the model on the test set.

        Args:
            model: Trained model to evaluate.
            dataset: Dataset to evaluate the model on.

        Returns:
            dict: The scores for each metric.
        """

        criterion = torch.nn.MSELoss()
        dataloader = self.load_data(return_dataloader=True,shuffle_train=False)
        #Calculate the residuals for the training set
        results_train = model.predict(dataloader['train'])
        true_response = results_train['response_forecast']
        predicted_response = results_train['predictions']
        #Calculate the residuals for the training set

        residuals_train = criterion(true_response , predicted_response).detach().cpu().numpy().item()


        #Calculate the residuals for the validation set
        results_val = model.predict(dataloader['val'])
        true_response = results_val['response_forecast']
        predicted_response = results_val['predictions']
        #Calculate the L1 residuals for the test set
        residuals_val = criterion(true_response , predicted_response).detach().cpu().numpy().item()

        results_test = model.predict(dataloader['test'])
        true_response = results_test['response_forecast']
        predicted_response = results_test['predictions']
        #Calculate the L1 residuals for the test set
        residuals_test = criterion(true_response , predicted_response).detach().cpu().numpy().item()

        res_dict={"residuals_train":residuals_train,"residuals_val":residuals_val,"residuals_test":residuals_test}

        #write the results to a file
        self.write_results(res_dict, 'evaluation_results')

        return res_dict


if __name__ == "__main__":

    dataset = "SMDSystem"
    model = "ModularNeuralLaplace"
    exp = PredictionPerformance(dataset,model)
    #exp.run_experiment(default_parameters=True)
    results=exp.run_experiment()

