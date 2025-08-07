import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import Constants as CONST
import json

plt.rc('text', usetex=True)
plt.rc('font', family='serif')



def load_hparams(path: Path, name_algo, seed: list):


    path_seed = Path(path / f"{name_algo}_{seed}_best_hparams.json")
    if not path_seed.exists():
        print("Warning: File not found:", path_seed, "Filling with NaN")
        return {}
    with open(path_seed, 'r') as f:
        data = json.load(f)

    return data

def load_results_for_algo(path: Path, name_algo, metrics: list, seeds: list):
    df = pd.DataFrame(index=seeds, columns=metrics)
    for seed in seeds:
        path_seed = Path(path / f"{name_algo}_{seed}.json")
        if not path_seed.exists():
            for metric in metrics:
                df.loc[seed, metric] = np.nan
            print("Warning: File not found:", path_seed, "Filling with NaN")
            continue
        with open(path_seed, 'r') as f:
            data = json.load(f)
        for metric in metrics:
            df.loc[seed, metric] = data.get(metric, np.nan)
    return df


def process_algorithms(algos_dict, group_name, base_path, path_viz, path_tables, datasets, metrics, seeds):
    dfs = {}
    for dataset in datasets:
        path = Path(base_path, dataset)
        dfs[dataset] = {}
        for algo in algos_dict:
            dfs[dataset][algo] = load_results_for_algo(path, algo, metrics, seeds)

    for metric in metrics:
        data_list = []
        for dataset in datasets:
            for algo in algos_dict:
                values = dfs[dataset][algo][metric].dropna().astype(float).values
                for val in values:
                    data_list.append({"Dataset": rename_dict["datasets"].get(dataset, dataset),
                                      "Algorithm": algos_dict[algo],
                                      metric: val})
        df_metric = pd.DataFrame(data_list)

        plt.figure(figsize=(8, 6))
        min_val=df_metric[metric].min()
        ax = sns.boxplot(x="Algorithm", y=metric, hue="Dataset", data=df_metric, showfliers=False, width=0.5)
        ax.set_title(r'\textbf{Boxplot of ' + rename_dict["metrics"].get(metric, metric) + r'} (' + group_name + ')',
                     fontsize=20)
        ax.set_xlabel(r'\textbf{Model}', fontsize=18)
        ax.set_ylabel(rename_dict["metrics"].get(metric, metric), fontsize=18)


        ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.xticks(fontsize=18, rotation=45)
        plt.yticks(fontsize=18)
        plt.legend(title="Dataset", title_fontsize=18, fontsize=18)
        plt.tight_layout()

        plt.savefig(Path(path_viz, f"{metric}_boxplot_{group_name}.pdf"))
        plt.savefig(Path(path_viz, f"{metric}_boxplot_{group_name}.png"))
        plt.close()

        table_data = {}
        for algo in algos_dict:
            table_data[algos_dict[algo]] = {}
            for dataset in datasets:
                values = dfs[dataset][algo][metric].dropna().astype(float).values
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    table_data[algos_dict[algo]][
                        rename_dict["datasets"].get(dataset, dataset)] = f"{mean_val:.2e} ({std_val:.2e})"
                else:
                    table_data[algos_dict[algo]][rename_dict["datasets"].get(dataset, dataset)] = "n.a."

        table_df = pd.DataFrame(table_data)


        for index, row in table_df.iterrows():
            try:
                min_val = min(float(value.split(" ")[0]) for value in row if
                              "n.a." not in value)  # Kleinsten Wert pro Zeile finden
                for col in table_df.columns:
                    if "n.a." not in table_df.loc[index, col]:  # Nur numerische Werte betrachten
                        val = float(table_df.loc[index, col].split(" ")[0])
                        if val == min_val:
                            table_df.loc[index, col] = f"\\textbf{{{table_df.loc[index, col]}}}"  # Fetten Wert setzen
            except ValueError:
                continue

        table_df.index.name = "Dataset"

        latex_table = table_df.to_latex(escape=False)
        latex_table = (
            "\\begin{table*}[htb]\n\\centering\n"
            "\\caption{Mean ± (standard deviation) for "
            f"{rename_dict['metrics'].get(metric, metric)}."+"\\textbf{Bold} values indicate the best result per data set}\n"
            f"\\label{{tab:{metric}_{group_name}}}\n"
            f"{latex_table}\n"
            "\\end{table*}"
        )

        table_file = Path(path_tables, f"{metric}_table_{group_name}.tex")
        with open(table_file, 'w') as f:
            f.write(latex_table)

        table_df.to_csv(Path(path_tables, f"{metric}_table_{group_name}.csv"))

        #load hyperparameters for all datasets per algo and put them into a table
        for algo in algos_dict:
            hparams=[]
            for dataset in datasets:
                path=Path(base_path, dataset)
                hparams.append(load_hparams(path, algo, seeds[0]))
            #to dataframe and then to table
            df=pd.DataFrame(hparams)
            df.to_csv(Path(path_tables, f"{algo}_hparams_{group_name}.csv"))
            df.to_latex(Path(path_tables, f"{algo}_hparams_{group_name}.tex"))
        # --- Additionally: Create table in two-line format ---

        # Create separate dictionaries for means and standard deviations
        table_data_means = {}
        table_data_stds = {}
        for algo in algos_dict:
            table_data_means[algos_dict[algo]] = {}
            table_data_stds[algos_dict[algo]] = {}
            for dataset in datasets:
                ds_name = rename_dict["datasets"].get(dataset, dataset)
                values = dfs[dataset][algo][metric].dropna().astype(float).values
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    table_data_means[algos_dict[algo]][ds_name] = f"{mean_val:.2e}"
                    table_data_stds[algos_dict[algo]][ds_name] = f"({std_val:.2e})"
                else:
                    table_data_means[algos_dict[algo]][ds_name] = "n.a."
                    table_data_stds[algos_dict[algo]][ds_name] = ""

        # Convert dictionaries to DataFrames
        table_means = pd.DataFrame(table_data_means)
        table_stds = pd.DataFrame(table_data_stds)
        table_means.index.name = "Dataset"
        table_stds.index.name = "Dataset"

        # Bold the minimal (best) mean value per row (dataset)
        for index, row in table_means.iterrows():
            try:
                # Convert values; ignore "n.a." entries
                numeric_values = {col: float(val) for col, val in row.items() if val != "n.a."}
                if numeric_values:
                    min_val = min(numeric_values.values())
                    for col in table_means.columns:
                        if row[col] != "n.a." and float(row[col]) == min_val:
                            table_means.loc[index, col] = f"\\textbf{{{row[col]}}}"
            except ValueError:
                continue

        # Bold the minimal (best) standard deviation per row (dataset)
        for index, row in table_stds.iterrows():
            try:
                numeric_values = {}
                for col, val in row.items():
                    if val and val != "n.a.":
                        # Remove parentheses and convert to float
                        numeric_val = float(val.strip("()"))
                        numeric_values[col] = numeric_val
                if numeric_values:
                    min_val = min(numeric_values.values())
                    for col in table_stds.columns:
                        current_val = table_stds.loc[index, col]
                        if current_val and current_val != "n.a.":
                            if float(current_val.strip("()")) == min_val:
                                table_stds.loc[index, col] = f"\\textbf{{{current_val}}}"
            except ValueError:
                continue

        # Combine the two DataFrames: for each dataset, add two rows (first: mean, second: standard deviation)
        combined_rows = []
        new_index = []
        for ds in table_means.index:
            combined_rows.append(table_means.loc[ds])
            combined_rows.append(table_stds.loc[ds])
            new_index.append(ds)
            new_index.append("")  # Second row without dataset name
        combined_table = pd.DataFrame(combined_rows, index=new_index)

        # Convert to LaTeX format
        latex_table_two_line = combined_table.to_latex(escape=False)
        latex_table_two_line = (
            "\\begin{table*}[htb]\n\\centering\n"
            "\\caption{Mean and standard deviation (in parentheses) for "
            f"{rename_dict['metrics'].get(metric, metric)}."+"\\textbf{Bold} values indicate the best result per dataset}\n"
            f"\\label{{{{tab:{metric}_{group_name}_two_line}}}}\n"
            f"{latex_table_two_line}\n"
            "\\end{table*}"
        )

        # Save the two-line table in an additional file
        table_file_two_line = Path(path_tables, f"{metric}_table_{group_name}_two_line.tex")
        with open(table_file_two_line, 'w') as f:
            f.write(latex_table_two_line)

base_path = Path(CONST.RESULTS_PATH, "Compare_Methods")
path_viz = Path(base_path, "viz_paper")
path_viz.mkdir(exist_ok=True, parents=True)
path_tables = Path(base_path, "tables")
path_tables.mkdir(exist_ok=True, parents=True)

algos = {
    "LaplaceNeuralOperator": "LNO",
    "LSTMSeq2Seq": "LSTM",
    #"ModularNeuralLaplace": "LP-Net",
    #"ModularNeuralLaplace_corr": "LP-Net iso.",
    "LaplaceNet": "LP-Net"
}

rename_dict = {
    "metrics": {
        "residuals_test": "MSE on test set",
    },
    "datasets": {
        "SMDSystem": "SMD System",
        "LNO_1D_Duffing_c0": "Duffing $c=0$",
        "LNO_1D_Duffing_c05": "Duffing $c=0.5$",
        "LNO_1D_Lorenz_rho5": "Lorenz $\\rho=5$",
        "LNO_1D_Lorenz_rho10": "Lorenz $\\rho=10$",
        "LNO_1D_Pendulum_c0": "Pendulum $c=0$",
        "LNO_1D_Pendulum_c05": "Pendulum $c=0.5$",
    }
}
datasets = ["SMDSystem",
            "LNO_1D_Duffing_c0",
            "LNO_1D_Duffing_c05",
            "LNO_1D_Lorenz_rho5",
            "LNO_1D_Lorenz_rho10",
            "LNO_1D_Pendulum_c0",
            "LNO_1D_Pendulum_c05",
            "MackeyGlass"
            ]
metrics = ["residuals_test"]
seeds = [42,1,2,3,4,5]

process_algorithms(algos, "Forecasting-Based", base_path, path_viz, path_tables, datasets, metrics, seeds)

