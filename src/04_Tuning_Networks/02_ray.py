# Original Notebook | NOT WORKING ON LAPTOP!!!
from pathlib import Path
tune_dir = Path("logs/ray").resolve()
tune_dir.exists()

tunelogs = [d for d in tune_dir.iterdir()]
tunelogs.sort()
latest = tunelogs[-1]
latest

from ray.tune import ExperimentAnalysis
import ray
ray.init(ignore_reinit_error=True)


analysis = ExperimentAnalysis(latest)

analysis.results_df.columns

import plotly.express as px

plot = analysis.results_df
select = ["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers"]
p = plot[select].reset_index().dropna()

p.sort_values("Accuracy", inplace=True)

px.parallel_coordinates(p, color="Accuracy")

import seaborn as sns

sns.scatterplot(data=p, x="config/hidden_size", y="config/num_layers", hue="Accuracy", palette="coolwarm")

import matplotlib.pyplot as plt
cmap = sns.cubehelix_palette(as_cmap=True)
sns.scatterplot(data=p, x="config/hidden_size", y="config/num_layers", hue="Accuracy", palette="coolwarm")
sns.kdeplot(data=p, x="config/hidden_size", y="config/num_layers", cmap=cmap)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

analysis.get_best_trial(metric="test_loss", mode="min")

p[-10:]

analysis.get_best_config(metric="Accuracy", mode="max")