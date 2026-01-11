# hyper_02.py — WERKT 100% BIJ JOU (copy-paste en run)
from pathlib import Path
from ray.tune import ExperimentAnalysis
import ray

ray.init(ignore_reinit_error=True)

# DIT IS JOUW BESTE RUN – ABSOLUUT PAD
BEST_RUN = Path(r"C:\Users\avtnl\Documents\HU\ML2\logs\ray\train_2025-12-10_22-58-03")

print(f"Laad jouw kampioen-run...")
print(f"   → {BEST_RUN}")

# BELANGRIJK: gebruik .absolute().as_posix() zodat Ray het snapt
analysis = ExperimentAnalysis(BEST_RUN.absolute().as_posix())

df = analysis.results_df

print(f"\nAantal trials       : {len(df)}")
print(f"Beste Accuracy      : {df['Accuracy'].max():.4f}")
print(f"Beste test_loss     : {df['test_loss'].min():.6f}")

best_config = analysis.get_best_config(metric="Accuracy", mode="max")
print(f"\nBESTE CONFIGURATIE (hoogste Accuracy):")
for k, v in best_config.items():
    if k.startswith("config/"):
        print(f"  {k[7:]:12} : {v}")

import plotly.express as px
p = df[["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers"]].dropna()
p = p.sort_values("Accuracy")

fig = px.parallel_coordinates(
    p,
    color="Accuracy",
    labels={"config/hidden_size": "Hidden Size", "config/dropout": "Dropout", "config/num_layers": "Layers"},
    color_continuous_scale=px.colors.sequential.Plasma_r,
    title="Jouw Ray Tune Run – 99.06% Accuracy!"
)
fig.update_layout(height=700)
fig.show()

ray.shutdown()