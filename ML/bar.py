import matplotlib.pyplot as plt
import numpy as np


models = ["MfePFN", "LR", "Ridge", "DT", "RF", "GBR", "SAINT", "Tabnet"]
metrics = ["MSE", "MAPE", "1-R2", "MAE"]


data = {
    "1-R2": [0.001396966, 0.00166544, 0.00413895, 0.00192088, 0.00256968, 0.01838042, 0.0938, 0.0561],
    "MSE": [0.000118846, 0.00014169, 0.00035212, 0.00016342, 0.00021861, 0.00156371, 0.0084, 0.0048],
    "MAE": [0.00891710055511789, 0.00910364, 0.01767821, 0.00972574, 0.01121535, 0.03304832, 0.069, 0.0579],
    "MAPE": [0.029257, 0.02985173, 0.05798824, 0.03190219, 0.03679533, 0.10856869, 0.23, 0.1893]
}

fig, ax = plt.subplots(figsize=(12, 8))


x = np.arange(len(models))


colors = ['#F5B46F', '#70A3C4', '#354E97', '#E05B3F']


num_metrics = len(metrics)
total_width = 0.6
width = total_width / num_metrics
gap = 0.02

for i, metric in enumerate(metrics):
    ax.bar(x + i * width + i * gap, data[metric], width, label=metric, color=colors[i])


ax.set_yscale('log')


ax.set_ylabel('Values (log scale)')
ax.set_title('Comparison of Regression Models')
ax.set_xticks(x + (total_width + gap) / 2)
ax.set_xticklabels(models, rotation=45, fontsize=10)
ax.legend()


plt.tight_layout()
plt.savefig("bar_chart.png", dpi=300, bbox_inches='tight')
plt.show()