import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义模型名称和对应的指标值
models = {
    "K-NN": [0.8261, 0.8066, 0.805, 0.8369 ,0.7958, 0.8261],
    "SVM": [1.0000, 0.9996, 0.9900, 0.9995, 0.9890, 0.9783],
    "Ridge": [0.9074,0.9647, 0.8600, 0.9593, 0.8409, 0.8043],
    "RF": [0.6852, 0.7538, 0.7050, 0.6965, 0.6943, 0.7283],
    "TabPFN": [0.9907, 0.9999, 0.9950, 0.9999, 0.9891, 0.9891],
    "tltd": [0.8187, 0.9003, 0.8294, 0.8974, 0.8294, 0.8404],
    "autoXGB": [0.7619, 0.8708, 0.7492, 0.8587, 0.7472, 0.7361],
    "Saint": [0.4050, 0.7840, 0.6705, 0.8460, 0.7520, 0.8630]
}

categories = ['Specificity', 'AUROC', 'Accuracy', 'AUPRC', 'F1', 'RECALL']
num_categories = len(categories)

# 定义雷达图的角度
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

# 复制第一个值到最后一个位置，使雷达图闭合
for model in models:
    models[model] += models[model][:1]

angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
# ax.set_facecolor('lightgray')  # 设置雷达图背景颜色

# 设置标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

# 设置同心圆
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.2', '0.4', '0.6', '0.8', '1.0'], angle=0, fontsize=10)
ax.set_rlabel_position(30)

# 定义颜色列表
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

# 遍历模型并绘制
for idx, (model, values) in enumerate(models.items()):
    ax.plot(angles, values, '-', linewidth=1, label=model, color=colors[idx], markersize=8)
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)

# 添加标题
plt.title('Comparison of Classification Models', position=(0.5, 1.15), fontsize=16, fontweight='bold')

# 显示图表
plt.tight_layout()
plt.savefig("radar_chart.png", dpi=300, bbox_inches='tight')
plt.show()