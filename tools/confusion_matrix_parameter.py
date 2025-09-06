import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # CLS-CIFAR100（10%）sin
# cm = np.array([
#     [71.61, 71.64, 71.68, 71.65, 71.63],
#     [71.63, 71.89, 71.92, 71.87, 71.66],
#     [71.79, 72.25, 72.28, 72.23, 71.88],
#     [71.62, 71.95, 71.99, 71.92, 71.76],
#     [71.58, 71.71, 71.81, 71.73, 71.72]
# ])

# # CLS-CIFAR100（10%）cos
# cm = np.array([
#     [71.59, 71.56, 71.54, 71.51, 71.45],
#     [71.61, 71.58, 71.55, 71.53, 71.49],
#     [71.65, 71.62, 71.59, 71.55, 71.51],
#     [71.68, 71.65, 71.62, 71.56, 71.53],
#     [71.72, 71.69, 71.65, 71.59, 71.55]
# ])

# # DET-VOC07+12（10%）sin
# cm = np.array([
#     [83.99, 84.08, 84.27, 84.22, 84.14],
#     [83.96, 84.05, 84.21, 84.13, 84.06],
#     [84.15, 84.57, 84.91, 84.83, 84.79],
#     [84.01, 84.24, 84.73, 84.62, 84.59],
#     [84.06, 84.39, 84.85, 84.74, 84.65]
# ])

# DET-VOC07+12（10%）cos
cm = np.array([
    [83.68, 83.65, 83.56, 83.51, 83.41],
    [83.82, 83.73, 83.72, 83.68, 83.61],
    [83.75, 83.76, 83.65, 83.62, 83.55],
    [84.01, 83.89, 83.77, 83.71, 83.62],
    [83.88, 83.81, 83.79, 83.78, 83.67]
])

# 将混淆矩阵转换为小数形式
cm_decimal = cm
# cm_decimal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 类别标签
x_labels = ['500', '1000', '1500', '2000', '2500']
y_labels = ['100', '200', '300', '400', '500']

# 设置字体为 Times New Roman 并增大字体
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 18})  # 增加字体大小

# 绘制小数形式的混淆矩阵
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm_decimal, annot=True, fmt=".2f", cmap='Blues', cbar=True,
            xticklabels=x_labels, yticklabels=y_labels, vmin=np.min(cm), vmax=np.max(cm))

# ax = sns.heatmap(cm_decimal, annot=True, fmt=".2f", cmap='Blues', cbar=True,
#             xticklabels=labels, yticklabels=labels,
#             annot_kws={"color": "white"})  # 设置文本颜色为黑色

plt.xlabel('Values of Loops')
plt.ylabel('Values of Thresholds')
plt.tight_layout()

plt.show()