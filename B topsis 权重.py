import numpy as np
import pandas as pd

# 模拟数据 (行是不同方案，列是指标)
data = {
    'Tourism Revenue': [500, 600, 550, 700],  # f1: 旅游收入
    'Community Satisfaction': [0.8, 0.75, 0.9, 0.7]  # f2: 群众满意度
}

dataframe = pd.DataFrame(data)

# 使用 TOPSIS 方法计算每个方案的综合得分

def topsis(decision_matrix, weights):
    """TOPSIS 方法计算得分"""
    # 步骤 1: 归一化决策矩阵
    normalized_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix ** 2, axis=0))

    # 步骤 2: 加权归一化矩阵
    weighted_matrix = normalized_matrix * weights

    # 步骤 3: 计算理想解 (正理想解和负理想解)
    ideal_best = np.max(weighted_matrix, axis=0)  # 正理想解
    ideal_worst = np.min(weighted_matrix, axis=0)  # 负理想解

    # 步骤 4: 计算与理想解的欧几里得距离
    distance_to_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    distance_to_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

    # 步骤 5: 计算综合得分 (接近度系数)
    scores = distance_to_worst / (distance_to_best + distance_to_worst)
    return scores

# 权重设置 (f1 和 f2 的重要性)
weights = np.array([0.6, 0.4])  # 旅游收入权重: 0.6，满意度权重: 0.4

# 计算 TOPSIS 综合得分
scores = topsis(dataframe.values, weights)

# 将得分添加到数据框中
dataframe['TOPSIS Score'] = scores

# 输出最终排名
dataframe['Rank'] = dataframe['TOPSIS Score'].rank(ascending=False)

# 打印结果
print(dataframe)

# 可视化得分
import matplotlib.pyplot as plt

plt.bar(dataframe.index, dataframe['TOPSIS Score'], color='skyblue')
plt.xlabel('Option Index')
plt.ylabel('TOPSIS Score')
plt.title('TOPSIS Score for Each Option')
plt.xticks(dataframe.index, labels=[f"Option {i+1}" for i in dataframe.index])
plt.show()