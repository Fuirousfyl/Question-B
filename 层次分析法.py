import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata

# AHP 方法
def ahp(matrix):
    """
    使用 AHP 方法计算权重向量
    :param matrix: 比较矩阵
    :return: 权重向量和一致性比率（CR）
    """
    # Step 1: 归一化矩阵
    col_sum = np.sum(matrix, axis=0)  # 每列求和
    norm_matrix = matrix / col_sum  # 每列归一化

    # Step 2: 计算权重向量
    weight_vector = np.mean(norm_matrix, axis=1)  # 对行取平均

    # Step 3: 计算一致性指标 CI 和一致性比率 CR
    n = matrix.shape[0]
    lamda_max = np.sum(np.dot(matrix, weight_vector) / weight_vector) / n  # 最大特征值
    ci = (lamda_max - n) / (n - 1)  # 一致性指标 CI
    ri_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_values[n]  # 查 RI 表
    cr = ci / ri if ri != 0 else 0  # 一致性比率 CR

    return weight_vector, cr

# 主函数
if __name__ == "__main__":
    # 定义比较矩阵
    comparison_matrix = np.array([
        [1, 3, 5, 7, 9, 2],   # 参数 1（游客数量）对其他参数的重要性
        [1/3, 1, 3, 5, 7, 2], # 参数 2（游客消费）
        [1/5, 1/3, 1, 3, 5, 1], # 参数 3（季节性影响）
        [1/7, 1/5, 1/3, 1, 3, 1/2], # 参数 4（温度）
        [1/9, 1/7, 1/5, 1/3, 1, 1/3], # 参数 5（降水量）
        [1/2, 1/2, 1, 2, 3, 1]  # 参数 6（日照时长）
    ])

    # 计算权重和一致性比率
    weights, cr = ahp(comparison_matrix)

    # 输出权重和一致性比率
    print("参数权重向量:", weights)
    print("一致性比率 (CR):", cr)

    # 检验一致性
    if cr < 0.1:
        print("判断矩阵具有一致性，可以接受！")
    else:
        print("判断矩阵的一致性较差，需要重新调整！")

    # 归一化权重
    weights_normalized = weights / np.sum(weights)
    print("归一化后的参数权重:", weights_normalized)

    # 数据准备
    data = {
        'tourists': [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2500000, 2800000, 3000000],
        'cost': [375, 450, 525, 600, 700, 800, 900, 1000, 1100, 1200],
        'seasonality': [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.1, 2.5],
        'temperature': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        'precipitation': [60, 65, 70, 75, 80, 85, 90, 95, 100, 105],
        'sunlight': [7.15, 9.4, 11.77, 14.4, 16.78, 18.25, 17.6, 15.43, 12.85, 10.32]
    }

    df = pd.DataFrame(data)
    X = df[['tourists', 'cost', 'seasonality', 'temperature', 'precipitation', 'sunlight']]
    y = df['tourists'] * df['cost'] * df['seasonality']

    # 应用权重到数据特征
    weighted_X = X * weights_normalized

    # 使用多项式回归
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(weighted_X)
    poly_regressor = LinearRegression()
    poly_regressor.fit(X_poly, y)
    y_pred = poly_regressor.predict(X_poly)

    # 输出模型评估结果
    print("模型评估:")
    print("均方误差 (MSE):", mean_squared_error(y, y_pred))
    print("R-squared:", r2_score(y, y_pred))

    # 绘制实际 vs 预测
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['tourists'], y=y, color='green', s=100, label='Actual Income', edgecolor='black')
    sns.lineplot(x=df['tourists'], y=y_pred, color='blue', lw=3, label='Predicted Income')
    plt.title('Tourists vs Income (Weighted Polynomial Regression)', fontsize=16, fontweight='bold')
    plt.xlabel('Tourists (Thousands)', fontsize=14)
    plt.ylabel('Income (Million Dollars)', fontsize=14)
    plt.legend(title='Legend', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # 绘制残差
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['tourists'], y=residuals, color='purple', s=100, label='Residuals', edgecolor='black')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Residuals of the Weighted Polynomial Regression Model', fontsize=16, fontweight='bold')
    plt.xlabel('Tourists (Thousands)', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(title='Legend', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
