import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata

# Step 1: 准备数据
data = {
    'tourists': [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2500000, 2800000, 3000000],  # 游客数量
    'cost': [375, 450, 525, 600, 700, 800, 900, 1000, 1100, 1200],  # 游客消费（百万美元）
    'seasonality': [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.1, 2.5],  # 季节性影响
    'temperature': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44],  # 温度（单位：华氏度）
    'precipitation': [60, 65, 70, 75, 80, 85, 90, 95, 100, 105],  # 降水量（单位：英寸）
    'sunlight': [7.15, 9.4, 11.77, 14.4, 16.78, 18.25, 17.6, 15.43, 12.85, 10.32]  # 日照时长（单位：小时）
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# Step 2: 使用多项式回归来考虑非线性关系
X = df[['tourists', 'cost', 'seasonality', 'temperature', 'precipitation', 'sunlight']]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

y = df['tourists'] * df['cost'] * df['seasonality']
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)

y_pred = poly_regressor.predict(X_poly)
residuals = y - y_pred


# Step 3: 可视化函数 - 绘制每个图表
def plot_actual_vs_predicted(df, y, y_pred):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.scatterplot(x=df['tourists'], y=y, color='green', s=100, label='Actual Income', edgecolor='black',
                    marker='o')
    sns.lineplot(x=df['tourists'], y=y_pred, color='blue', lw=3, label='Predicted Income')
    plt.title('Tourists vs Income (Polynomial Regression)', fontsize=16, fontweight='bold')
    plt.xlabel('Tourists (Thousands)', fontsize=14)
    plt.ylabel('Income (Million Dollars)', fontsize=14)
    plt.legend(title='Legend', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_residuals(df, residuals):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['tourists'], y=residuals, color='purple', s=100, label='Residuals', edgecolor='black',
                    marker='X')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Residuals of the Polynomial Regression Model', fontsize=16, fontweight='bold')
    plt.xlabel('Tourists (Thousands)', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(title='Legend', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted_scatter(y, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, color='green', s=100, label='Predicted vs Actual Income', edgecolor='black',
                    marker='o')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', lw=2)
    plt.title('Actual vs Predicted Income', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Income', fontsize=14)
    plt.ylabel('Predicted Income', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_to_tourists(tourists_range, sensitivity_predictions):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=tourists_range, y=sensitivity_predictions, color='indianred', lw=2,
                 label='Income Sensitivity to Tourists')
    plt.title('Sensitivity of Income to Tourists', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Tourists', fontsize=14)
    plt.ylabel('Predicted Income (Million Dollars)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Legend', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_glacier_area_loss_vs_predicted(y_glacier, y_glacier_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_glacier, y=y_glacier_pred, color='blue', s=100, label='Predicted vs Actual Glacier Area Loss',
                    edgecolor='black', marker='o')
    plt.plot([min(y_glacier), max(y_glacier)], [min(y_glacier), max(y_glacier)], color='red', linestyle='--')
    plt.title('Predicted vs Actual Glacier Area Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Glacier Area Loss', fontsize=14)
    plt.ylabel('Predicted Glacier Area Loss', fontsize=14)
    plt.legend(title='Legend', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_glacier_residuals(y_glacier, residuals_glacier):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_glacier, y=residuals_glacier, color='purple', s=100, label='Residuals', edgecolor='black',
                    marker='X')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Residuals of the Glacier Area Loss Model', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Glacier Area Loss', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(title='Legend', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_to_temperature(temperature_range, sensitivity_predictions_temp):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=temperature_range, y=sensitivity_predictions_temp, color='indianred', lw=2,
                 label='Sensitivity to Temperature')
    plt.title('Sensitivity of Glacier Area Loss to Temperature', fontsize=16, fontweight='bold')
    plt.xlabel('Temperature (°C)', fontsize=14)
    plt.ylabel('Predicted Glacier Area Loss (Square Kilometers)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Legend', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_to_tourists_glacier(tourists_range, sensitivity_predictions_tourists):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=tourists_range, y=sensitivity_predictions_tourists, color='indianred', lw=2,
                 label='Sensitivity to Tourists')
    plt.title('Sensitivity of Glacier Area Loss to Tourists', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Tourists', fontsize=14)
    plt.ylabel('Predicted Glacier Area Loss (Square Kilometers)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Legend', fontsize=12)
    plt.tight_layout()
    plt.show()

#雷达图
def plot_radar_chart(df, features):
    # 归一化特征
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # 计算特征的平均值
    mean_values = df_scaled.mean()

    # 设置雷达图
    categories = list(features)
    values = list(mean_values)

    # 角度计算
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    # 确保图形闭合
    values += values[:1]
    angles += angles[:1]

    # 绘制雷达图
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # 设置渐变色
    ax.fill(angles, values, color='indianred', alpha=0.4)  # 填充颜色，使用透明度
    ax.plot(angles, values, color='indianred', linewidth=3)  # 设置边界颜色和线宽

    # 绘制网格线
    ax.set_yticklabels([])  # 不显示径向标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='darkblue')  # 设置类别标签的字体样式

    # 自定义网格线
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # 设置标题
    plt.title('Radar Chart of Average Features (Normalized)', fontsize=16, fontweight='bold', color='darkblue')

    # 在每个区域的顶端添加标签，表示各个维度
    for i, angle in enumerate(angles[:-1]):
        ax.text(angle, values[i] + 0.1, f"{categories[i]}", horizontalalignment='center', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()

#热力图
def plot_heatmap(df):
    # 计算特征之间的相关性
    corr_matrix = df.corr()

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
    plt.title('Correlation Heatmap of Features', fontsize=16)
    plt.tight_layout()
    plt.show()

#曲面图
def plot_3d_surface(ax, df, x_feature, y_feature, z_feature):
    # 提取要绘制的特征
    x = df[x_feature]
    y = df[y_feature]
    z = df[z_feature]

    # 创建网格进行插值
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                                 np.linspace(y.min(), y.max(), 100))

    # 使用griddata进行插值，生成平滑的z值
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # 绘制曲面图
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # 设置标题和标签
    ax.set_xlabel(x_feature, fontsize=12)
    ax.set_ylabel(y_feature, fontsize=12)
    ax.set_zlabel(z_feature, fontsize=12)
    ax.set_title(f'{x_feature}, {y_feature}, and {z_feature}', fontsize=16)



# Step 7: 输出回归模型的评估指标
def print_model_evaluation(y_glacier, y_glacier_pred):
    mse = mean_squared_error(y_glacier, y_glacier_pred)
    r2 = r2_score(y_glacier, y_glacier_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")


# 主函数
def main():
    # Step 1: 准备数据
    data = {
        'tourists': [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2500000, 2800000, 3000000],  # 游客数量
        'cost': [375, 450, 525, 600, 700, 800, 900, 1000, 1100, 1200],  # 游客消费（百万美元）
        'seasonality': [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.1, 2.5],  # 季节性影响
        'temperature': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44],  # 温度（单位：华氏度）
        'precipitation': [60, 65, 70, 75, 80, 85, 90, 95, 100, 105],  # 降水量（单位：英寸）
        'sunlight': [7.15, 9.4, 11.77, 14.4, 16.78, 18.25, 17.6, 15.43, 12.85, 10.32]  # 日照时长（单位：小时）
    }

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # Step 2: 使用多项式回归来考虑非线性关系
    X = df[['tourists', 'cost', 'seasonality', 'temperature', 'precipitation', 'sunlight']]
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    y = df['tourists'] * df['cost'] * df['seasonality']
    poly_regressor = LinearRegression()
    poly_regressor.fit(X_poly, y)

    y_pred = poly_regressor.predict(X_poly)
    residuals = y - y_pred

    # Step 3: 可视化函数 - 绘制每个图表
    plot_actual_vs_predicted(df, y, y_pred)
    plot_residuals(df, residuals)
    plot_actual_vs_predicted_scatter(y, y_pred)
    plot_radar_chart(df[['tourists', 'cost', 'seasonality', 'temperature', 'precipitation', 'sunlight']], ['tourists', 'cost', 'seasonality', 'temperature', 'precipitation', 'sunlight'])
    plot_heatmap(df)
    fig = plt.figure(figsize=(15, 12))

    # 创建多个子图（2x2）
    ax1 = fig.add_subplot(221, projection='3d')  # 第1个子图
    ax2 = fig.add_subplot(222, projection='3d')  # 第2个子图
    ax3 = fig.add_subplot(223, projection='3d')  # 第3个子图
    ax4 = fig.add_subplot(224, projection='3d')  # 第4个子图

    # 调用3D曲面图函数，绘制每个组合
    plot_3d_surface(ax1, df, 'tourists', 'cost', 'seasonality')
    plot_3d_surface(ax2, df, 'tourists', 'temperature', 'precipitation')
    plot_3d_surface(ax3, df, 'tourists', 'seasonality', 'sunlight')
    plot_3d_surface(ax4, df, 'cost', 'seasonality', 'temperature')

    plt.tight_layout()  # 调整子图间距
    plt.show()
    # Step 4: 灵敏度分析 - 通过改变一个输入特征来评估灵敏度
    tourists_range = np.linspace(1000000, 3000000, 100)  # 从100万到300万游客数量
    sensitivity_predictions = []

    for tourists in tourists_range:
        X_sensitivity = np.array([[tourists, 800, 1.5, 40, 80, 12]])  # 使用固定的其他输入
        X_sensitivity_poly = poly.transform(X_sensitivity)
        sensitivity_predictions.append(poly_regressor.predict(X_sensitivity_poly)[0])

    plot_sensitivity_to_tourists(tourists_range, sensitivity_predictions)

    # Step 5: 使用线性回归分析冰川退缩与气温和游客数量的关系
    glacier_data = {
        'year': [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016],  # 年份
        'temperature': [5.2, 5.5, 5.6, 5.7, 5.8, 6.0, 6.1, 6.3, 6.4, 6.5],  # 年均温度
        'tourists': [100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000],  # 年度游客数量
        'glacier_area_loss': [1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5]  # 冰川面积损失（平方公里）
    }

    # 将数据转换为DataFrame
    glacier_df = pd.DataFrame(glacier_data)

    X_glacier = glacier_df[['temperature', 'tourists']]  # 选取气温和游客数量作为特征
    y_glacier = glacier_df['glacier_area_loss']  # 目标变量是冰川面积损失

    regressor_glacier = LinearRegression()
    regressor_glacier.fit(X_glacier, y_glacier)
    y_glacier_pred = regressor_glacier.predict(X_glacier)
    residuals_glacier = y_glacier - y_glacier_pred

    plot_glacier_area_loss_vs_predicted(y_glacier, y_glacier_pred)
    plot_glacier_residuals(y_glacier, residuals_glacier)

    # Step 6: 灵敏度分析 - 通过改变气温和游客数量来评估冰川退缩的变化
    temperature_range = np.linspace(5.0, 7.0, 100)  # 从5°C到7°C的气温变化
    sensitivity_predictions_temp = []

    for temperature in temperature_range:
        X_sensitivity = np.array([[temperature, 200000]])  # 使用固定的游客数量（200,000）
        sensitivity_predictions_temp.append(regressor_glacier.predict(X_sensitivity)[0])

    plot_sensitivity_to_temperature(temperature_range, sensitivity_predictions_temp)

    tourists_range = np.linspace(100000, 280000, 100)  # 从100,000到280,000游客数量的变化
    sensitivity_predictions_tourists = []

    for tourists in tourists_range:
        X_sensitivity = np.array([[6.0, tourists]])  # 使用固定的气温（6°C）
        sensitivity_predictions_tourists.append(regressor_glacier.predict(X_sensitivity)[0])

    plot_sensitivity_to_tourists_glacier(tourists_range, sensitivity_predictions_tourists)

    # Step 7: 输出回归模型的评估指标
    print_model_evaluation(y_glacier, y_glacier_pred)


# 调用主函数
if __name__ == "__main__":
    main()
