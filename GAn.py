import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Step 1: 准备数据
data = {
    'tourists': [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2500000, 2800000, 3000000],  # 游客数量
    'spending': [375, 450, 525, 600, 700, 800, 900, 1000, 1100, 1200],  # 游客消费（百万美元）
    'seasonality': [6, 6.1, 6.2, 7.3, 7.5, 8.6, 8.8, 9.0, 9.1, 10],  # 季节性影响
    'temperature': [35, 36, 37, 38, 39, 40, 41, 42, 43, 44],  # 温度（单位：华氏度）
    'precipitation': [60, 65, 70, 75, 80, 85, 90, 95, 100, 105],  # 降水量（单位：英寸）
    'sunlight': [7.15, 9.4, 11.77, 14.4, 16.78, 18.25, 17.6, 15.43, 12.85, 10.32]  # 日照时长（单位：小时）
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# Step 2: 遗传算法优化模型
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 注册遗传算法工具
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 1000000, 3000000)  # 游客数量范围
toolbox.register("attr_float_spending", random.uniform, 300, 1500)  # 消费范围
toolbox.register("attr_float_seasonality", random.uniform, 1.0, 2.5)  # 季节性影响范围

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float, toolbox.attr_float_spending, toolbox.attr_float_seasonality), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 优化目标函数
def tourism_model(individual):
    tourists = individual[0]
    spending = individual[1]
    seasonality = individual[2]

    # 收入计算
    income = tourists * spending * seasonality

    # 环境压力
    temperature = 40  # 固定温度
    environment_impact = 0.01 * tourists * temperature

    # 满意度假设
    satisfaction = max(0, 100 - environment_impact / 100)

    # 目标函数综合
    weight_income = 1.0  # 收入权重
    weight_environment = -0.5  # 环境压力权重（负值表示最小化）
    weight_satisfaction = 0.2  # 满意度权重

    # 添加惩罚项
    penalty = 0
    if environment_impact > 1e6:  # 如果环境压力超过阈值，加入惩罚
        penalty += 1e6 * (environment_impact - 1e6)
    if satisfaction < 70:  # 如果满意度低于70，也加入惩罚
        penalty += 1e4 * (70 - satisfaction)

    # 综合目标函数
    return (
        weight_income * income +
        weight_environment * environment_impact +
        weight_satisfaction * satisfaction -
        penalty,
    )

# 注册遗传算法操作
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", tourism_model)

# 创建初始种群
population = toolbox.population(n=50)

# 设置算法参数
generations = 100
cx_prob = 0.7
mut_prob = 0.2
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# 运行遗传算法
hof = tools.HallOfFame(1)  # 保存最佳个体
algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
                    stats=stats, halloffame=hof, verbose=True)

# 提取最优解
best_individual = hof[0]  # 最优个体
tourists_best = best_individual[0]
spending_best = best_individual[1]
seasonality_best = best_individual[2]

# 计算最优解的收入和其他指标
income_best = tourists_best * spending_best * seasonality_best
temperature = 40
environment_impact_best = 0.01 * tourists_best * temperature
satisfaction_best = max(0, 100 - environment_impact_best / 100)

# 输出最优解
print(f"最佳个体: 游客数量={tourists_best:.0f}, 每游客消费={spending_best:.2f}, 季节性影响={seasonality_best:.2f}")
print(f"最优收入: {income_best:.2f}")
print(f"最优环境压力: {environment_impact_best:.2f}")

