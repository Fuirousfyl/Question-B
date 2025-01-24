import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt

class SustainableTourismModel:
    def __init__(self):
        self.c1, self.c2 = 10, 0.01  # 环境成本参数，降低权重
        self.g1, self.g2 = 0.01, 0.005  # 冰川融化参数
        self.k, self.gamma = 1e-6, 1.2  # 居住压力参数，减小增长速率
        self.V_resident = 50000  # 居民基数

    def environmental_cost(self, V):
        return self.c1 * V + self.c2 * V**2

    def glacier_melt(self, G_t, V_t):
        return G_t + (self.g1 * V_t - self.g2 * G_t)

    def housing_pressure(self, V):
        return self.k * (V / self.V_resident)**self.gamma

# 多目标优化函数
def multi_objective(x):
    V, E = x[0], x[1]
    model = SustainableTourismModel()
    R = 1000 * V  # 调整每位游客的收入
    C = model.environmental_cost(V) - 0.8 * E  # 环境成本调整后
    S = model.housing_pressure(V)  # 居住压力
    G = model.glacier_melt(0.05, V)  # 假设初始冰川融化速度为 0.05

    # 增加保护机制，避免非物理解
    if R <= 0 or C < 0 or G < 0:
        return [float('inf'), float('inf'), float('inf')]

    return [-R, C + S, G]

# DEAP 设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 5000, 100000)  # 调整游客数量范围

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float,), n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", multi_objective)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=5000, up=100000, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=5000, up=100000, eta=20.0, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# 优化求解
population = toolbox.population(n=100)
algorithms.eaMuPlusLambda(
    population, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=50, verbose=False
)

# 蒙特卡洛模拟
def monte_carlo_simulation():
    np.random.seed(42)
    lambda_visitors = 15000  # 平均游客数量
    simulations = 1000  # 模拟次数
    net_returns = []

    for _ in range(simulations):
        V = np.random.poisson(lambda_visitors)  # 随机生成游客数量
        model = SustainableTourismModel()

        # 收入和成本计算
        R = 1000 * V  # 增大收入权重
        E = 0.05 * R  # 环境保护投资与收入挂钩
        C = model.environmental_cost(V) - 0.8 * E  # 环境成本
        S = model.housing_pressure(V)  # 居住压力

        # 计算净收益
        net_return = R - C - S
        net_returns.append(net_return)

        # 调试信息
        print(f"V: {V}, R: {R:.2f}, C: {C:.2f}, S: {S:.2f}, Net Return: {net_return:.2f}")

    # 输出净收益期望
    print(f"净收益期望：${np.mean(net_returns):.2f} ± ${np.std(net_returns):.2f}")

# 绘制 Pareto 前沿
def plot_pareto_front(population):
    fits = np.array([ind.fitness.values for ind in population])
    valid_indices = np.all(np.isfinite(fits), axis=1)  # 确保只绘制有效解
    fits = fits[valid_indices]
    plt.scatter(-fits[:, 0], fits[:, 1], c=fits[:, 2], cmap='viridis')
    plt.xlabel('Net Revenue')
    plt.ylabel('Total Cost')
    plt.colorbar(label='Glacier Melt Rate')
    plt.title('Pareto Front')
    plt.show()

if __name__ == "__main__":
    monte_carlo_simulation()
    plot_pareto_front(population)
