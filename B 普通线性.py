import numpy as np
from scipy.optimize import minimize

# 参数设置
V_max = 1500000  # 基础设施的最大承载能力
alpha = 0.2  # 环境保护投资占总收入的最大比例
P_min = 0.6  # 最低游客满意度
G_max = 0.05  # 冰川融化速度的最大值

# 初始参数
initial_guess = [500000, 250000000, 5000000, 10]  # 初始值: [V, R, E, T]

# 模拟参数
C_per_visitor = 0.05  # 每位游客带来的环境成本 (USD)
I_per_visitor = 0.03  # 每位游客带来的基础设施成本 (USD)
S_per_visitor = 0.02  # 每位游客带来的社会成本 (USD)
P_base = 0.9  # 基础游客满意度
P_drop_rate = 0.000001  # 游客满意度随游客数量下降的速率
G_per_visitor = 1e-6  # 每位游客带来的冰川融化速度

# 目标函数
def objective(vars):
    V, R, E, T = vars  # 决策变量: [游客数量, 旅游收入, 环境保护投资, 游客税]

    # 成本计算
    C = C_per_visitor * V  # 环境成本
    I = I_per_visitor * V  # 基础设施成本
    S = S_per_visitor * V  # 社会成本

    # 净收入 (目标: 最大化收入 - 总成本)
    N = R - (C + I + S)

    # 增加对收入和税收的权重，避免选择游客数为 0
    return -(N) + C + S - 0.1 * T

# 约束条件
def constraint_tourist_capacity(vars):
    V, _, _, _ = vars
    return V_max - V  # 游客数量不能超过承载能力

def constraint_investment(vars):
    _, R, E, _ = vars
    return alpha * R - E  # 环境保护投资不能超过总收入的一定比例

def constraint_satisfaction(vars):
    V, _, _, _ = vars
    P = P_base - P_drop_rate * V  # 游客满意度计算
    return P - P_min  # 游客满意度不能低于阈值

def constraint_glacier_melt(vars):
    V, _, _, _ = vars
    G = G_per_visitor * V  # 冰川融化速度计算
    return G_max - G  # 冰川融化速度不能超过阈值

def constraint_revenue(vars):
    V, R, _, _ = vars
    expected_revenue = V * (100 + vars[3])  # 计算旅游收入
    return R - expected_revenue

# 约束列表
constraints = [
    {"type": "ineq", "fun": constraint_tourist_capacity},
    {"type": "ineq", "fun": constraint_investment},
    {"type": "ineq", "fun": constraint_satisfaction},
    {"type": "ineq", "fun": constraint_glacier_melt},
    {"type": "eq", "fun": constraint_revenue}  # 收入需与游客数量一致
]

# 决策变量边界
bounds = [(1, V_max),  # 游客数量
          (1, None),  # 总收入
          (0, None),  # 环境保护投资
          (0, 100)]  # 游客税

# 优化求解
solution = minimize(
    objective,
    initial_guess,
    bounds=bounds,
    constraints=constraints,
    method='trust-constr',
    options={'maxiter': 1000}
)

# 输出结果
optimal_V, optimal_R, optimal_E, optimal_T = solution.x
optimal_V = max(0, optimal_V)  # 确保输出合理性
optimal_R = max(0, optimal_R)
optimal_E = max(0, optimal_E)
optimal_T = max(0, optimal_T)
print(f"Optimal Daily Visitors: {optimal_V:.0f}")
print(f"Optimal Total Revenue: ${optimal_R:.2f}")
print(f"Optimal Environmental Investment: ${optimal_E:.2f}")
print(f"Optimal Tourist Tax: ${optimal_T:.2f}")



