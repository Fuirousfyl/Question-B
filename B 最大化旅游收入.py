import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 参数设置
MAX_VISITORS = 1500000  # 环境承载力的游客数量限制
MIN_REVENUE = 200000000  # 最低旅游收入要求
MAX_DISSATISFACTION = 0.2  # 群众和游客的不满比例上限
MAX_GLACIER_MELT = 0.01  # 冰川融化速度上限
CARBON_PER_VISITOR = 0.1  # 每位游客的碳排放量 (tons)
INITIAL_VISITORS = 1600000  # 初始年度游客数量
INITIAL_REVENUE = 375000000  # 初始收入 (USD)
INITIAL_SATISFACTION = 0.8  # 初始社区满意度 (0-1)
YEARS = 20  # 模拟时间

# 目标函数和约束

def objective(vars):
    """多目标优化函数，平衡收入、满意度和环境保护"""
    visitor_tax, visitor_cap = vars

    # 模拟的输出变量
    revenue = []
    dissatisfaction = []
    glacier_melt = []
    visitors = INITIAL_VISITORS
    satisfaction = INITIAL_SATISFACTION

    for year in range(YEARS):
        # 限制游客数量
        visitors = max(0, min(visitors, visitor_cap))

        # 计算收入
        annual_revenue = visitors * (100 + visitor_tax)
        revenue.append(annual_revenue)

        # 计算不满比例
        dissatisfaction_ratio = max(0, (visitors - MAX_VISITORS) / MAX_VISITORS)
        dissatisfaction.append(dissatisfaction_ratio)

        # 冰川融化速度 (假设与游客数量和碳排放成正比)
        glacier_melt_speed = visitors * CARBON_PER_VISITOR * 1e-6
        glacier_melt.append(glacier_melt_speed)

        # 更新下一年的游客数量
        visitors *= 0.99  # 自然减少 1%

    # 总收入
    total_revenue = sum(revenue)

    # 平均不满比例
    average_dissatisfaction = np.mean(dissatisfaction)

    # 总冰川融化速度
    total_glacier_melt = sum(glacier_melt)

    # 综合目标：最大化收入，最小化不满和冰川融化
    return -total_revenue + 1e6 * average_dissatisfaction + 1e6 * total_glacier_melt

# 约束条件
def constraint_population(vars):
    _, visitor_cap = vars
    return MAX_VISITORS - visitor_cap  # 人口限制: 小于环境承载力

def constraint_revenue(vars):
    visitor_tax, visitor_cap = vars
    visitors = visitor_cap  # 假设游客达到限额
    revenue = visitors * (100 + visitor_tax)
    return revenue - MIN_REVENUE  # 收入需高于最低要求

def constraint_dissatisfaction(vars):
    _, visitor_cap = vars
    dissatisfaction_ratio = max(0, (visitor_cap - MAX_VISITORS) / MAX_VISITORS)
    return MAX_DISSATISFACTION - dissatisfaction_ratio  # 不满比例需低于上限

def constraint_glacier_melt(vars):
    _, visitor_cap = vars
    glacier_melt_speed = visitor_cap * CARBON_PER_VISITOR * 1e-6  # 假设年均计算
    return MAX_GLACIER_MELT - glacier_melt_speed  # 冰川融化速度需低于上限

# 初始猜测和边界
x0 = [20, 1400000]  # 初始猜测: 访客税 20 美元，游客限额 140 万
bounds = [(5, 50), (500000, 2000000)]  # 访客税和游客限额的边界
constraints = [
    {"type": "ineq", "fun": constraint_population},
    {"type": "ineq", "fun": constraint_revenue},
    {"type": "ineq", "fun": constraint_dissatisfaction},
    {"type": "ineq", "fun": constraint_glacier_melt}
]

# 优化求解
solution = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 500})

# 最优结果
optimal_tax, optimal_cap = solution.x
print(f"Optimal Visitor Tax: ${optimal_tax:.2f}")
print(f"Optimal Visitor Cap: {optimal_cap:.0f} visitors")

# 可视化结果
years = range(YEARS)
visitors = INITIAL_VISITORS
revenue = []
dissatisfaction = []
glacier_melt = []

for year in years:
    visitors = max(0, min(visitors, optimal_cap))
    annual_revenue = visitors * (100 + optimal_tax)
    dissatisfaction_ratio = max(0, (visitors - MAX_VISITORS) / MAX_VISITORS)
    glacier_melt_speed = visitors * CARBON_PER_VISITOR * 1e-6

    revenue.append(annual_revenue)
    dissatisfaction.append(dissatisfaction_ratio)
    glacier_melt.append(glacier_melt_speed)

    visitors *= 0.99

# 绘制结果
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(years, revenue, label="Revenue")
plt.title("Annual Revenue")
plt.xlabel("Year")
plt.ylabel("USD")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(years, dissatisfaction, label="Dissatisfaction", color='orange')
plt.title("Average Dissatisfaction Ratio")
plt.xlabel("Year")
plt.ylabel("Ratio")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(years, glacier_melt, label="Glacier Melt Speed", color='green')
plt.title("Annual Glacier Melt Speed")
plt.xlabel("Year")
plt.ylabel("Speed")
plt.legend()

plt.tight_layout()
plt.show()

