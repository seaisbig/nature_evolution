import random
import numpy as np
import operator
import math
from copy import deepcopy
import math
import sympy as sp
from sympy import symbols, cos, sin, sqrt, simplify
import matplotlib
matplotlib.use('TkAgg')  # 指定 TkAgg 作为后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体，确保字体路径正确，'SimHei' 是黑体的名字，适用于 Windows 系统
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#===============超参数=================#
cross_rate = 0.7 # 交叉率
mutate_rate = 0.5 # 变异率
data_num = 200 # 真实数据的数量
num_generations = 500 # 迭代代数
population_size = 100 # 单个种群数量
max_depth = 3 # 最大深度
#=====================================#

OPERATORS = {
    '+':operator.add,
    '-':operator.sub,
    '*':operator.mul,
    '/':lambda x,y:x/y if y != 0 else 1,
    'sin':math.sin,
    'cos':math.cos,
    'exp':lambda x:math.exp(x) if x < 700 else 1e10
}

TERMINALS = ['x']  # 变量
CONSTANTS = [random.uniform(-10, 10) for _ in range(5)] # 随机常量

# 生成随机表达式树
def generation_random_expression(max_depth=max_depth):
    if max_depth == 0:
        # 随机返回终端或者常量
        return random.choice(TERMINALS+CONSTANTS)
    else: # 否则随机选择一个符号
        operator_choice = random.choice(list(OPERATORS.keys()))
        if operator_choice in ['sin','cos','exp']: # 一元运算符
            return [operator_choice,generation_random_expression(max_depth-1)]
        else:
            return [operator_choice,generation_random_expression(max_depth-1),generation_random_expression(max_depth-1)]
        
def sex(tree,sex): # 定义性别
    return (tree,sex)

def evaluate_expression(expr,x_value): # 树，x的值
    if isinstance(expr,(int,float)): # 如果expr是int或者float类型【常数】
        return expr
    elif isinstance(expr,str): # 如果expr是str类型【变量】是变量就把x带入计算
        return x_value
    elif isinstance(expr,list): # 如果expr是list类型【计算符】
        op = expr[0]
        if op in ['sin','cos','exp']: #一元运算符
            return OPERATORS[op](evaluate_expression(expr[1],x_value))
        else: # 二元运算符
            return OPERATORS[op](evaluate_expression(expr[1],x_value),evaluate_expression(expr[2],x_value))


def fitness_function(expr,data): # 适应度函数expr为一个树，data为一个点，其中组成部分为x和y
    total_error = 0
    for x,y in data:
        try:
            pred =  evaluate_expression(expr,x)
            pred = np.clip(pred,-1e10,1e10)
        except (OverflowError,ZeroDivisionError):
            return float('inf') # 无穷大
        total_error += (pred-y)**2
    return total_error/len(data) # 平均损失

def crossover(expr1, expr2): 
    # 随机交换叶子节点
    if random.random() < cross_rate:
        # 检查当前节点是否是列表结构（表达式树的节点）
        if isinstance(expr1, list) and isinstance(expr2, list):
            # 随机选择交叉点，可能在左子树或者右子树
            if len(expr1) > 1 and len(expr2) > 1:
                crossover_point1 = random.randint(1, len(expr1) - 1)
                crossover_point2 = random.randint(1, len(expr2) - 1)
                # 递归处理交叉点
                new_expr1 = deepcopy(expr1)
                new_expr2 = deepcopy(expr2)

                # 递归交叉子树
                new_expr1[crossover_point1], new_expr2[crossover_point2] = crossover(new_expr1[crossover_point1], new_expr2[crossover_point2]) 
                return new_expr1, new_expr2
                # 如果遇到叶子节点或者其中一个不是列表，直接交换这两个叶子节点
        return deepcopy(expr2), deepcopy(expr1)
    # 如果不交叉，直接返回原始表达式
    return deepcopy(expr1), deepcopy(expr2)
def mutation(expr,max_depth=max_depth): # 将当前节点进行变异，深度不超过max_depth
    if random.random() < mutate_rate:
        return generation_random_expression(max_depth)
    return expr

# 轮盘赌选择
def select(population,fitnesses): # population为许多个树，fitnesses为每个树的适应度
    inverse_fitnesses = [1/(f+1e-10) for f in fitnesses] # 防止除0错误
    total_fitness = sum(inverse_fitnesses)
    pick = random.uniform(0,total_fitness) # 阈值
    current = 0
    for i,fitness in enumerate(inverse_fitnesses): # 对每棵树进行操作
        current += fitness
        if current > pick: # 当叠加的损失超过了随机值后返回（随机选择深度） 越大越容易超过阈值
            return deepcopy(population[i]) # 深层copy
def parse_expression(exp):
    x = symbols('x')
    if isinstance(exp, list):
        if len(exp) == 0:  # 防止空列表的错误
            return None
        op = exp[0]
        if op in ('+', '-', '*', '/'):
            left = parse_expression(exp[1])  # 递归处理左子树
            right = parse_expression(exp[2])  # 递归处理右子树
            if left is None or right is None:  # 检查是否解析失败
                return None
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / (right + 1e-10)  # 防止除以零
        elif op == 'cos':
            return cos(parse_expression(exp[1]))
        elif op == 'sin':
            return sin(parse_expression(exp[1]))
    else:
        if isinstance(exp, str) and exp == 'x':  # 处理变量 x
            return x
        return exp  # 处理常数

# 化简表达式并返回最简形式
def to_simplified_string(prefix_expr):
    """将前缀表达式化简为最简多项式形式。"""
    sympy_expr = parse_expression(prefix_expr)  # 解析表达式
    if sympy_expr is None:
        return "Invalid expression"
    
    simplified_expr = simplify(sympy_expr)  # 化简
    return sp.expand(simplified_expr)  # 返回最简形式

# population_size:随机生成的种群数量 data:
def symbolic_regression(num_generations, population_size, data):
    # 初始化男女种群
    best = []
    best_num = 1000
    population_male = [generation_random_expression() for _ in range(population_size)]
    population_female = [generation_random_expression() for _ in range(population_size)]
    for generation in range(num_generations):
        # 种群全部评价适应度
        fitnesses_male = [fitness_function(ind,data) for ind in population_male]
        fitnesses_female = [fitness_function(ind,data) for ind in population_female]
        # 给出最佳适应度 
        best_fitness_male = min(fitnesses_male)
        best_fitness_female = min(fitnesses_female)
        best_fitness = min(best_fitness_female,best_fitness_male)
        if best_fitness < best_num:
            best_num = best_fitness
        best.append(best_num)
        # 最佳的树
        best_expr_male = population_male[fitnesses_male.index(best_fitness_male)]
        best_expr_female = population_female[fitnesses_female.index(best_fitness_female)]
        # 输出循环次数和当前最大适应度
        print(f'Generation:{generation} \nBest fitness_male = {best_fitness_male} Best fitness_female = {best_fitness_female}') 
        # 选择新的种群
        new_population_male = []
        new_population_female = []
        # 男性
        while len(new_population_male) < population_size:
            # 选择两个个体
            parent_male = select(population_male,fitnesses_male)
            parent_female = select(population_female,fitnesses_female)
            # 交叉
            child1,child2 = crossover(parent_male,parent_female)
            # 变异
            child1 = mutation(child1)
            child2 = mutation(child2)
                
            new_population_male.append(child1)
            new_population_male.append(child2)
        population_male = new_population_male[:]
        # 女性
        while len(new_population_female) < population_size:
            # 选择两个个体
            parent_male = select(population_male,fitnesses_male)
            parent_female = select(population_female,fitnesses_female)
            # 交叉
            child1,child2 = crossover(parent_male,parent_female)
            # 变异
            child1 = mutation(child1)
            child2 = mutation(child2)
                
            new_population_female.append(child1)
            new_population_female.append(child2)
        population_female = new_population_female[:]
    plt.title(f"设定性别种群的进化速度\n最终适应度{best[-1]}")
    plt.plot(best)
    plt.show()
    if min(min(fitnesses_female),min(fitnesses_male)) == min(fitnesses_male):
        return best_expr_male
    else:
        return best_expr_female

# 生成示例数据 y = x^2 + 2x + 1
data = [(x, x**2 + 2*x + 1) for x in np.linspace(-10, 10, data_num)]

# 执行符号回归
best_expression = symbolic_regression(num_generations=num_generations, population_size=population_size, data=data)

# 输出最优表达式
print("Best expression found:", best_expression)
print(to_simplified_string(best_expression))