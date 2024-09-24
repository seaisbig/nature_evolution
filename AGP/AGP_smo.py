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
mutate_rate = 0.3 # 变异率
mutate_rate_higher = 0.6 # 变异率提升
mutate_revoluted = 0.3 #基准对比变异率
data_num = 200 # 真实数据的数量
num_generations = 100 # 迭代代数
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
CONSTANTS = [random.uniform(-10, 10) for _ in range(10)] # 随机常量

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
def mutation(expr,max_depth=max_depth,mutate_rate=mutate_rate): # 将当前节点进行变异，深度不超过max_depth
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
def to_simplified_string(prefix_expr):
    """将前缀表达式化简为最简多项式形式。"""
    def parse_expression(exp):
        x = symbols('x')
        if isinstance(exp, list):
            op = exp[0]
            if op in ('+', '-', '*', '/'):
                left = parse_expression(exp[1])
                right = parse_expression(exp[2])
                if op == '+':
                    return left + right
                elif op == '-':
                    return left - right
                elif op == '*':
                    return left * right
                elif op == '/':
                    return left / right
            elif op == 'cos':
                return cos(parse_expression(exp[1]))
            elif op == 'sin':
                return sin(parse_expression(exp[1]))
            elif op == 'exp':
                return math.exp(parse_expression(exp[1]))
        else:
            if isinstance(exp, str) and exp == 'x':
                return x
            return exp # 直接返回
    sympy_expr = parse_expression(prefix_expr)
    simplified_expr = simplify(sympy_expr)
    return sp.expand(simplified_expr)


def mutate_revolution(fitness_a, fitness_b, lenA, lenB):
    """变异率调节函数，提升弱势种群变异率。"""
    if lenA != 0 & lenB != 0:
        if lenA >= lenB:
            promoted_rate = lenA / lenB
        else:
            promoted_rate = lenB / lenA
    else:
        promoted_rate = 1

    dynamic_revolution_rate = 0

    if fitness_a >fitness_b:
        dynamic_revolution_rate = mutate_rate_higher * np.log( (fitness_a + fitness_b) / fitness_b)
    else:
        if fitness_a < fitness_b:
            dynamic_revolution_rate = mutate_rate_higher * np.log( (fitness_a + fitness_b) / fitness_a)

    mutate_revoluted = mutate_rate * promoted_rate + 0.5 * dynamic_revolution_rate

    return mutate_revoluted

# population_size:随机生成的种群数量 data:
def symbolic_regression(num_generations, population_size, data):
    # 初始化两个敌对种群
    best = []
    best_a = []
    best_b = []
    best_num = 1000
    best_num_a = 1000
    best_num_b = 1000
    population_a = [generation_random_expression() for _ in range(population_size)]
    population_b = [generation_random_expression() for _ in range(population_size)]
    #初始化两种群的变异率
    mutate_rate_a = mutate_rate
    mutate_rate_b = mutate_rate
    for generation in range(num_generations):
        # 种群全部评价适应度
        fitnesses_a = [fitness_function(ind,data) for ind in population_a]
        fitnesses_b = [fitness_function(ind,data) for ind in population_b]
        # 给出最佳适应度 
        best_fitness_a = min(fitnesses_a)
        best_fitness_b = min(fitnesses_b)
        best_fitness = min(best_fitness_b,best_fitness_a)
        if best_fitness < best_num:
            best_num = best_fitness
        best.append(best_num)
        if best_fitness_a < best_num_a:
            best_num_a = best_fitness_a
        best_a.append(best_num_a)
        if best_fitness_b < best_num_b:
            best_num_b = best_fitness_b
        best_b.append(best_num_b)
        # 最佳的树
        best_expr_a = population_a[fitnesses_a.index(best_fitness_a)]
        best_expr_b = population_b[fitnesses_b.index(best_fitness_b)]
        #定义一个函数，判断哪一个胜出
        def f(a,b):
            return 'a' if a < b else 'b'
        # 输出循环次数和当前最大适应度
        print('-'*10)
        print(f'Generation:{generation} \nBest fitness_a = {best_fitness_a} Best fitness_b = {best_fitness_b}') 
        print(f'fitness a = {sum(fitnesses_a)} fitness b = {sum(fitnesses_b)}\
            {f(sum(fitnesses_a),sum(fitnesses_b))} win!') 
        print(f'mutate_rate_a:{mutate_rate_a} mutate_rate_b:{mutate_rate_b}')
        print('-'*10)
        # 选择新的种群
        new_population_a = []
        new_population_b = []
        # a种群
        while len(new_population_a) < population_size:
            if f(sum(fitnesses_a),sum(fitnesses_b))=='b':
                mutate_rate_a = mutate_revolution(sum(fitnesses_a),sum(fitnesses_b), len(new_population_a), len(new_population_b))
            else:
                mutate_rate_a = mutate_rate
            # 选择两个个体(可能自交)
            parent_a_1 = select(population_a,fitnesses_a)
            parent_a_2 = select(population_a,fitnesses_a)
            # 交叉
            child1,child2 = crossover(parent_a_1,parent_a_2)
            # 变异
            child1 = mutation(child1,mutate_rate=mutate_rate_a)
            child2 = mutation(child2,mutate_rate=mutate_rate_a)
        
            new_population_a.append(child1)
            new_population_a.append(child2)
        population_a = new_population_a[:]
        # b种群
        while len(new_population_b) < population_size:
            # 选择两个个体
            if f(sum(fitnesses_a),sum(fitnesses_b)) == 'a':
                mutate_rate_b = mutate_revolution(sum(fitnesses_a),sum(fitnesses_b), len(new_population_a), len(new_population_b))
            else:
                mutate_rate_b = mutate_rate
            parent_b_1 = select(population_b,fitnesses_b)
            parent_b_2 = select(population_b,fitnesses_b)
            # 交叉
            child1,child2 = crossover(parent_b_1,parent_b_2)
            # 变异
            child1 = mutation(child1,mutate_rate=mutate_rate_b)
            child2 = mutation(child2,mutate_rate=mutate_rate_b)
                
            new_population_b.append(child1)
            new_population_b.append(child2)
        population_b = new_population_b[:]
    plt.title(f"最终适应度{best[-1]}")
    plt.plot(best_a,label='a')
    plt.plot(best_b,label='b')
    plt.show()
    if min(min(fitnesses_b),min(fitnesses_a)) == min(fitnesses_a):
        return best_expr_a
    else:
        return best_expr_b

# 生成示例数据 y = x^2 + 2x + 1
data = [(x, x**2 + 2*x + 1) for x in np.linspace(-10, 10, data_num)]

# 执行符号回归
best_expression = symbolic_regression(num_generations=num_generations, population_size=population_size, data=data)

# 输出最优表达式
print("Best expression found:", best_expression)
print(to_simplified_string(best_expression))