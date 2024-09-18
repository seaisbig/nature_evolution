import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

POP_SIZE = 1000        # 种群大小
DNA_SIZE = 1          # 每个个体的DNA长度
CROSS_RATE = 0.7      # 交叉率
MUTATION_RATE = 0.1  # 变异率 
N_GENERATIONS = 1000   # 最大代数
X_BOUND = [0.001, 5]     # DNA每个基因的取值范围 [-5, 5]

def fitness_function(dna): # 利用此函数计算此dna在环境中的适应度
    return 1/dna[:,0]**dna[:,0]

def create_population(pop_size,dna_size): # 初始化种群 种群大小为100，种群dna长度为1
    return np.random.uniform(X_BOUND[0],X_BOUND[1],(pop_size,dna_size))

def select(population, fitness): # 按照适应度选择适应度越大的越容易生存
    idx = np.random.choice(np.arange(POP_SIZE),size=POP_SIZE,replace=True,p=fitness/fitness.sum())
    return population[idx]    #从0到POP_SIZE-1中随机选择，且选择有放回，且选择概率为fitness/fitness.sum()

def crossover(parent, pop): # pop 是一个包含多个个体的种群，用于从中随机选择一个个体与 parent 进行交叉操作
    if np.random.rand() < CROSS_RATE:
        # 随机选一个交叉点进行交叉
        i = np.random.randint(0, POP_SIZE, size=1)  # 随机选择一个个体
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)  # 随机选择交叉位置
        parent[cross_points] = pop[i, cross_points]
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = np.random.uniform(X_BOUND[0],X_BOUND[1])
    return child        
los = []
def genetic_algorithm():
    population = create_population(POP_SIZE,DNA_SIZE)
    less = 1
    for generation in range(N_GENERATIONS):
        fitness = fitness_function(population)
        best_idx = np.argmax(fitness)
        print(f'Generation:{generation},Best fitness = {fitness[best_idx]:.2f},DNA = {list(np.round(np.array(population[best_idx]),5))},loss = {np.abs(population[best_idx,0]-0.36787)/0.36787*100:3f}%')
        t = np.abs(population[best_idx, 0] - 0.36787) / 0.36787 * 100
        if t < less:
            less = t
        los.append(less)
        population = select(population,fitness)
        population_copy = population.copy()
        for parent in population:
            child = crossover(parent,population_copy)
            child = mutate(child)
            parent[:] = child #更新换代

genetic_algorithm()
avg_los = sum(los) / len(los)
plt.title(f"my solution average loss:{avg_los}best loss:{los[-1]}")
plt.plot(los)
plt.savefig('my_loss.png')