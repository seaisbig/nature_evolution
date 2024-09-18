# 这是一个基于自然哲学的遗传算法的项目
## 有12个创新点有待尝试
+ 1.多性别计算对比1
+ 2.有较高生殖竞争力的个体会获得更多的后代
+ 3.增加寿命控制函数（和适应度相关）
+ 4.冻结一部分极端个体之后到特定代数后重新加入种群
+ 5.无性繁殖，禁止自交
+ 6.禁止近亲结婚，制造天敌，对抗进化（衰减函数）1
+ 7.选择一部分精英组成一个新国家 1（引用论文（fitness分层），加入扰动种群-精英/劣质）
+ 8.*进行vae生成，通过某种方式扩张种群，以避免过拟合
+ 9.将树投射到隐空间之中拟合最可能的解
+ 10.定期投入垃圾群体进行扰动（原始人）
+ 11.适应度低的灭绝概率高
+ 12.控制生物多样性（量化）->（1.加入原始人、2.加入生成的优秀个体、3.瞬时增加变异率

## 有3个点有待解决
+ 1.将列表转移到cuda上
+ 2.修改将输出列表转化为函数的函数，将其depth拓展到>3
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
      
+ 3.检查随时会出现的None值返回
  `
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
  `
## 前置工作
+ 1.查阅论文 
+ 2.依次做消融
+ 3.编故事
