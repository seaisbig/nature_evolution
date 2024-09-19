import math
import sympy as sp
from sympy import symbols, cos, sin, sqrt, simplify

def evaluate_expression(expr):
    if isinstance(expr, list):
        op = expr[0]
        if op == '+':
            return evaluate_expression(expr[1]) + evaluate_expression(expr[2])
        elif op == '-':
            return evaluate_expression(expr[1]) - evaluate_expression(expr[2])
        elif op == '*':
            return evaluate_expression(expr[1]) * evaluate_expression(expr[2])
        elif op == '/':
            return evaluate_expression(expr[1]) / evaluate_expression(expr[2])
        elif op == 'cos':
            return math.cos(evaluate_expression(expr[1]))
        elif op == 'sin':
            return math.sin(evaluate_expression(expr[1]))
        elif op == 'sqrt':
            return math.sqrt(evaluate_expression(expr[1]))
    else:
        return expr

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
        elif op == 'sqrt':
            return sqrt(parse_expression(exp[1]))
    else:
        if isinstance(exp, str) and exp == 'x':
            return x
        return exp

def to_simplified_string(prefix_expr):
    """将前缀表达式化简为最简多项式形式。"""
    # 解析表达式
    sympy_expr = parse_expression(prefix_expr)
    # 化简表达式
    simplified_expr = simplify(sympy_expr)
    # 返回化简后的多项式形式
    return sp.expand(simplified_expr)

# 输入的前缀表达式
expression = ['*', 
              ['+', 
                ['-', 'x', 0.7189788277820988], 
                ['/', -5.190273302634612, -2.754374893989551]
              ], 
              ['/', 
                ['+', 0.7189788277820988, 'x'], 
                ['cos', -6.0436052310199955]
              ]
]

# 解析和输出化简后的多项式形式
polynomial = to_simplified_string(expression)
print("化简后的多项式形式：", polynomial)