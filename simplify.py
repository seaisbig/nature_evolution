import random
import numpy as np
import operator
import math
from copy import deepcopy
import math
import sympy as sp
from sympy import symbols, cos, sin, sqrt, simplify,exp
import matplotlib
matplotlib.use('TkAgg')  # 指定 TkAgg 作为后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
                return exp(parse_expression(exp[1]))
        else:
            if isinstance(exp, str) and exp == 'x':
                return x
            return exp # 直接返回
    sympy_expr = parse_expression(prefix_expr)
    simplified_expr = simplify(sympy_expr)
    return sp.expand(simplified_expr)
