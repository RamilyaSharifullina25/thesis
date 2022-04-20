import torch
from torch import nn
import tsfresh

# выпишем все функции библиотеки которые нам нужны
listOfFunctions = [f for f in tsfresh.feature_extraction.feature_calculators.__dict__.values()
                   if hasattr(f,'__call__')]

listOfFunctions = listOfFunctions[21:]

# в needed_actions запишем только те функции, которые принимают на вход один позиционный оргумент
needed_actions = []
for i in range(len(listOfFunctions)):
    if listOfFunctions[i].__code__.co_argcount == 1:
        needed_actions.append(listOfFunctions[i])

needed_actions = needed_actions[4:] # первые 4 функции возвращают только True/False поэтому их не рассматриваем


labels_diff = [0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 32]
differentiable_needed_actions = [needed_actions[i] for i in labels_diff]


# def mean_similarity(x, y):
#     return (1 - abs(x - y) / (abs(x) + abs(y))).mean()

# def root_mean_square_similarity(x, y):
#     return ((1 - abs(x - y) / (abs(x) + abs(y))) ** 2).mean() ** 0.5

# def coisine_between_angles(x, y):
#     return (x * y).sum() / ((x ** 2).sum() ** 0.5 * (y ** 2).sum() ** 0.5)

# def pearson_corr_funct(x, y):
#     return ((x - x.mean()) * (y - y.mean())).sum() / (((x - x.mean()) ** 2.0).sum() ** 0.5 * ((y - y.mean()) ** 2.0).sum() ** 0.5)

# def eucledian_distance(x, y):
#     return (((x - y) ** 2).sum()) ** 0.5

# def minkowski_distance(x, y, p):
#     return (((x - y) ** p).sum()) ** (1 / p)

  