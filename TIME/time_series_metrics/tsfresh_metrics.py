import torch
from torch import nn
import tsfresh

# выпишем все функции библиотеки которые нам нужны
listOfFunctions = [f for f in tsfresh.feature_extraction.feature_calculators.__dict__.values()
                   if hasattr(f,'__call__')]

listOfFunctions = listOfFunctions[21:]

# в needed_actions запишем только те функции, которые принимают на вход один позиционный аргумент
needed_actions = []
for i in range(len(listOfFunctions)):
    if listOfFunctions[i].__code__.co_argcount == 1:
        needed_actions.append(listOfFunctions[i])

needed_actions = needed_actions[4:] # первые 4 функции возвращают только True/False поэтому их не рассматриваем


labels_diff = [0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 32]
tsfresh_metrics = [needed_actions[i] for i in labels_diff]
  