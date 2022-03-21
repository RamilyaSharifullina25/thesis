def mean_similarity(x, y):
    return (1 - abs(x - y) / (abs(x) + abs(y))).mean()

def root_mean_square_similarity(x, y):
    return ((1 - abs(x - y) / (abs(x) + abs(y))) ** 2).mean() ** 0.5

def coisine_between_angles(x, y):
    return (x * y).sum() / ((x ** 2).sum() ** 0.5 * (y ** 2).sum() ** 0.5)

def pearson_corr_funct(x, y):
    return ((x - x.mean()) * (y - y.mean())).sum() / (((x - x.mean()) ** 2.0).sum() ** 0.5 * ((y - y.mean()) ** 2.0).sum() ** 0.5)

def eucledian_distance(x, y):
    return (((x - y) ** 2).sum()) ** 0.5

# def minkowski_distance(x, y, p):
#     return (((x - y) ** p).sum()) ** (1 / p)


    
  