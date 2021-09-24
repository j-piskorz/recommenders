import pandas as pd
import numpy as np
from testing_tools import train_test, LOO_HR_BPR, grid_search, sparse_ratio
from testing_tools import relabel

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_ratio = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1]

grid = {
   'factors': [20, 25, 30, 35],
   'regularization': [0.001, 0.003, 0.01, 0.03, 0.1],
   'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.03]}

results = pd.Series(index=limited_ratio)
params = []
results_detailed = []
numbers = []

for ratio in limited_ratio:
    ratings_sparse = sparse_ratio(ratings, ratio)
    train, test = train_test(ratings_sparse, 1)

    no_movies = len(ratings_sparse.movieId.unique())
    no_users = len(ratings_sparse.userId.unique())
    numbers.append((no_users, no_movies))
    best = grid_search(grid, train, test, no_users, no_movies, 400, 20)

    res = []
    res.append(best['hr'])
    bpr_params = {
        'learning_rate': best['learning_rate'],
        'regularization': best['regularization'],
        'factors': best['factors'],
        'iterations': 400
    }
    params.append(bpr_params)

    for _ in range(9):
        ratings_sparse = sparse_ratio(ratings, ratio)
        train, test = train_test(ratings_sparse, 1)
        no_movies = len(ratings_sparse.movieId.unique())
        no_users = len(ratings_sparse.userId.unique())
        numbers.append((no_users, no_movies))

        res.append(LOO_HR_BPR(
            'all', train, test, bpr_params,
            no_users, no_movies, 20))

    results[ratio] = np.mean(res)
    results_detailed.append(res)

print(results)
print(params)
print(results_detailed)
print(numbers)
