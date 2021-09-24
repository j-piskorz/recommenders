import pandas as pd
import numpy as np
from testing_tools import train_test, LOO_HR_BPR, grid_search, sparse_movies
from testing_tools import relabel

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_no_movies = [50, 100, 200, 500, 1000, 2000, 4000, 8000]

grid = {
   'factors': [25, 30, 35, 40],
   'regularization': [0.0003, 0.001, 0.003, 0.01, 0.03],
   'learning_rate': [0.001, 0.003, 0.01]}

results = pd.Series(index=limited_no_movies)
results_detailed = []
params = []

for no_movies in limited_no_movies:
    ratings_sparse = sparse_movies(ratings, no_movies)
    train, test = train_test(ratings_sparse, 1)
    no_users = len(ratings_sparse.userId.unique())

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
        ratings_sparse = sparse_movies(ratings, no_movies)
        no_users = len(ratings_sparse.userId.unique())
        train, test = train_test(ratings_sparse, 1)
        res.append(LOO_HR_BPR(
            'all', train, test, bpr_params,
            no_users, no_movies, 20))

    results[no_movies] = np.mean(res)
    results_detailed.append(res)

print(results)
print(params)
print(results_detailed)
