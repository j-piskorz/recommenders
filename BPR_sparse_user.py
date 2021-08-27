import pandas as pd
import numpy as np
from testing_tools import train_test, LOO_HR_BPR, grid_search, sparse_users
from testing_tools import relabel

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_no_users = [50, 75, 100, 150, 200, 300, 400, 500, 600]

grid = {
   'factors': [20, 25, 30, 35, 40],
   'regularization': [0.001, 0.003, 0.01, 0.03, 0.1, 0.2],
   'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.03]}

results = pd.Series(index=limited_no_users)

for no_users in limited_no_users:
    ratings_sparse = sparse_users(ratings, no_users)
    train, test = train_test(ratings_sparse, 1)

    no_movies = len(ratings_sparse.movieId.unique())
    best = grid_search(grid, train, test, no_users, no_movies, 400, 20)

    res = []
    res.append(best['hr'])
    bpr_params = {
        'learning_rate': best['learning_rate'],
        'regularization': best['regularization'],
        'factors': best['factors'],
        'iterations': 400
    }

    for _ in range(9):
        train, test = train_test(ratings_sparse, 1)
        res.append(LOO_HR_BPR(
            [], train, test, bpr_params,
            no_users, no_movies, 20))

    results[no_users] = np.mean(res)

results.to_csv('./BPR_sparse.csv', index=True)
