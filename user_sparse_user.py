import pandas as pd
import numpy as np
from testing_tools import train_test, sparse_users, relabel
from memory_based import grid_search_user, cos_sim, LOO_HR_userCF

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_no_users = [50, 75, 100, 150, 200, 300, 400, 500, 600]
k_list = [10, 20, 30, 50, 100, 200, 500]


results = pd.Series(index=limited_no_users)
params = []
results_detailed = []

for no_users in limited_no_users:
    ratings_sparse = sparse_users(ratings, no_users)
    train, test = train_test(ratings_sparse, 1)

    no_movies = len(ratings_sparse.movieId.unique())
    k, hr = grid_search_user(k_list, cos_sim, train, test, no_users, no_movies)

    res = [hr]
    params.append(k)

    for _ in range(9):
        ratings_sparse = sparse_users(ratings, no_users)
        train, test = train_test(ratings_sparse, 1)
        no_movies = len(ratings_sparse.movieId.unique())

        res.append(LOO_HR_userCF(
            'all', train, test,
            no_users, no_movies,
            cos_sim, 20, k))

    results[no_users] = np.mean(res)
    results_detailed.append(res)

print(results)
print(params)
print(results_detailed)
