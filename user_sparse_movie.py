import pandas as pd
import numpy as np
from testing_tools import train_test, sparse_movies, relabel
from memory_based import grid_search_user, cos_sim, LOO_HR_userCF

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_no_movies = [50, 100, 200, 500, 1000, 2000, 4000, 8000]
k_list = [10, 20, 30, 50, 100, 200, 500]


results = pd.Series(index=limited_no_movies)
params = []
results_detailed = []

for no_movies in limited_no_movies:
    ratings_sparse = sparse_movies(ratings, no_movies)
    train, test = train_test(ratings_sparse, 1)

    no_users = len(ratings_sparse.userId.unique())
    k, hr = grid_search_user(k_list, cos_sim, train, test, no_users, no_movies)

    res = [hr]
    params.append(k)

    for _ in range(9):
        ratings_sparse = sparse_movies(ratings, no_movies)
        train, test = train_test(ratings_sparse, 1)
        no_users = len(ratings_sparse.userId.unique())

        res.append(LOO_HR_userCF(
            'all', train, test,
            no_users, no_movies,
            cos_sim, 20, k))

    results[no_movies] = np.mean(res)
    results_detailed.append(res)

print(results)
print(params)
print(results_detailed)
