import pandas as pd
import numpy as np
from testing_tools import train_test, sparse_users, relabel
from memory_based import grid_search_item, cos_sim, LOO_HR_itemCF
from memory_based import sim_matrix

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_no_users = [50, 75, 100, 150, 200, 300, 400, 500, 600]
k_list = [10, 20, 30, 50, 100, 200, 500]


results = pd.Series(index=limited_no_users, dtype=np.float64)
params = []
results_detailed = []

for no_users in limited_no_users:
    ratings_sparse = sparse_users(ratings, no_users)
    train, test = train_test(ratings_sparse, 1)
    no_movies = len(ratings_sparse.movieId.unique())

    k, hr = grid_search_item(k_list, cos_sim, train, test, no_users, no_movies)

    res = [hr]
    params.append(k)

    for _ in range(9):
        ratings_sparse = sparse_users(ratings, no_users)
        train, test = train_test(ratings_sparse, 1)
        no_movies = len(ratings_sparse.movieId.unique())

        S = sim_matrix(train, no_movies, cos_sim, k)
        res.append(LOO_HR_itemCF('all', train, test, no_users, S, 20))

    results[no_users] = np.mean(res)
    results_detailed.append(res)

print(results)
print(params)
print(results_detailed)
