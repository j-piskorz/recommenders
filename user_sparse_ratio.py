import pandas as pd
import numpy as np
from testing_tools import train_test, sparse_ratio, relabel
from memory_based import grid_search_user, cos_sim, LOO_HR_userCF

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# sparse user test
limited_ratio = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1]
k_list = [10, 20, 30, 50, 100, 200, 500]


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

    k, hr = grid_search_user(k_list, cos_sim, train, test, no_users, no_movies)

    res = [hr]
    params.append(k)

    for _ in range(9):
        ratings_sparse = sparse_ratio(ratings, ratio)
        train, test = train_test(ratings_sparse, 1)
        no_movies = len(ratings_sparse.movieId.unique())
        no_users = len(ratings_sparse.userId.unique())
        numbers.append((no_users, no_movies))

        res.append(LOO_HR_userCF(
            'all', train, test,
            no_users, no_movies,
            cos_sim, 20, k))

    results[ratio] = np.mean(res)
    results_detailed.append(res)

print(results)
print(params)
print(results_detailed)
print(numbers)
