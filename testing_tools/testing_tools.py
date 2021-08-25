import numpy as np
from scipy.sparse import csr_matrix
import implicit


def create_matrix(ratings, no_users, no_movies):
    R = csr_matrix(
            (ratings['rating'], (ratings['movieId'], ratings['userId'])),
            shape=(no_movies, no_users)
            )
    R.eliminate_zeros()

    return R


def train_test(ratings, N):
    """Splits the ratings into the train set and the test set, where for each user
    exactly N ratings are delegated to the test set."""

    def fn(obj):
        return obj.loc[np.random.choice(obj.index, N, False), :]
    test = ratings.groupby('userId', as_index=False).apply(fn)
    test_set = test.index.get_level_values(1)
    train = ratings.drop(test_set)

    return train, test


def LOO_HR_BPR(sample, train, test, bpr_params, no_users, no_movies,
               N, bpr=None):
    """Evaluates the recommender system using leave one out approach
    and the hit rank measure."""
    rated_test = [int(test.loc[test['userId'] == u].movieId)
                  for u in range(no_users)]
    R_train = create_matrix(train, no_users, no_movies)

    if not bpr:
        bpr = implicit.bpr.BayesianPersonalizedRanking(**bpr_params)
        bpr.fit(R_train)

    hits = 0

    for user in sample:
        predict = bpr.recommend(user, R_train.T, N)
        predict = [x[0] for x in predict]
        if rated_test[user] in predict:
            hits += 1

    return hits/len(sample)


def grid_search(grid, train, test, no_users, no_movies, n_iters, N):
    results = []
    R_train = create_matrix(train, no_users, no_movies)

    for n in grid['factors']:
        for reg in grid['regularization']:
            for lr in grid['learning_rate']:
                bpr = implicit.bpr.BayesianPersonalizedRanking(
                    factors=n,
                    learning_rate=lr,
                    regularization=reg,
                    iterations=n_iters,
                    dtype=np.float64)
                bpr.fit(R_train)
                hr = LOO_HR_BPR(
                    list(range(no_users)),
                    train, test, [], no_users,
                    no_movies, N, bpr)
                results.append({
                    'hr': hr,
                    'factors': n,
                    'regularization': reg,
                    'learning_rate': lr})

    hr_max = [item['hr'] for item in results]
    best = hr_max.index(max(hr_max))

    return best, results
