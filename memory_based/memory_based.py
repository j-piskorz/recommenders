import pandas as pd
import numpy as np
from tqdm import trange


def cos_sim(user1, user2):  # symmetric
    if len(user1) == 0 or len(user2) == 0:
        return 0.
    user1 = set(user1)
    user2 = set(user2)

    return len(user1 & user2)/((len(user1) * len(user2))**0.5)


def cond_sim(user1, user2):  # asymmetric
    if len(user1) == 0 or len(user2) == 0:
        return 0.
    user1 = set(user1)
    user2 = set(user2)

    return len(user1 & user2)/len(user1)


def jacc_sim(user1, user2):  # symmetric
    if len(user1) == 0 or len(user2) == 0:
        return 0.
    user1 = set(user1)
    user2 = set(user2)

    return len(user1 & user2)/len(user1 | user2)


def userCF(active_user, rated, no_users, no_movies, sim, N, k):
    """Calculates top-N recommendation list using user-based CF.

    Parameters:
    -----------
        user: int
            ID of the active user
        rated: list
            list containing under rated[i] all the movie IDs of films
            rated by user i
        sim: function
            similarity function to be used
        N: int
            the desired number of recommended items
        k: int
            k-nearest neighbours to be used in the prediction calculation
    """
    users = list(range(no_users))
    movies = list(range(no_movies))

    # establishing similarity of other users to the active user
    similarity = [sim(rated[active_user], rated[u]) for u in users]
    similarity = pd.Series(similarity, index=users).drop(active_user)
    similarity = similarity.sort_values(ascending=False)[0:k]

    top_users = list(similarity.index)

    predict = pd.Series([0.]*no_movies, index=movies)

    for movie in movies:
        if movie in rated[active_user]:
            continue
        top_rated = np.array([1 if movie in rated[u] else 0
                              for u in top_users])
        predict[movie] = (np.dot(top_rated, np.array(similarity))
                          / sum(similarity))

    return predict.sort_values(ascending=False)[0:N]


def LOO_HR_userCF(sample, train, test, no_users, no_movies, sim, N, k):
    """Evaluates the recommender system using leave one out approach
    and the hit rank measure."""
    rated_train = [list(train.loc[train['userId'] == u].movieId)
                   for u in range(no_users)]
    rated_test = [int(test.loc[test['userId'] == u].movieId)
                  for u in range(no_users)]

    hits = 0

    if type(sample) != list:
        sample = list(range(no_users))

    loop = trange(len(sample), desc="Calculating hit rank")

    for i in loop:
        user = sample[i]
        predict = userCF(user, rated_train, no_users, no_movies, sim, N, k)
        predict = list(predict.index)
        if rated_test[user] in predict:
            hits += 1

    return hits/len(sample)


def sim_matrix(ratings, no_movies, sim, k):
    # all users who rated given movie:
    rated_movies = [list(ratings.loc[ratings.movieId == i].userId.unique())
                    for i in range(no_movies)]
    # number of users which watched given movie

    S = np.array([[np.float32(sim(rated_movies[i], rated_movies[j]))
                  for i in range(no_movies)] for j in range(no_movies)])
    S = pd.DataFrame(S, columns=list(range(no_movies)),
                     index=list(range(no_movies)))

    for movie in range(no_movies):
        S.loc[movie, movie] = 0.
        m = list(S[movie])
        m.sort(reverse=True)
        k_min = m[k - 1]
        S[movie] = [S.loc[movie, item] if S.loc[movie, item] >= k_min else 0.
                    for item in range(no_movies)]

    return S


def itemCF(active_user, ratings, S, N):
    """Calculates top-N recommendation list using item-based CF.

    Parameters:
    -----------
        active_user: int
            ID of the active user
        ratings: pd.DataFrame
            DataFrame containing all the ratings of all the users
        S: pd.DataFrame
            DataFrame containing under S[i, j] the similarity between
            movies with IDs i and j (only nearest neighbour
            similarities are stored and other are equal 0)
        N: int
            the desired number of recommended items
    """
    movies = S.index
    rated = list(ratings.loc[ratings.userId == active_user].movieId.unique())

    user_ratings = np.array([1. if movie in rated else 0.
                             for movie in movies])

    rec = [np.dot(user_ratings, np.array(S[movie]))/np.sum(S[movie])
           for movie in movies]
    rec = pd.Series(rec, index=movies).sort_values(ascending=False)
    rec = rec.drop(rated)[0:N]

    return rec


def LOO_HR_itemCF(sample, train, test, no_users, S, N):
    """Evaluates the recommender system using leave one out
    approach and the hit rank measure."""
    rated_test = [int(test.loc[test['userId'] == u].movieId)
                  for u in range(no_users)]

    hits = 0

    if type(sample) != list:
        sample = list(range(no_users))

    loop = trange(len(sample), desc="Calculating hit rank")

    for i in loop:
        user = sample[i]
        predict = itemCF(user, train, S, N)
        predict = list(predict.index)
        if rated_test[user] in predict:
            hits += 1

    return hits/len(sample)


def grid_search_user(k_list, sim, train, test, no_users, no_movies):
    results = []
    for k in k_list:
        if k > no_users:
            results.append(0)
        else:
            results.append(LOO_HR_userCF(
                'all', train, test,
                no_users, no_movies,
                sim, 20, k))

    best = results.index(max(results))

    return k_list[best], results[best]


def grid_search_item(k_list, sim, train, test, no_users, no_movies):
    results = []
    for k in k_list:
        if k >= no_movies:
            results.append(0)
        else:
            S = sim_matrix(train, no_movies, sim, k)
            results.append(LOO_HR_itemCF(
                'all', train, test, no_users, S, 20))

    best = results.index(max(results))

    return k_list[best], results[best]
