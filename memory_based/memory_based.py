import pandas as pd
import numpy as np
from tqdm import trange


def cos_sim(user1, user2):  # symmetric
    user1 = set(user1)
    user2 = set(user2)

    return len(user1 & user2)/((len(user1) * len(user2))**0.5)


def cond_sim(user1, user2):  # asymmetric
    user1 = set(user1)
    user2 = set(user2)

    return len(user1 & user2)/len(user1)


def jacc_sim(user1, user2):  # symmetric
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

    if sample == []:
        sample = list(range(no_users))

    loop = trange(len(sample), desc="Calculating hit rank")

    for i in loop:
        user = sample[i]
        predict = userCF(user, rated_train, no_users, no_movies, sim, N, k)
        predict = list(predict.index)
        if rated_test[user] in predict:
            hits += 1

    return hits/len(sample)


def sim_matrix(ratings, no_movies, sim, top, k):
    # all users who rated given movie:
    rated_movies = [list(ratings.loc[ratings.movieId == i].userId.unique())
                    for i in range(no_movies)]
    # number of users which watched given movie
    top_rated = ratings.groupby(['movieId']).count().userId
    top_rated = top_rated.sort_values(ascending=False).index[0:top]

    rated_movies_top = {item: rated_movies[item] for item in top_rated}

    S = np.array([[sim(rated_movies_top[i], rated_movies_top[j])
                  for i in top_rated] for j in top_rated])
    S = pd.DataFrame(S, columns=top_rated, index=top_rated)

    for movie in top_rated:
        S.loc[movie, movie] = 0.
        m = list(S[movie])
        m.sort(reverse=True)
        k_min = m[k]
        S[movie] = [S.loc[movie, item] if S.loc[movie, item] >= k_min else 0.
                    for item in top_rated]

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
    top_rated = list(S.index)
    rated = list(ratings.loc[ratings.userId == active_user].movieId.unique())
    rated_from_top = list(set(rated) & set(top_rated))

    user_ratings = np.array([1. if movie in rated else 0.
                             for movie in top_rated])

    rec = [np.dot(user_ratings, np.array(S[movie]))/np.sum(S[movie])
           for movie in top_rated]
    rec = pd.Series(rec, index=top_rated).sort_values(ascending=False)
    rec = rec.drop(rated_from_top)[0:N]

    return rec


def LOO_HR_itemCF(sample, train, test, no_users, S, N):
    """Evaluates the recommender system using leave one out
    approach and the hit rank measure."""
    rated_test = [int(test.loc[test['userId'] == u].movieId)
                  for u in range(no_users)]

    hits = 0

    if sample == []:
        sample = list(range(no_users))

    loop = trange(len(sample), desc="Calculating hit rank")

    for i in loop:
        user = sample[i]
        predict = itemCF(user, train, S, N)
        predict = list(predict.index)
        if rated_test[user] in predict:
            hits += 1

    return hits/len(sample)
