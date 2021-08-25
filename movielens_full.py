import pandas as pd
import numpy as np
from testing_tools import train_test, LOO_HR_BPR

loc = r"/Users/juliannapiskorz/OneDrive - Imperial College London/Model-" \
    r"based ML recommenders/MovieLens Data/ratings_full.csv"
ratings = pd.read_csv(loc)

good = ratings.groupby(['userId']).count()
good = good.loc[ratings.groupby(['userId']).count().movieId > 10].index
good = np.random.choice(good, size=10000, replace=False)
ratings = ratings.loc[ratings['userId'].isin(good)]

no_users = len(ratings.userId.astype('int').unique())
no_movies = len(ratings.movieId.astype('int').unique())

# relable the users and movies
movies = list(ratings.movieId.astype('int').unique())
change = pd.Series(list(range(no_movies)), index=movies)
ratings['movieId'] = ratings['movieId'].map(change)
users = list(ratings.userId.astype('int').unique())
change = pd.Series(list(range(no_users)), index=users)
ratings['userId'] = ratings['userId'].map(change)

# change the ratings to the unary data
ratings["rating"] = 1

# set the environment for tests
train, test = train_test(ratings, 1)

bpr_params = {
    'factors': 30,
    'regularization': 0.1,
    'learning_rate': 0.01,
    'iterations': 500
}

print(LOO_HR_BPR(list(range(no_users)), train,
                 test, bpr_params, no_users, no_movies, 20))


# obtained the hit rank = 0.1134