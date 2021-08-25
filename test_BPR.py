import pandas as pd
import numpy as np
from user_item.user_item import train_test, LOO_HR_BPR

loc = r"/Users/juliannapiskorz/OneDrive - Imperial College London/Model-" \
    r"based ML recommenders/MovieLens Data/ratings.csv"
ratings = pd.read_csv(loc)

# relable the users and movies
movies = list(ratings.movieId.astype('int').unique())
change = pd.Series(list(range(9724)), index=movies)
ratings['movieId'] = ratings['movieId'].map(change)
ratings['userId'] = np.array(ratings['userId']) - 1

# change the ratings to the unary data
ratings["rating"] = 1

# set the environment for tests
train, test = train_test(ratings, 1)

bpr_params = {
    'factors': 30,
    'regularization': 0.1,
    'learning_rate': 0.01,
    'iterations': 4000
}

res = []

for _ in range(3):
    res += [LOO_HR_BPR(list(range(610)), train,
                       test, bpr_params, 610, 9724, 20)]

print(res)
