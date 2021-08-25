import pandas as pd
import numpy as np
from user_item.user_item import train_test, grid_search

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
grid = {
   'factors': [20, 30, 40, 50],
   'regularization': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
   'learning_rate': [0.001, 0.01, 0.1, 0.3]}
print(grid_search(grid, train, test, 610, 9724, 250, 20))
