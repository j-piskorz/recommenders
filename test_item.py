from memory_based import LOO_HR_itemCF, cos_sim, sim_matrix
import pandas as pd
import numpy as np
from testing_tools import train_test

ratings = pd.read_csv("ratings.csv")

# relable the users and movies
movies = list(ratings.movieId.astype('int').unique())
change = pd.Series(list(range(9724)), index=movies)
ratings['movieId'] = ratings['movieId'].map(change)
ratings['userId'] = np.array(ratings['userId']) - 1

# change the ratings to the unary data
ratings["rating"] = 1

# set the environment for tests
train, test = train_test(ratings, 1)

S = sim_matrix(ratings, 9724, cos_sim, 30)

smpl = np.random.choice(list(range(610)), 100)
print(LOO_HR_itemCF(smpl, train, test, 610, S, 20))
