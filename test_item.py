from testing_tools import create_matrix
import pandas as pd
import numpy as np
from implicit.nearest_neighbours import CosineRecommender
from testing_tools import train_test, LOO_HR_BPR

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

R_train = create_matrix(train, 610, 9724)
rec = CosineRecommender(K=30)
rec.fit(R_train)

print(LOO_HR_BPR(list(range(610)), train, test, [], 610, 9724, 20, rec))
