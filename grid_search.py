import pandas as pd
from testing_tools import train_test, grid_search, relabel

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# set the environment for tests
train, test = train_test(ratings, 1)
grid = {
   'factors': [20, 25, 30, 35, 40],
   'regularization': [0.001, 0.003, 0.01, 0.03, 0.1, 0.2],
   'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.03]}
print(grid_search(grid, train, test, 610, 9724, 250, 20))
