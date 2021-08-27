import pandas as pd
from testing_tools import train_test, LOO_HR_BPR, relabel

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# set the environment for tests
train, test = train_test(ratings, 1)

bpr_params = {
    'factors': 30,
    'regularization': 0.1,
    'learning_rate': 0.01
}

res = []

for n in [200, 250, 400, 600, 1000]:
    bpr_params = {
        'factors': 30,
        'regularization': 0.1,
        'learning_rate': 0.01,
        'iterations': n
    }
    res += [LOO_HR_BPR([], train,
                       test, bpr_params, 610, 9724, 20)]

print(res)
