from memory_based import LOO_HR_userCF, cos_sim
import pandas as pd
import numpy as np
from testing_tools import train_test, relabel

ratings = pd.read_csv("ratings.csv")
ratings = relabel(ratings)

# set the environment for tests
train, test = train_test(ratings, 1)

smpl = np.random.choice(list(range(610)), 50)
print(LOO_HR_userCF(smpl, train, test, 610, 9724, cos_sim, 20, 30))
