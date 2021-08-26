import pandas as pd
import numpy as np
from testing_tools import train_test, LOO_HR_BPR

loc = r"/Users/juliannapiskorz/OneDrive - Imperial College London/Model-" \
    r"based ML recommenders/MovieLens Data/ratings.csv"
ratings = pd.read_csv(loc)

