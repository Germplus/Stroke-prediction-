import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import seaborn as sns

from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# read data
my_data=pd.read_csv('Dataset/healthcare-dataset-stroke-data.csv')

my_new_data =my_data.drop(['id'], axis=1)

print(my_new_data.info())


# sns.set(rc={"axes.facecolor":"#EAE0D5","figure.facecolor":"#EAE0D5", "grid.color":"#C6AC8F",
#             "axes.edgecolor":"#C6AC8F", "axes.labelcolor":"#0A0908", "xtick.color":"#0A0908",
#             "ytick.color":"#0A0908"})
#
# palettes = ['#9B856A', '#475962', '#598392', '#124559', '#540B0E']
#
#
# plt.figure(figsize=(8,5))
# sns.boxplot(data=my_data, x='gender', y='bmi', palette=palettes)
# plt.show()