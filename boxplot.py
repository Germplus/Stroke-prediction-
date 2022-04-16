
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# read data from csv
my_data=pd.read_csv('Dataset/healthcare-dataset-stroke-data.csv')

# dropped 'id'column
my_new_data =my_data.drop(['id'], axis=1)

print(my_new_data.info())

sns.boxplot(data=my_new_data, x='gender', y='bmi',)
plt.show()


