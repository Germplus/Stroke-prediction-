import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize
from plotly.offline import plot, iplot, init_notebook_mode
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import pickle
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')



# Load data set from source to dropping bmi and gender ('other').
my_data=pd.read_csv('Dataset/healthcare-dataset-stroke-data.csv')
my_new_data =my_data.drop(['id'], axis=1)
my_new_data1= my_new_data[my_new_data['gender'] != 'Other']
my_new_data2 =my_new_data1.drop(['bmi'], axis=1)

# convert age into intiger.
my_new_data2['age']=my_new_data2['age'].astype('int')


# encoding updated dataset.
enco = LabelEncoder()
gender=enco.fit_transform(my_new_data2['gender'])
smoking_status=enco.fit_transform(my_new_data2['smoking_status'])
work_type=enco.fit_transform(my_new_data2['work_type'])
Residence_type=enco.fit_transform(my_new_data2['Residence_type'])
ever_married=enco.fit_transform(my_new_data2['ever_married'])
my_new_data2['ever_married']=ever_married
my_new_data2['Residence_type']=Residence_type
my_new_data2['smoking_status']=smoking_status
my_new_data2['gender']=gender
my_new_data2['work_type']=work_type

# print(my_new_data2.head())

# split data into train and text
X = my_new_data2.drop('stroke',axis=1)
y = my_new_data2["stroke"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=11)


# Scaler
scaler = StandardScaler()

scaler.fit(X)
X_scaled = scaler.transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


# print(X.head())

# model fitting
model= DecisionTreeClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(confusion_matrix(y_test, pred, labels=model.classes_))
print(classification_report(y_test, pred))


pd.DataFrame(model.feature_importances_, index=X.columns,
             columns=['Feature Importance']).sort_values(by='Feature Importance').plot.bar()
plt.show()

# open a file, where you ant to store the data
# file = open('stroke.pkl', 'wb')

# dump information to that file
# pickle.dump(model, file)
