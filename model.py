import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
from xgboost import XGBClassifier
import joblib


warnings.filterwarnings('ignore')

# Load data set from source to dropping bmi and gender ('other').
my_data = pd.read_csv('Dataset/healthcare-dataset-stroke-data.csv')
my_new_data = my_data.drop(['id'], axis=1)
my_new_data = my_new_data[my_new_data['gender'] != 'Other']
my_new_data = my_new_data.drop(['bmi'], axis=1)

# convert age into integer.
my_new_data['age'] = my_new_data['age'].astype('int')

# encoding updated dataset.
enco = LabelEncoder()
gender = enco.fit_transform(my_new_data['gender'])
smoking_status = enco.fit_transform(my_new_data['smoking_status'])
work_type = enco.fit_transform(my_new_data['work_type'])
residence_type = enco.fit_transform(my_new_data['Residence_type'])
ever_married = enco.fit_transform(my_new_data['ever_married'])
my_new_data['ever_married'] = ever_married
my_new_data['Residence_type'] = residence_type
my_new_data['smoking_status'] = smoking_status
my_new_data['gender'] = gender
my_new_data['work_type'] = work_type

# split data into train and test
X = my_new_data.drop('stroke', axis=1)
y = my_new_data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversampling the minority class
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train.ravel())

# Fitting the model
model = XGBClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print(classification_report(y_test, y_pred))

# Saving the model
final_model = 'final_model.sav'
joblib.dump(model, final_model)

# Feature importance
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns,
                                    columns=['Feature Importance']).sort_values(by='Feature Importance')
feature_importances.plot(kind='barh', figsize=(11, 10))
plt.show()

