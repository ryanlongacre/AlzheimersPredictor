import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
data_file = pd.read_csv("../data/alzheimers_disease_data.csv")

#get rid of patient Id and doctor who worked w them
data_file = data_file.drop(columns=['PatientID', 'DoctorInCharge'])

#drops data sets that don't have a diagnosis
data_file.dropna(subset=['Diagnosis'], inplace=True)

print(data_file.info)

numerical_cols = [col for col in data_file.columns if col != 'Diagnosis']

#doesn't inlcude all of the 1s and 0s data in the standardization
#ex: won't standardize someone's history of alzheimer's
binary_cols = [col for col in numerical_cols if data_file[col].nunique() == 2]

categorical_cols = ['Ethnicity', 'EducationLevel']

#filter out the binary cols
cols_to_standardize = [col for col in numerical_cols if col not in binary_cols and col not in categorical_cols]

#converts all of the data in the list so it has a mean of 0 and a standard deviation of 1
#standardizes it
scaler = StandardScaler()
data_file[cols_to_standardize] = scaler.fit_transform(data_file[cols_to_standardize])


print(data_file.info)

print(data_file.columns)

#x is the input data for the model, which in this case is every column except for Diagnosis
#y is the output for the model, the diagnosis column that we are trying to predict
x = data_file.drop(columns = ['Diagnosis'])
y = data_file['Diagnosis']

#creates the random forest model, can specify number of decision trees
model = RandomForestClassifier(n_estimators = 300)

#trains the random forest model to learn patters in the data
#will learn how x is related to y
model.fit(x,y)

#mode.feature_importances_ contains the importance of each feature
#this is an arrya of values that sums to 1
#pd.Series creates a panda series (like an array) with indices of the colums
feature_importance = pd.Series(model.feature_importances_, index=x.columns)
feature_importance.sort_values(ascending=False).plot(kind="bar")
plt.xticks(rotation=45, ha="right")
plt.gca().set_xticks(range(0, len(feature_importance), 2))
plt.tight_layout()

plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)

data_file['RiskGroup'] = kmeans.fit_predict(data_file.drop(columns=['Diagnosis']))
