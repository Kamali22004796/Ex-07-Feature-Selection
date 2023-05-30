# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

~~~
Developed by:KAMALI.E
Registor No :212222110015
~~~.py
#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
~~~

# OUPUT

![1](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/4abbea35-340b-4647-8127-51d8df91cf5b)

![2](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/bbd0d425-d24a-4188-9fc9-12be3734628c)

![3](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/59de7ffa-2f5e-46fa-859e-2484a45f9af4)

![4](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/4f55d2c8-3d6e-4e12-a522-6deeeab2a11b)

![5](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/5cf3f9d7-6ff1-4f9e-9336-ecfee7ffd05f)

![6](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/4622ecd9-8482-40dd-9d24-ebd424b180bf)

![7](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/93c7b76f-2a98-43c6-bf92-cb13071eea4c)

![8](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/1afe34c4-f8bd-4e88-b86f-e63613d9c575)

![9](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/26a14f97-6da8-4884-89f7-c698f06c54bd)

![10](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/8d251966-4603-455e-8833-e4c86065d2fb)

![11](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/5c6b2daa-0fe4-43a1-8063-9cb0212edbe3)

![12](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/b756d0fe-b561-439d-a8b5-d74deccd8b0e)

![13](https://github.com/Kamali22004796/Ex-07-Feature-Selection/assets/120567837/bebc0b13-86c4-414a-b077-9287cfc63de9)


#RESULT 

Thus, Sucessfully performed the various feature selection techniques on a given dataset.
