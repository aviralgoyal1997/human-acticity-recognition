import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
df1=pd.read_csv('test.csv')
sd=df1['Activity']
train=df
target=train.Activity
train.drop('Activity',1,inplace=True)
df1.drop('Activity',1,inplace=True)
combined=train.append(df1)
combined.reset_index(inplace=True)
combined.drop('index',inplace=True,axis=1)
    


    
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf=RandomForestClassifier()
clf.fit(train,target)
model=SelectFromModel(clf,prefit=True)
train_reduced=model.transform(train)
test_reduced=model.transform(df1)
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

model=RandomForestClassifier(**parameters)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_reduced, target, test_size=0.2)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

print(model.score(test_reduced,sd))
