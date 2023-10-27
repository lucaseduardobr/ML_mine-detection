# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""





import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors
import pandas as pd




df = pd.read_csv('minas.csv', delimiter=';')


#limpando data
df.replace('?',-99999,inplace=True)
#trocando ponto por virgula
df = df.replace(',', '.', regex=True)
#vendo se as features sao verdadeiramente importantes
#df.drop(['S'], axis=1,inplace=True)
df.drop(['H'], axis=1,inplace=True)


# definindo X e Y
X = np.array(df.drop(['M'], axis=1))
y = df[['M']]
y = y.applymap(lambda x: 2 if x > 1.5 else x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
#clf.n_neighbors = 2
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy*100)

#prediction
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[10,2,1,1,1,2,3,2,1]]) #transpondo
prediction = clf.predict(example_measures)
print(prediction)



g=54,4;