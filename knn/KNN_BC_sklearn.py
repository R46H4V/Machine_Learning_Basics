import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd 

df = pd.read_csv('BCancer.data')
# we have some missing data replaced by '?' in the dataset
df.replace('?', -99999, inplace=True) # -99999 is to keep the data as a large outlier

#no use of id column in the prediction of the tumor
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'] , 1)) # features
y = np.array(df['class']) # lables

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2) # split the data into testing and training

clsf = neighbors.KNeighborsClassifier()
clsf.fit (x_train, y_train)

#test
accuracy = clsf.score(x_test, y_test)
print(accuracy)

# make an example to predict
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,1,2,3,3,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clsf.predict(example_measures)
print(prediction)