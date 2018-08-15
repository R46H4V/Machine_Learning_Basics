import pandas as pd 
from sklearn import neighbors, preprocessing, cross_validation
import numpy as np 

df = pd.read_csv('iris_data.csv')

#df.replace(" ",-99999, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)

example_test = np.array([4,1,6,3])
example_test = example_test.reshape(1, -1)

print (example_test)

prediction = clf.predict(example_test)

print(prediction)