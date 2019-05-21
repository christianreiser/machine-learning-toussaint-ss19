import numpy as np
import csv
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

df = pandas.read_csv('data2Class.txt',delim_whitespace=True)
x1 = df.loc[:,'x1']
x2 = df.loc[:,'x2']
y =  np.asarray(df.loc[:,'y' ])
X =  np.asarray(df.loc[:,'x1':'x2'])
"""
L = np.zeros(5)
k = 10; #number of partions
lambda_vec = np.array([0.001,0.01,0.1,1,10,20,100,1000])
beta = np.zeros((len(df),3))


i=0
while i <= len(lambda_vec):
    beta = beta - np.linalg.multi_dot([np.transpose(X),W,X]) + 2*lambda_vec[i]*I
"""

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)
print(clf.predict(X[:2, :]))

print(clf.predict_proba(X[:2, :]))


print(clf.score(X, y))



#print((X))
"""
plot = plt.figure().gca(projection='3d')
plot.scatter(x1, x2, y)
plot.set_xlabel('x1')
plot.set_ylabel('x2')
plot.set_zlabel('y')
plt.show()
"""
