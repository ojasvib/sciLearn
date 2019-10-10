#Train a linear regression model on the dataset(LR_X.csv,LR_Y.csv). Report the coefficients

from sklearn import linear_model
reg = linear_model.LinearRegression()
X=np.genfromtxt('LR_X.csv',delimiter=',')
Y=np.genfromtxt('LR_Y.csv',delimiter=',')
reg.fit(X,Y)
reg.coef_
