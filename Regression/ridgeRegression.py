#Train an l2 regularized(ridgeRegression) linear regression model on the datase, report coefficients
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[10])
X=np.genfromtxt('LR_X.csv',delimiter=',')
Y=np.genfromtxt('LR_Y.csv',delimiter=',')
reg.fit(X,Y)
reg.coef_
