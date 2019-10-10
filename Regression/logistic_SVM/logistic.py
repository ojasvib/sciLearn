#l2 regularized logistic regression classifier
#first two features of dataset used
#only the first two features 

#report best classification accuracy corr to var hyperparam c

from sklearn import linear_model
reg = linear_model.LogisticRegression(penalty=’l2’,C=10000)
X=np.genfromtxt('irisX.csv',delimiter=',',skip_footer=50,usecols=(0,1))
Y=np.genfromtxt('irisY.csv',delimiter=',',skip_footer=50)
reg.fit(X,Y)
reg.coef_
test_X=np.genfromtxt('irisX.csv',delimiter=',',skip_header=100,usecols=(0,1))
test_Y=np.genfromtxt('irisY.csv',delimiter=',',skip_header=100)
reg.score(test_X, test_Y)

