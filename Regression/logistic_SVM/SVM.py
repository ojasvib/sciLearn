#SVM classifier
#first two features for your model
#RBF-kernel, gamma=0.5, one-vs-rest classifier, no-feature-normalization.

#report best classification accuracy
from sklearn import svm
reg=svm.SVC(C=0.01,kernel='rbf',gamma=0.5)
X=np.genfromtxt('irisX.csv',delimiter=',',skip_footer=50,usecols=(0,1))
Y=np.genfromtxt('irisY.csv',delimiter=',',skip_footer=50)
reg.fit(X,Y)
test_X=np.genfromtxt('irisX.csv',delimiter=',',skip_header=100,usecols=(0,1))
test_Y=np.genfromtxt('irisY.csv',delimiter=',',skip_header=100)
reg.score(test_X, test_Y)
len(reg.support_vectors_)
