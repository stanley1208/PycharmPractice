import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


iris=datasets.load_iris()
X=iris["data"][:,(2,3)] # petal length, petal width
y=iris["target"]

setosa_or_versicolor=(y==0) | (y==1)
X=X[setosa_or_versicolor]
y=y[setosa_or_versicolor]

C=5
alpha=1/(C*len(X))

lin_clf=LinearSVC(loss="hinge",C=C,random_state=42)
svm_clf=SVC(kernel="linear",C=C)
sgd_clf=SGDClassifier(loss="hinge",learning_rate="constant",eta0=0.001,alpha=alpha,max_iter=1000,tol=1e-3,random_state=42)

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

lin_clf.fit(X_scaled,y)
svm_clf.fit(X_scaled,y)
sgd_clf.fit(X_scaled,y)

print("linearSVC:",lin_clf.intercept_,lin_clf.coef_)
print("SVC:",svm_clf.intercept_,svm_clf.coef_)
print("SGDClassifier(alpha={:.5f}):".format(sgd_clf.alpha),sgd_clf.intercept_,sgd_clf.coef_)

# Compute the slope and bias of each decision boundary
w1=-lin_clf.coef_[0,0]/lin_clf.coef_[0,1]
b1=-lin_clf.intercept_[0]/lin_clf.coef_[0,1]
w2=-svm_clf.coef_[0,0]/svm_clf.coef_[0,1]
b2=-svm_clf.intercept_[0]/svm_clf.coef_[0,1]
w3=-sgd_clf.coef_[0,0]/sgd_clf.coef_[0,1]
b3=-sgd_clf.intercept_[0]/sgd_clf.coef_[0,1]


# Transform the decision boundary lines back to the original scale
line1=scaler.inverse_transform([[-10,-10*w1+b1],[10,10*w1+b1]])
line2=scaler.inverse_transform([[-10,-10*w2+b2],[10,10*w2+b2]])
line3=scaler.inverse_transform([[-10,-10*w3+b3],[10,10*w3+b3]])


# Plot all three decision boundaries
plt.figure(figsize=(10,6))
plt.plot(line1[:,0],line1[:,1],"k:",label="LinearSVC")
plt.plot(line2[:,0],line2[:,1],"b--",label="SVC")
plt.plot(line3[:,0],line3[:,1],"r-",label="SGDClassifier")
print(X[:,0][y==1])
plt.plot(X[:,0][y==1],X[:,1][y==1],"bs")    # label="Iris versicolor"
plt.plot(X[:,0][y==0],X[:,1][y==0],"yo")    # label="Iris setosa"
plt.xlabel("Patel length",fontsize=14)
plt.ylabel("Patel width",fontsize=14)
plt.legend(loc="best",fontsize=14)
plt.axis([0,5.5,0,2])
plt.show()