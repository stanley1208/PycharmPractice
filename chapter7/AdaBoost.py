from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundary(clf,X,y,axes=[-1.5,2.45,-1,1.5],alpha=0.5,contour=True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1,x2=np.meshgrid(x1s,x2s)
    X_new=np.c_[x1.ravel(),x2.ravel()]
    y_pred=clf.predict(X_new).reshape(x1.shape)
    custom_cmap=ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1,x2,y_pred,alpha=0.3,cmap=custom_cmap)
    if contour:
        custom_cmap2=ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contourf(x1,x2,y_pred,alpha=0.3,cmap=custom_cmap2)
    plt.plot(X[:,0][y==0],X[:,1][y==0],"yo",alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$",fontsize=18)
    plt.ylabel(r"$x_2$",fontsize=18,rotation=0)

X,y=make_moons(n_samples=1000,noise=0.30,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

ada_clf=AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),n_estimators=200,
    algorithm="SAMME.R",learning_rate=0.5,random_state=42
)

ada_clf.fit(X_train,y_train)

plot_decision_boundary(ada_clf,X,y)
plt.show()