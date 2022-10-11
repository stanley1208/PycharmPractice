import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


iris=load_iris()
X=iris.data[:,2:] # petal length and width
y=iris.target
# print(X)
tree_clf=DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X,y)

export_graphviz(
    tree_clf,
    out_file=os.path.join(IMAGES_PATH,"iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH,"iris_tree.dot"))

def plot_decision_boundary(clf,X,y,axes=[0,7.5,0,3],iris=True,legend=True,plot_training=True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s=np.linspace(axes[2],axes[3],100)
    x1,x2=np.meshgrid(x1s,x2s)
    X_new=np.c_[x1.ravel(),x2.ravel()]
    y_pred=clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1,x2,y_pred,alpha=0.5,cmap=custom_cmap)
    if not iris:
        custom_cmap2=ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1,x2,y_pred,alpha=0.3,cmap=custom_cmap2)
    if plot_training:
        plt.plot(X[:,0][y==0],X[:,1][y==0],"yo",label="Iris setosa")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris virginica")
    if iris:
        plt.xlabel("Petal length",fontsize=18)
        plt.ylabel("Petal width",fontsize=18)
    else:
        plt.xlabel(r"$x_1", fontsize=18)
        plt.ylabel(r"$x_2", fontsize=18,rotation=0)
    if legend:
        plt.legend(loc="lower right",fontsize=14)


plt.figure(figsize=(10,6))
plot_decision_boundary(tree_clf,X,y)
plt.plot([2.45,2.45],[0,3],"k-",linewidth=2)
plt.plot([2.45,7.5],[1.75,1.75],"k--",linewidth=2)
plt.plot([4.95,4.95],[0,1.75],"k:",linewidth=2)
plt.plot([4.85,4.85],[1.75,3],"k:",linewidth=2)
plt.text(1.40,1.0,"Depth=0",fontsize=15)
plt.text(3.2,1.80,"Depth=1",fontsize=13)
plt.text(4.05,0.5,"Depth=2",fontsize=11)
plt.show()


print(tree_clf.predict_proba([[5,1.5]]))
print(tree_clf.predict([[5,1.5]]))