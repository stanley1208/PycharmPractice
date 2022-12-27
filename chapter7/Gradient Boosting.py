from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_predictions(regressors,X,y,axes,label=None,style="r-",data_style="b.",data_label=None):
    x1=np.linspace(axes[0],axes[1],500)
    y_pred=sum(regressor.predict(x1.reshape(-1,1)) for regressor in regressors)
    plt.plot(X[:,0],y,data_style,label=data_label)
    plt.plot(x1,y_pred,style,linewidth=2,label=label)
    if label or data_label:
        plt.legend(loc="upper center",fontsize=16)
    plt.axis(axes)


np.random.seed(42)
X=np.random.rand(100,1)-0.5
y=3*X[:,0]**2+0.05*np.random.rand(100)

tree_reg1=DecisionTreeRegressor(max_depth=2,random_state=42)
tree_reg1.fit(X,y)

y2=y-tree_reg1.predict(X)
tree_reg2=DecisionTreeRegressor(max_depth=2,random_state=42)
tree_reg2.fit(X,y2)

y3=y2-tree_reg2.predict(X)
tree_reg3=DecisionTreeRegressor(max_depth=2,random_state=42)
tree_reg3.fit(X,y3)

X_new=np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)

plt.figure(figsize=(10,10))

plt.subplot(321)
plot_predictions([tree_reg1],X,y,axes=[-0.5,0.5,-0.1,0.8],label="$h_1(x_1)$",style="g-",data_label="Training set")
plt.ylabel("$y$",fontsize=16,rotation=0)
plt.title("Residuals and tree predictions",fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1],X,y,axes=[-0.5,0.5,-0.1,0.8],label="$h_1(x_1)=h_1(x_1)$",data_label="Training set")
plt.ylabel("$y$",fontsize=16,rotation=0)
plt.title("Ensemble predictions",fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2],X,y2,axes=[-0.5,0.5,-0.1,0.8],label="$h_2(x_1)$",style="g-",data_style="k+",data_label="Residuals")
plt.ylabel("$y-h_1(x_1)$",fontsize=16)


plt.subplot(324)
plot_predictions([tree_reg1,tree_reg2],X,y,axes=[-0.5,0.5,-0.1,0.8],label="$h(x_1)=h_1(x_1)+h_2(x_1)$")
plt.ylabel("$y$",fontsize=16,rotation=0)


plt.subplot(325)
plot_predictions([tree_reg3],X,y3,axes=[-0.5,0.5,-0.1,0.8],label="$h_3(x_1)$",style="g-",data_style="k+")
plt.ylabel("$y-h_1(x_1)-h_2(x_1)$",fontsize=16)
plt.xlabel("$x_1$",fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1,tree_reg2,tree_reg3],X,y,axes=[-0.5,0.5,-0.1,0.8],label="$h(x_1)=h_1(x_1)+h_2(x_1)+h_3(x_1)$")
plt.xlabel("$x_1$",fontsize=16)
plt.ylabel("$y$",fontsize=16,rotation=0)

plt.show()







