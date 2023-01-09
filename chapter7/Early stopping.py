import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

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
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=42)

gbrt=GradientBoostingRegressor(max_depth=2,n_estimators=120,random_state=42)
gbrt.fit(X_train,y_train)


errors=[mean_squared_error(y_val,y_pred) for y_pred in gbrt.staged_predict(X_val)]

best_estimators=np.argmin(errors)+1

gbrt_best=GradientBoostingRegressor(max_depth=2,n_estimators=best_estimators,random_state=42)
gbrt_best.fit(X_train,y_train)


min_error=np.min(errors)


plt.figure(figsize=(10,8))

plt.subplot(121)
plt.plot(np.arange(1,len(errors)+1),errors,"b.-")
plt.plot([best_estimators,best_estimators],[0,min_error],"k--")
plt.plot([0,120],[min_error,min_error],"k--")
plt.plot(best_estimators,min_error,"ko")
plt.text(best_estimators,min_error*1.2,"Minimum",ha="center",fontsize=16)
plt.axis([0,120,0,0.01])
plt.xlabel("Number of trees")
plt.ylabel("Error",fontsize=16,rotation=0)
plt.title("validation error",fontsize=16)

plt.subplot(122)
plot_predictions([gbrt_best],X,y,axes=[-0.5,0.5,-0.1,0.8])
plt.title("Best model (%d trees)"%best_estimators,fontsize=16)
plt.xlabel("$x_1$",fontsize=16)
plt.ylabel("$y$",fontsize=16,rotation=0)

plt.show()



