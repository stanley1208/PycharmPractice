from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X,y=make_moons(n_samples=10000,noise=0.4,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

params={
    'max_leaf_nodes':list(range(2,100)),
    'min_samples_split':[2,3,4]
}
grid_search_cv=GridSearchCV(DecisionTreeClassifier(random_state=42),params,verbose=1,cv=3)
grid_search_cv.fit(X_train,y_train)

print(grid_search_cv.best_estimator_)

y_pred=grid_search_cv.predict(X_test)
print(accuracy_score(y_test,y_pred))



