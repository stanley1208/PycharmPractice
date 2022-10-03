import numpy as np
from sklearn import datasets
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X1D=np.linspace(-4,4,9).reshape(-1,1)

def gaussian_rbf(x,landmark,gamma):
    return np.exp(-gamma*np.linalg.norm(x-landmark,axis=1)**2)

gamma=0.3

x1s=np.linspace(-4.5,4.5,200).reshape(-1,1)
x2s=gaussian_rbf(x1s,-2,gamma)
x3s=gaussian_rbf(x1s,1,gamma)

XK=np.c_[gaussian_rbf(X1D,-2,gamma),gaussian_rbf(X1D,1,gamma)]
yk=np.array([0,0,1,1,1,1,1,0,0])

plt.figure(figsize=(10,6))

plt.subplot(121)
plt.grid(True,which='both')
plt.axhline(y=0,color='k')
plt.scatter(x=[-2,1],y=[0,0],s=150,alpha=0.5,c="red")
plt.plot(X1D[:,0][yk==0],np.zeros(4),"bs")
plt.plot(X1D[:,0][yk==1],np.zeros(5),"g^")
plt.plot(x1s,x2s,"g--")
plt.plot(x1s,x3s,"b:")
plt.gca().get_yaxis().set_ticks([0,0.25,0.5,0.75,1])
plt.xlabel(r"$x_1$",fontsize=20)
plt.ylabel(r"Similarity",fontsize=14)
plt.annotate(r'$\mathbf{x}$',
             xy=(X1D[3,0],0),
             xytext=(-0.5,0.20),
             ha="center",
             arrowprops=dict(facecolor='black',shrink=0.1),
             fontsize=18,
             )
plt.text(-2,0.9,"$x_2$",ha="center",fontsize=20)
plt.text(1,0.9,"$x_3$",ha="center",fontsize=20)
plt.axis([-4.5,4.5,-0.1,1.1])


plt.subplot(122)
plt.grid(True,which="both")
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
plt.plot(XK[:,0][yk==0],XK[:,1][yk==0],'bs')
plt.plot(XK[:,0][yk==1],XK[:,1][yk==1],'g^')
plt.xlabel(r"$x_2$",fontsize=20)
plt.ylabel(r"$x_3$",fontsize=20,rotation=0)
plt.annotate(r'$\phi\left(\mathbf{x}\right)$',
             xy=(XK[3,0],XK[3,1]),
             xytext=(0.65,0.50),
             ha="center",
             arrowprops=dict(facecolor='black',shrink=0.1),
             fontsize=18,
             )
plt.plot([-0.1,1.1],[0.57,-0.1],"r--",linewidth=3)
plt.axis([-0.1,1.1,-0.1,1.1])
# plt.subplots_adjust(right=1)

plt.show()


