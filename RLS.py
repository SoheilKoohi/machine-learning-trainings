
def APA(x,y,meu,delta,q):
    x_shape=x.shape
    teta=np.zeros(x_shape)
    for n in range(q-1,x_shape[0]):
        X_n=x[n-q+1:n+1,:]
        e=y[n-q+1:n+1,]- (X_n @ teta[n-1,:])[:,np.newaxis]
        teta[n]=teta[n-1,:] +np.transpose( meu * np.transpose(X_n) @ 
            np.linalg.inv(delta* np.eye(q)+ X_n @ np.transpose(X_n)) @ e)
    return teta

def NLMS(x,y,meu,delta):
    x_shape=x.shape
    teta=np.zeros(x_shape)
    for n in range(1,x_shape[0]):
        e=y[n]-teta[n-1,:] @ x[n,:]
        teta[n]= teta[n-1] + (meu /(delta+ x[n]@x[n]))*x[n]*e
    return teta
#data generation
import numpy as np
teta_act=np.random.randn(200,1)

x=np.random.randn(1000,200)
noise=np.random.normal(0,np.sqrt(.01),(1000,1))
y= x @ teta_act + noise
teta_APA=APA(x,y,.2,.001,30)
teta_APA=np.transpose(teta_APA)
teta_NLMS=np.transpose(NLMS(x,y,1.2,.001))
teta_act @ np.ones((1,1000))
mse_apa=((teta_APA-teta_act) **2).mean(axis=0)
mse_nlms=((teta_NLMS-teta_act) **2).mean(axis=0)