#%%
#estimation of a linear model using APA algorithm.q is the number of time shift
def APA(x,y,meu,delta,q):
    x_shape=x.shape
    e=np.zeros(x.shape[0])
    teta=np.zeros(x_shape)
    for n in range(q-1,x_shape[0]):
        X_n=x[n-q+1:n+1,:]
        E=y[n-q+1:n+1,]- (X_n @ teta[n-1,:])[:,np.newaxis]
        teta[n]=teta[n-1,:] +np.transpose( meu * np.transpose(X_n) @ 
            np.linalg.inv(delta* np.eye(q)+ X_n @ np.transpose(X_n)) @ E)
        ee=y[n]-teta[n-1,:] @ x[n,:]
        e[n]=ee**2
    return teta,e
#%%
#estimation of a linear model using APA algorithm.
def NLMS(x,y,meu,delta):
    x_shape=x.shape
    teta=np.zeros(x_shape)
    e=np.zeros(x.shape[0])
    for n in range(1,x_shape[0]):
        ee=y[n]-teta[n-1,:] @ x[n,:]
        teta[n]= teta[n-1] + (meu /(delta+ x[n]@x[n]))*x[n]*ee
        e[n]=ee**2
    return teta,e
#%%
#estimation of a linear model using RLS algorithm. beta is forgetting factor
def RLS(x,y,beta,gamma):
    teta=np.zeros(x.shape)
    P= (1/gamma) * np.eye(x.shape[1])
    e=np.zeros(x.shape[0])
    for n in range(x.shape[0]):
        ee= y[n]-teta[n-1] @ x[n]
        z=P @ x[n]
        k= (1/(beta + x[n] @ z)) * z
      
        teta[n]= teta[n-1] + k * ee
        P=(1/beta) * ( P - k[:,np.newaxis] @ np.transpose(z[:,np.newaxis]))
        e[n]=ee**2
    return teta,e
#%%
#data generation
#here we generate 100 different experiment and implement different algorithms
#then average the MSE of the errors y[n] - teta[n-1] * x[n]
import numpy as np
N=3500
l=200
ex=100
teta_act=np.random.randn(l,1)
E_APA=np.zeros((N,ex))
E_NLMS=np.zeros((N,ex))
E_RLS=np.zeros((N,ex))
for ex_num in range(ex):
    print(ex_num)
    x=np.random.randn(N,l)
    noise=np.random.normal(0,np.sqrt(.01),(N,1))
    yy= x @ teta_act
    y=yy+noise
    _,E_APA[:,ex_num]=APA(x,y,.2,.001,30)
    _,E_NLMS[:,ex_num]=NLMS(x,y,1.2,.001)
    _,E_RLS[:,ex_num]=RLS(x,y,1,.01)
mean_APA=np.mean(E_APA,axis=1)
mean_NLMS=np.mean(E_NLMS,axis=1)
mean_RLS=np.mean(E_RLS,axis=1)
#teta_APA=np.transpose(teta_APA)
#teta_NLMS=np.transpose(teta_NLMS)
#teta_RLS=np.transpose(teta_RLS)
#teta_act=teta_act @ np.ones((1,1000))
#mse_apa=((teta_APA-teta_act) **2).mean(axis=0)
#mse_nlms=((teta_NLMS-teta_act) **2).mean(axis=0)
#mse_rls=((teta_RLS-teta_act) **2).mean(axis=0)

#%%
#visualization of the MSE in dB
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(10*np.log10(mean_APA),'r',label='APA')
ax.plot(10*np.log10(mean_NLMS),'b',label='NLMS')
ax.plot(10*np.log10(mean_RLS),'k',label='RLS')
plt.xlabel('sample number')
plt.ylabel('average MSE in dB')
legend=ax.legend(shadow=True)
plt.show()








