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
import numpy as np
N=1000
l=5
ex=200
alpha=.95 #parameter of time varying model
noise_par_var=.1 #parameter of time varying model
teta_act=np.random.randn(l,)
E_NLMS=np.zeros((N,ex))
E_RLS=np.zeros((N,ex))
for ex_num in range(ex):
    y=np.zeros((N,))
    x=np.random.randn(N,l)
    teta=teta_act
    noise_out=np.random.normal(0,np.sqrt(.001),(N,1))
    noise_par=np.random.normal(0,np.sqrt(noise_par_var),(N,l))
    for n in range(1,N):
        y[n]= teta @ x[n] +noise_out[n]
        teta = alpha * teta + noise_par[n]
    _,E_NLMS[:,ex_num]=NLMS(x,y,.5,.001)
    _,E_RLS[:,ex_num]=RLS(x,y,.997,.001)
mean_NLMS=np.mean(E_NLMS,axis=1)
mean_RLS=np.mean(E_RLS,axis=1)
#%%
#visualization of the MSE in dB
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(10*np.log10(mean_NLMS),'b',label='NLMS')
ax.plot(10*np.log10(mean_RLS),'r',label='RLS')
plt.xlabel('sample number')
plt.ylabel('average MSE in dB')
legend=ax.legend(shadow=True)
plt.show()
    
        


    
    
    
    
    
    
    
    
    
    
    
    