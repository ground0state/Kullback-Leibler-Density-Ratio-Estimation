
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

def basis(x, Y, h):
    A = np.array([])
    for y in Y:
        A = np.append(A, kernel(x, y, h))
    return A

def kernel(x, y, h):
    norm = np.linalg.norm(x-y)
    return np.exp(-(norm**2)/(2*h**2))

def main():

    DEBUG = True

    """preparation for data"""
    normal_data = np.loadtxt("normal_data.txt",delimiter=",")
    error_data = np.loadtxt("error_data.txt",delimiter=",")
    kernel_data = np.loadtxt("normal_data.txt",delimiter=",")
    
    N_NML = len(normal_data) #Number of normal data
    N_ERR = len(error_data) #Number of error data
    N_KNL = len(kernel_data) #Number of kernel data
    
###Set Your Parameters##########################
    """parameters of density ratio estimation"""
    size = 0.01 #Step size #comment; Too large step size causes negative density ratio, so small step size is preferable.
    cnt = 10000 #Number of loop
    split = 4 #Data split
    SILVERMAN = 1.06*np.std(normal_data, axis=0)/pow(N_NML,1/5)
    #Candidate of bandwidth
    try:
        normal_data.shape[1]
        band_width_candidate = 0.8*SILVERMAN  #use for vector data
    except IndexError:
        band_width_candidate = [0.6*SILVERMAN,0.7*SILVERMAN,0.8*SILVERMAN] #use for scalar data
################################################

    """Cross-validation of width"""
    M_NML = N_NML // split #Block size of normal data
    M_ERR = N_ERR // split #Block size of error data
    #comment; To make it the same block size, using Floor Division

    split_normal_data = normal_data[0:split*M_NML] #Data with omitted remainder
    split_error_data = error_data[0:split*M_ERR] #Data with omitted remainder
    basis_list_normal = [] #Basis for each width
    basis_list_error = [] #Basis for each width
    
    J_h = np.array([]) #J of h bandwidth
    for h in band_width_candidate:        
        #basis for normal data
        basis_mtx_normal = np.empty((0, N_KNL))
        for x in split_normal_data:
            basis_mtx_normal = np.append(basis_mtx_normal, basis(x, kernel_data, h).reshape(1,-1), axis=0)
            
        basis_list_normal.append(basis_mtx_normal)
        
        
        #basis for error data
        basis_mtx_error = np.empty((0, N_KNL))
        for x in split_error_data:
            basis_mtx_error = np.append(basis_mtx_error, basis(x, kernel_data, h).reshape(1,-1), axis=0)
        
        basis_list_error.append(basis_mtx_error)

        
        J_k = np.array([])#J of k-th split
        for k in range(1, split+1):
            #using data other than k-th split
            basis_mtx_normal_otk = np.append(basis_mtx_normal[0:(k-1)*M_NML,:], basis_mtx_normal[k*M_NML:split*M_NML,:], axis=0)
            basis_mtx_error_otk = np.append(basis_mtx_error[0:(k-1)*M_ERR,:], basis_mtx_error[k*M_ERR:split*M_ERR,:], axis=0)
            
            #using data k-th split
            basis_mtx_normal_k = basis_mtx_normal[(k-1)*M_NML:k*M_NML,:]
            basis_mtx_error_k = basis_mtx_normal[(k-1)*M_ERR:k*M_ERR,:]
            
            #Gradient method
            theta = np.ones(N_KNL) #initial value
            dJ_dtheta_ini = np.sum(basis_mtx_error_otk, axis=0) / ((split-1)*M_ERR)
            for i in range(cnt+1):
                dJ_dtheta = dJ_dtheta_ini - np.sum(basis_mtx_normal_otk.T / np.dot(basis_mtx_normal_otk, theta), axis=1)/((split-1)*M_NML)
                theta = theta - size * dJ_dtheta
                if np.min(np.dot(basis_mtx_normal_k, theta)) <= 0:
                    theta = theta + size * dJ_dtheta
                    if DEBUG:
                        print("Cross-validation of width counter stop; h=",h,"k=",k,"i=",i)
                    break

            #Evaluation function
            J = np.dot(theta, np.sum(basis_mtx_error_k, axis=0)) / M_ERR             - np.sum(np.log(np.dot(basis_mtx_normal_k, theta))) / M_NML
            J_k = np.append(J_k ,J)
                      
        J_h = np.append(J_h, np.average(J_k))
   
    arg_h = np.argmin(J_h)
    band_width = band_width_candidate[arg_h] #bandwidth that J is minimized
    
    basis_mtx_normal = basis_list_normal[arg_h]
    basis_mtx_error = basis_list_error[arg_h]
    
    """Density ratio estimation"""
    #Gradient method
    J_list = []
    theta_list = []
    theta = np.ones(N_KNL) #initial value
    dJ_dtheta_ini = np.sum(basis_mtx_error, axis=0) / (split*M_ERR)
    for i in range(1, cnt+1):
        dJ_dtheta = dJ_dtheta_ini - np.sum((basis_mtx_normal.T / np.dot(basis_mtx_normal, theta)), axis=1)/(split*M_NML)
        theta = theta - size * dJ_dtheta
        if np.min(np.dot(basis_mtx_normal, theta)) <= 0 or np.min(np.dot(basis_mtx_error, theta)) <= 0:
            theta = theta + size * dJ_dtheta
            if DEBUG:
                print("Density ratio estimation counter stop; i=",i)
            break
        #Evaluation function
        J = np.dot(theta, np.sum(basis_mtx_error, axis=0)) / (split*M_ERR)             - np.sum(np.log(np.dot(basis_mtx_normal,theta))) / (split*M_NML)
        J_list.append(J)
        theta_list.append(theta)
   
    a = np.array([]) #Degree of error
    r = np.array([]) #Density ratio
    infty = 1000
    for basis_error in basis_mtx_error:
        if np.dot(theta, basis_error) == 0:
            r = np.append(r, 0)
            a = np.append(a, infty)
        else:
            density_ratio = np.dot(theta, basis_error)
            r = np.append(r, density_ratio)
            a = np.append(a, -np.log(density_ratio))
    if DEBUG:        
        print("Band width candidate:",band_width_candidate)
        print("J_h:",J_h)

    """Making figure"""
    x = np.arange(0, len(J_list), 1)
    y = J_list
    
    plt.subplot(211)
    plt.plot(x, y)
    plt.title("Evaluation function")
    
    u = np.arange(0, len(a), 1)
    v = a
    
    plt.subplot(212)
    plt.plot(u, v)
    plt.title("Degree of error")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
    
    

