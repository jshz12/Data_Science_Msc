
import numpy as np 
from scipy.io import mmread
import scipy.sparse as sp
import time 
import psutil

def D_matrix(G):
    
    n = G.shape[0]
    vec = np.zeros(n)
    d = np.zeros(n)
    
    vec = np.sum(G,axis=0).T
    for i in range(n):
        if(vec[i]!= 0):
            d[i] = 1/vec[i]

    return(sp.diags(d))

#PR storing
def PR1(A,m,tol):
    
    n = A.shape[0]
    
    e = np.ones(n)
    z = np.ones(n)
    x0 = np.zeros(n)
    xk = np.ones(n)/n
    count = 0
    
    z /=n 
    z[np.unique(A.indices)] *= m 
    while(np.linalg.norm(xk-x0,np.inf) > tol):
        x0 = xk
        xk = (1-m)*A@x0 + e*(z@x0)     
        count+=1
        
    xk = xk/np.sum(xk)

    
    return(xk,count)

#PR without storing
def PR2(G,m,tol):
    
    n = G.shape[0]
    L = []
    n_j = []
    for j in range(n):
        #webpages with link to page j
        indices = G.indices[G.indptr[j]:G.indptr[j+1]]
        L.append(indices)
        #length of each component of the vector L
        n_j.append(len(indices))
                   
    x0 = np.zeros(n)
    xk = np.ones(n)/n #to enter the while  
    count = 0
    while(np.linalg.norm(xk-x0,np.inf) > tol):
       xk = x0
       x0 = np.zeros(n)
       for j in range(n):
           if(n_j[j] == 0):
               x0+= xk[j]/n
           else:
               for i in L[j]:
                   x0[i] = x0[i] + xk[j] / n_j[j]                   
       x0 = (1-m)*x0 + m/n
       count+=1
       
    x0/=np.sum(x0)    
    
    return(x0,count)
    


######### MAIN ####

G = mmread('p2p-Gnutella30.mtx') #reading the Sparse matrix
D = D_matrix(G) #computing the D matrix
A = G@D #creation of A

print("Exercise 1 \n")
process = psutil.Process()
start_memory = 0
end_memory = 0

t0 = time.time()
start_memory = process.memory_info().rss
sol1,iter1 = PR1(A,0.15,1e-12)
t1 = time.time()
end_memory = process.memory_info().rss




indices1 = np.argsort(-sol1)
print("The sorted PR vector of M_m using the power method and storing matrices is: \n", np.sort(sol1)[::-1])
print("\n")
print("The first 10 scores of the previous vector are:\n", np.sort(sol1)[::-1][:10])
print("\n")
print("The TOP 10 PR score correspond to the following pages\n", indices1[:10])
print("\n")
print("Time needed:", t1-t0)
print("\n")
print("Iterations:", iter1)
print("\n")
print("Memory used in KB: ",(end_memory - start_memory)/1024)
print("\n")


start_memory = 0
end_memory = 0

print("Exercise 2\n")
start_memory = process.memory_info().rss
t3 = time.time()
sol2,iter2 = PR2(sp.csc_matrix(G), 0.15, 1e-12)
t4 = time.time()
end_memory = process.memory_info().rss



indices2 = np.argsort(-sol2)
print("The sorted PR vector of M_m using the power method and without storing matrices is: \n", np.sort(sol2)[::-1])
print("\n")
print("The first 10 scores of the previous vector are:\n", np.sort(sol2)[::-1][:10])
print("\n")
print("The TOP 10 PR score correspond to the following pages\n", indices2[:10])
print("\n")
print("Time needed:", t4-t3)
print("\n")
print("Iterations:", iter1)
print("\n")
print("Memory used in KB: ", (end_memory - start_memory)/1024)
print("\n")
print("Norm between the solutions: ", np.linalg.norm(np.sort(sol1)[::-1]-np.sort(sol2)[::-1]))


###Comments: I wanted to measure the memory used in each function to be more precise in the memory, while writing the results and making comparisons
#However, I do not understand why sometimes I did not get the results that I was expecting. If I am not wrong, the psutil library and the way that I have used
# is the correct way to measure the memory used, meaning that in the second time that I am calling the function is not storing also the previous usage of memory
## I just wanted to comment it here, and I have not included in the memory because I did not want to confuse you.
    
        
    