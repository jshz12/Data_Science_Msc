import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import ldl  #C4
from scipy.linalg import solve_triangular #C4
from scipy.linalg import cholesky #C4

#EXERCISE C1
#Step size correction
def Newton_step(lamb0,dlamb,s0,ds):
  alp=1;
  idx_lamb0=np.array(np.where(dlamb<0)) #nos da los indices donde dlamb < 0
  if idx_lamb0.size>0:
    alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))

  idx_s0=np.array(np.where(ds<0)) #nos da los indices donde ds < 0
  if idx_s0.size>0:
    alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))

  return alp


#A = 0
#Used in both C2,C3,C4
def F(G,C,x,g,lam,s,d):
  #First component of F
  rl = G@x + g - C@lam
  #Second component of F
  rc = s+d-C.T@x
  #Third component of F
  rs= np.multiply(s,lam)
  return -rl,-rc,-rs

#C2
def create_MKKT_c2(n):
    
    
  m = 2*n
  N = n +2*m
  M = np.zeros((N,N))

  #Creation of MKKT matrix
  M[:n,:n] = np.identity(n)
  M[:n,n:n+n] = M[n:n+n,:n] = -np.identity(n)
  M[:n,m:n+m] = M[m:n+m,:n] = np.identity(n)

  M[n:n+m,n+m:N] = np.identity(m) #Id
  M[n+m:N,n+m:N] = np.identity(m) #Lambda
  M[n+m:N,n:n+m] = np.identity(m) #S

  return(M)

#C4
def create_MKKT_c4_1(n):
    
    
  m = 2*n
  N = n+m
  M = np.zeros((N,N))

  #Creation of MKKT matrix
  M[:n,:n] = np.identity(n) #G
  M[:n,n:m] = M[n:m,:n]= -np.identity(n)
  M[:n,m:N] = M[m:N,:n]= np.identity(n)
  M[n:N,n:N] = -np.identity(m)


  return(M)

def create_MKKT_c4_2(G,C,s,lam):
  n = len(G[0])
  m =len(s)
  M = np.zeros((n,n))
  M = G + C@np.diag(np.divide(np.ones(m), s))@np.diag(lam)@C.T
  return(M)

#EXERCISE C2
def c2(n,g):
  m = 2*n
  N = n+2*m
  tol = 10**(-16)

  M = create_MKKT_c2(n) #Let M be de MKKT matrix
  #print(M)
  #We want to study the condition number of the MKKt matrix at each step 
  cond_number = []
  cond_number.append( np.linalg.cond(M,2) )
  #Vectors
  d = np.full(m,-10)
  #g = np.random.normal(0,1,n)
  x= np.zeros(n)
  z = np.zeros(N)
  f = np.zeros(N)
  dz = np.zeros(N)
  lam = s = np.ones(m)
  mu = 1
  dlam = ds = np.ones(m)
  z = np.concatenate((x,lam,s)) #z0

  #G and C matrices
  G = np.identity(n)
  C= np.zeros((n,m))
  C[:n,:n] = np.identity(n)
  C[:n,n:n+n] = -np.identity(n)

  #Conditions to enter the loop
  rl,rc,rs = F(G,C,x,g,lam,s,d)
  cont = 0
  t0 = time.time()
  while( np.linalg.norm(rl)>=tol and np.linalg.norm(rc)>= tol and abs(mu)>=tol and cont<100):

    f = np.concatenate((rl,rc,rs))

    #First step
    dz = np.linalg.solve(M,f)
    dlam = dz[n:n+m]
    ds = dz[n+m:N]

    #Step size correction substep
    alpha = Newton_step(lam,dlam,s,ds)
    mu = (s.T@lam)/m
    mu_1= ((s+alpha*ds).T)@(lam+alpha*dlam)/m
    sigma =((mu_1/mu))**3

    #Second system to solve
    rs = rs - np.multiply(dlam,ds) + sigma*mu*np.ones(m)
    f[n+m:N]= rs
    dz = np.linalg.solve(M,f)

    #Step size correction substep
    dlam = dz[n:n+m]
    ds = dz[n+m:N]
    alpha = Newton_step(lam,dlam,s,ds)

    #Define new z
    z+= 0.95*alpha*dz
    x = z[:n]
    lam = z[n:n+m]
    s = z[n+m:N]

    #Updating matrix
    M[n+m:N,n+m:N] = np.diag(lam) #Lambda
    M[n+m:N,n:n+m] = np.diag(s) #S
    cond_number.append( np.linalg.cond(M,2) )
    cont += 1
    rl,rc,rs = F(G,C,x,g,lam,s,d)

  t1 = time.time()
  #print("n :", n, "time:", t1-t0, "distance between x and g:",np.linalg.norm(x+g) )
  return(t1-t0,cont,x,cond_number)

#EXERCISE C4
#C4, Strategy 1
def c4_1(n,g):
  m = 2*n
  N = n+m
  tol = 10**(-16)

  M = create_MKKT_c4_1(n)
  #print(M)
  #We want to study the condition number of the MKKt matrix at each step 
  cond_number = []
  cond_number.append( np.linalg.cond(M,2) )
  cond_number_L = []
  cond_number_D = []

  #Vectors
  d = np.full(m,-10)
  #g = np.random.normal(0,1,n)
  x= np.zeros(n)
  z = np.zeros(n+2*m)
  #f = np.zeros(n+2*m)
  lam = s = np.ones(m)
  mu = 1 #by definition, only at the initial case
  z = np.concatenate((x,lam,s)) #z0
  dlam = ds = np.ones(m)


  #G and C matrices
  G = np.identity(n)
  C= np.zeros((n,m))
  C[:n,:n] = np.identity(n)
  C[:n,n:m] = -np.identity(n)

  
  #Conditions to enter the loop
  r1,r2,r3 = F(G,C,x,g,lam,s,d)
  cont = 0
  t0 = time.time()
  while( np.linalg.norm(r1)>=tol and np.linalg.norm(r2)>= tol and abs(mu)>=tol and cont<100):

    #f = np.concatenate((r1,r2,r3))
    new_f = np.zeros(n+m) #smaller dimension than f
    new_f[:n] = r1
    new_f[n:] = r2 - np.multiply(np.divide(np.ones(m),lam),r3) #cambio de signo a r3

    #First step
    L,D,perm = ldl(M)
    #Compte the condition number of both matrices 
    if(cont==0):
        cond_number_L.append( np.linalg.cond(L,2) )
        cond_number_D.append( np.linalg.cond(D,2) )
    
   #Solve the LDLT system
    y = solve_triangular(L[perm,:],new_f[perm],lower =True)
    y_1 = np.divide(y,np.diagonal(D))
    dz = solve_triangular(L[perm,:].T,y_1,lower = False)

    dlam = dz[n:n+m]
    ds = np.multiply(np.divide(np.ones(m),lam),(r3-np.multiply(s,dlam))) # according to the new formula


    #Step size correction substep
    alpha = Newton_step(lam,dlam,s,ds)
    mu = (s.T@lam)/m
    mu_1= ((s+alpha*ds).T)@(lam+alpha*dlam)/m
    sigma =((mu_1/mu))**3


    #Segundo sistema a resolver

    r3 += - np.multiply(dlam,ds) + sigma*mu*np.ones(m)
    new_f[n:] = r2 - np.multiply(np.divide(np.ones(m),lam),r3)

    #LDLT again
    y = solve_triangular(L[perm,:],new_f[perm],lower =True)
    y_1 = np.divide(y,np.diagonal(D))
    dz = solve_triangular(L[perm,:].T,y_1,lower = False)

    #Step size correction substep
    dlam = dz[n:n+m]
    ds = np.multiply(np.divide(np.ones(m),lam),(r3-np.multiply(s,dlam)))
    alpha = Newton_step(lam,dlam,s,ds)
    dz_big = np.concatenate((dz,ds))

    #Define new z
    z= z+ 0.95*alpha*dz_big
    x = z[:n]
    lam = z[n:n+m]
    s = z[n+m:n+2*m]

    #Updating matrix

    M[n:N,n:N] = -np.diag(np.multiply(np.divide(np.ones(m),lam),s))
    cond_number.append( np.linalg.cond(M,2) )
    cont += 1
    r1,r2,r3 = F(G,C,x,g,lam,s,d)
    
  cond_number_L.append( np.linalg.cond(L,2) )
  cond_number_D.append( np.linalg.cond(D,2) )  
  t1 = time.time()
  
  return(t1-t0,cont,x,cond_number,cond_number_L,cond_number_D)

#C4, Strategy 2 
def c4_2(n,g):
  m = 2*n
  tol = 10**(-16)

  #Vectors
  d = np.full(m,-10)
  #g = np.random.normal(0,1,n)
  x= np.zeros(n)
  f = np.zeros(n)
  z = np.zeros(n+2*m)
  lam = s = np.ones(m)
  dx = np.zeros(n)
  dlam = ds = np.zeros(m)
  mu = 1 #initial case
  z = np.concatenate((x,lam,s)) #z0

  #G and C matrices
  G = np.identity(n)
  C= np.zeros((n,m))
  C[:n,:n] = np.identity(n)
  C[:n,n:m] = -np.identity(n)
  M = create_MKKT_c4_2(G,C,s,lam)
  #We want to study the condition number of the MKKt matrix at each step 
  cond_number = []
  cond_number.append( np.linalg.cond(M,2) )
  cond_number_L = []

  

  #Conditions to enter the loop
  r1,r2,r3 = F(G,C,x,g,lam,s,d)
  cont = 0
  t0 = time.time()
  while( np.linalg.norm(r1)>=tol and np.linalg.norm(r2)>= tol and abs(mu)>=tol and cont<100):

   #Solve the cholesky system
    f = r1 + C@np.diag(np.divide(np.ones(m),s))@(r3-np.diag(lam)@r2)
    L = cholesky(M)
    if(cont==0):
        cond_number_L.append( np.linalg.cond(L,2))
    y=solve_triangular(L,f,lower=True)
    dx=solve_triangular(L.T,y,lower=False)

    #Step size correction substep
    dlam = np.diag(np.divide(np.ones(m),s))@(r3 - np.diag(lam)@r2) - np.diag(np.divide(np.ones(m),s))@(np.diag(lam)@C.T@dx)
    ds = r2 + C.T@dx
    alpha = Newton_step(lam,dlam,s,ds)
    mu = (s.T@lam)/m
    mu_1= ((s+alpha*ds).T)@(lam+alpha*dlam)/m
    sigma =((mu_1/mu))**3


    #Segundo sistema a resolver
    r3 = r3 -np.multiply(dlam,ds) + sigma*mu*np.ones(m)
    f = r1 + C@np.diag(np.divide(np.ones(m),s))@(r3-np.diag(lam)@r2)

    y = solve_triangular(L,f,lower=True)
    dx = solve_triangular(L.T,y,lower=False)

    #Step size correction substep
    dlam = np.diag(np.divide(np.ones(m),s))@(r3 - np.diag(lam)@r2) - np.diag(np.divide(np.ones(m),s))@(np.diag(lam)@C.T@dx)
    ds = r2 + C.T@dx
    alpha = Newton_step(lam,dlam,s,ds)
    dz_big = np.concatenate((dx,dlam,ds))

    #Define new z
    z+= 0.95*alpha*dz_big
    x = z[:n]
    lam = z[n:n+m]
    s = z[n+m:n+2*m]

    #Updating matrix
    M = create_MKKT_c4_2(G,C,s,lam)
    cond_number.append( np.linalg.cond(M,2) )
    cont += 1
    r1,r2,r3 = F(G,C,x,g,lam,s,d)
    
  cond_number_L.append( np.linalg.cond(L,2))  
  t1 = time.time()
  return(t1-t0,cont,x,cond_number, cond_number_L)



#EXERCISE C3 (not only)

#Time vectors
t_c2 = np.zeros(100)
t_c4_1  = np.zeros(100)
t_c4_2 = np.zeros(100)
#Iteration vectors
it_c2 = np.zeros(100)
it_c4_1  = np.zeros(100)
it_c4_2 = np.zeros(100)
#Precision vectors
p_c2 = np.zeros(100)
p_c4_1  = np.zeros(100)
p_c4_2 = np.zeros(100)
# Initial Condition number vectors
in_cond_c2 = []
in_cond_c4_1 =[]
in_cond_c4_2 = []

in_l_c4_1 = []
in_d_c4_1 = []
in_l_c4_2 = []



#Last Condition number vectors 
last_cond_c2 = []
last_cond_c4_1 =[]
last_cond_c4_2 = []

last_l_c4_1 = []
last_d_c4_1 = []
last_l_c4_2 = []


#Auxiliar 
aux = []
aux1 = []
aux2 = []


#Dimension vector
numbers = np.zeros(100)
for n in range (1,101):
    
    g = np.random.normal(0,1,n)
    
    #c2
    t0,cont0,x0, aux= c2(n,g)
    in_cond_c2.append(aux[0]) #Condition number study 
    last_cond_c2.append(aux[-1])
    
    #c4_1
    t1,cont1,x1,aux,aux1,aux2 = c4_1(n,g)
    in_cond_c4_1.append(aux[0]) #Condition number study 
    in_l_c4_1.append(aux1[0])
    in_d_c4_1.append(aux2[0])
    last_cond_c4_1.append(aux[-1])
    last_l_c4_1.append(aux1[-1])
    last_d_c4_1.append(aux2[-1])
    
    #c4_2
    t2, cont2, x2, aux,aux1= c4_2(n,g)
    in_cond_c4_2.append(aux[0]) #Condition number study 
    last_cond_c4_2.append(aux[-1])
    in_l_c4_2.append(aux1[0])
    last_l_c4_2.append(aux1[-1])
    
    #Time
    t_c2[n-1] = t0    
    t_c4_1[n-1] = t1
    t_c4_2[n-1] = t2
    #Iterations
    it_c2[n-1] = cont0    
    it_c4_1[n-1] = cont1
    it_c4_2[n-1] = cont2
    #Precision 
    p_c2[n-1] = np.linalg.norm(x0+g)
    p_c4_1[n-1] = np.linalg.norm(x1+g)
    p_c4_2[n-1] = np.linalg.norm(x2+g)
    #Dimension
    numbers[n-1] = n
   
    
#print(t_c2[-1], t_c4_1[-1], t_c4_2[-1])

#Execution time plot
plt.plot(numbers,t_c2, color= 'green', label='np.linalg.solve' )
plt.plot(numbers,t_c4_1, color = 'red', label='ldl^t ')
plt.plot(numbers,t_c4_2, color = 'blue', label='cholesky')
plt.title("Execution time comparison")
plt.xlabel("Dimension n")
plt.ylabel("Execution time")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

#Iteration plots
plt.plot(numbers,it_c2, color= 'green' )
plt.title("Number of iterations with 'np.linalg.solve'")
plt.xlabel("Dimension n")
plt.ylabel("Number of iterations")
plt.show()

plt.plot(numbers,it_c4_1, color= 'red')
plt.title("Number of iterations with 'ldl^t'")
plt.xlabel("Dimension n")
plt.ylabel("Number of iterations")
plt.show()

plt.plot(numbers,it_c4_2, color= 'blue')
plt.title("Number of iterations with 'cholesky'")
plt.xlabel("Dimension n")
plt.ylabel("Number of iterations")
plt.show()



#Precision plots
plt.plot(numbers,p_c2, color= 'green')
plt.title("Precision with 'np.linalg.solve'")
plt.xlabel("Dimension n")
plt.ylabel("Precision")
plt.show()


plt.plot(numbers,p_c4_1, color = 'red')
plt.title("Precision with 'ldl^t'")
plt.xlabel("Dimension n")
plt.ylabel("Precision")
plt.show()


plt.plot(numbers,p_c4_2, color = 'blue')
plt.title("Precision with 'cholesky'")
plt.xlabel("Dimension n")
plt.ylabel("Precision")
plt.show()

#Condition number plots

#Comparison of initial condition number
plt.plot(numbers,in_cond_c2, color= 'green', label ='np.linalg.solve')
plt.plot(numbers,in_cond_c4_1, color = 'red', label='ldl^t ')
plt.plot(numbers,in_cond_c4_2, color = 'blue', label='cholesky')
plt.title("Initial condition number")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()


#Linalg solve
plt.plot(numbers,last_cond_c2, color= 'green', label ='np.linalg.solve')
plt.title("Last condition number with 'np.linalg.solve'")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.show()

#ldlt
plt.plot(numbers,last_cond_c4_1, color = 'red', label='ldl^t')
plt.title("Last condition number with 'ldl^t'")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.show()

#cholesky
plt.plot(numbers,last_cond_c4_2, color = 'blue', label='cholesky')
plt.title("Last condition number with 'cholesky'")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.show()


#ldlt, condition number of l and d
plt.plot(numbers,last_l_c4_1, color = 'red', label='ldl^t')
plt.title("Last condition number of l with 'ldl^t' ")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.show()

plt.plot(numbers,last_d_c4_1, color = 'red', label='ldl^t')
plt.title("Last condition number of d  with 'ldl^t' ")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.show()




#cholesky, condition number of l
plt.plot(numbers,last_l_c4_2, color = 'red', label='ldl^t')
plt.title("Last condition number of l with cholesky ")
plt.xlabel("Dimension n")
plt.ylabel("Condition number")
plt.show()






 
#Evolution of condition number for an specific dimension n 


#I wanted to study this for all the methods, for just a few (2 or 3) values of n, to study from another point of view the condition number. But I run out of time...
#for i in range(3):
#n = np.random.randint(10, 100)
#g = np.random.normal(0,1,n)
#t0,cont0,x0,aux= c2(n,g)
#print("this is de number of condition", aux[-1])
#plt.plot(np.arange(cont0+1),aux, color= 'green', label ='np.linalg.solve')
#plt.xlabel("Iteration")
#plt.ylabel("Condition number")
#plt.title("Condition number evolution")
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#plt.show()
    





















    


    
    
    
    
    
    
    
    
    
    
