import numpy as np
import time
from scipy.linalg import ldl
from scipy.linalg import solve_triangular


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

def F(G,C,A,x,g,lam,s,gamma,d,b):
  #First component of F
  rl = G@x + g - A@gamma - C@lam
  #Second component of F
  ra = b-A.T@x
  #Third component of F
  rc = s+d-C.T@x
  #Fourth component of F
  rs = np.multiply(s,lam)
  return -rl,-ra,-rc,-rs

def Create_MKKT_c5(A,G,C,n,m,p,s,lam):
  N =n+2*m+p
  M = np.zeros((N,N))

  M[:n, :n] = G
  M[:n, n:n+p] = -A
  M[:n, n+p:n+p+m] = -C
  M[n:n+p,:n] = -A.T
  M[n+p:n+p+m,:n] = -C.T
  M[n+p:n+p+m,n+p+m:N] = np.identity(m)
  M[n+p+m:N,n+p:n+p+m] = np.diag(s)
  M[n+p+m:N,n+p+m:N] = np.diag(lam)
  return(M)


def Create_MKKT_c6(A,G,C,n,m,p,s,lam):
  #Note that now the matrix is 3x3 block
  N =n+p+m
  M = np.zeros((N,N))

  M[:n, :n] = G
  M[:n, n:n+p] = -A
  M[:n, n+p:n+p+m] = -C
  M[n:n+p,:n] = -A.T
  M[n+p:n+p+m,:n] = -C.T
  M[n+p:N,n+p:N] = -np.identity(m)

  return(M)

def read_vec(name, n):
  v=np.zeros(n)
  data= [i.strip().split(' ') for i in open(name).readlines()]
  for i in range(len(data)):
    v[int(data[i][0])-1] = data[i][1]
  return(v)


def read_mat(name,n,m): #sym= if the matrix is symmetric
  M = np.zeros((n,m))
  data= [i.strip().split(' ') for i in open(name).readlines()]
  for i in range(len(data)):
    M[int(data[i][0])-1, int(data[i][1])-1] = data[i][2]
  return(M)


def c5(n,m,p,flag):
  N = n+2*m+p
  tol = 10**(-16)
  if(flag==0):
    A=read_mat('A.dad',n,p)
    C=read_mat('C.dad',n,m)
    G=read_mat('G.dad',n,n)
    G += G.T - np.diag(np.diag(G))
    #comprobar si la matriz es simetrica
    g=read_vec('g_vec.dad',n)
    b=read_vec('b.dad',p)
    d=read_vec('d.dad',m)
  else: 
    A=read_mat('A1.dad',n,p)
    C=read_mat('C1.dad',n,m)
    G=read_mat('G1.dad',n,n)
    G += G.T - np.diag(np.diag(G))
    #comprobar si la matriz es simetrica
    g=read_vec('g_vec1.dad',n)
    b=read_vec('b1.dad',p)
    d=read_vec('d1.dad',m)
        

  #Vector x, lam, s, gamma i mu

  x= np.zeros(n)
  z = np.zeros(N)
  f = np.zeros(N)
  lam = s = np.ones(m)
  gamma = np.ones(p)
  dlam = ds = np.zeros(m)
  #dgamma = np.zeros(p)
  mu = 1 #initial case
  z = np.concatenate((x,gamma,lam,s)) #z0
  M = Create_MKKT_c5(A,G,C,n,m,p,s,lam)
  #print(M) 

  #Conditions to enter the loop
  rl,ra,rc,rs = F(G,C,A,x,g,lam,s,gamma,d,b)
  cont = 0
  t0 = time.time()
  while( np.linalg.norm(rl)>=tol and np.linalg.norm(rc)>= tol and np.linalg.norm(ra)>= tol and abs(mu)>=tol and cont<100):

    f = np.concatenate((rl,ra,rc,rs))
    #First step
    dz = np.linalg.solve(M,f)
    dlam = dz[n+p:n+p+m]
    ds = dz[n+p+m:N]

    #Step size correction substep
    alpha = Newton_step(lam,dlam,s,ds)
    mu = (s.T@lam)/m
    mu_1 = ((s+alpha*ds).T)@(lam+alpha*dlam)/m
    sigma = (mu_1/mu)**3

    #Second system to solve
    aux = np.zeros(m)
    for i in range(m):
      aux[i] = rs[i] -dlam[i]*ds[i]+sigma*mu
    for i in range(m):
      f[i+n+p+m]= aux[i]

    dz = np.linalg.solve(M,f)

    #Step size correction substep
    dlam = dz[n+p:n+p+m]
    ds = dz[n+p+m:N]
    alpha = Newton_step(lam,dlam,s,ds)

    #Define new z
    z= z+ 0.95*alpha*dz

    x = z[:n]
    gamma = z[n:n+p]
    lam = z[n+p:n+p+m]
    s = z[n+p+m:N]

    #Updating matrix
    M[n+p+m:N,n+p:n+p+m] = np.diag(s)
    M[n+p+m:N,n+p+m:N] = np.diag(lam)
    cont += 1

    rl,ra,rc,rs = F(G,C,A,x,g,lam,s,gamma,d,b)

  t1 = time.time()
  comprobar = (1/2)*x.T@G@x+g.T@x
  print("time:", t1-t0, "f(x) = " ,comprobar, cont)
  

def ldlt(L,D,perm,new_f):
  N = len(new_f)
  y = solve_triangular(L[perm,:],new_f[perm],lower=True) #we permute both matrix and vector
  # # y_1 = np.linalg.solve(D,y)
  y_1 = np.zeros(N)
  y_1[0]=y[0]/D[0,0]
  y_1[-1]=y[-1]/D[-1,-1]
  for i in range(1,N-1):#We've found a 2x2 diagonal matrix
      if D[i,i-1]!=0:
        D_block = np.zeros((2,2))
        b_1 = np.zeros(2)
        D_block[0,0]=D[i-1,i-1]
        D_block[0,1]=D[i-1,i]
        D_block[1,0]=D[i,i-1]
        D_block[1,1]=D[i,i]
        b_1[0]=y[i-1]
        b_1[1]=y[i]
        b_1 = np.linalg.solve(D_block,b_1)
        y_1[i-1]=b_1[0]
        y_1[i]=b_1[1]
        i=i+1 #the next i is not a block matrix
      else:
          if D[i,i]!=0:
              y_1[i]=y[i]/D[i,i] # real diagonal matrix

  dz=solve_triangular(L[perm,:].T,y_1,lower=False)
  P=np.identity(N)
  P=P[perm,:]
  dz=np.matmul(P.T,dz)
  return(dz)


def c6(n,m,p,flag):
  #comprobar que G es simetrica
  N = n+p+m
  tol = 10**(-16)
  if(flag==0):
    A=read_mat('A.dad',n,p)
    C=read_mat('C.dad',n,m)
    G=read_mat('G.dad',n,n)
    G += G.T - np.diag(np.diag(G))
    #comprobar si la matriz es simetrica
    g=read_vec('g_vec.dad',n)
    b=read_vec('b.dad',p)
    d=read_vec('d.dad',m)
  else: 
    A=read_mat('A1.dad',n,p)
    C=read_mat('C1.dad',n,m)
    G=read_mat('G1.dad',n,n)
    G += G.T - np.diag(np.diag(G))
    #comprobar si la matriz es simetrica
    g=read_vec('g_vec1.dad',n)
    b=read_vec('b1.dad',p)
    d=read_vec('d1.dad',m)
  

  #Vector x, lam, s, gamma i mu

  x= np.zeros(n)
  z = np.zeros(N+m)
  #f = np.zeros(N+m
  lam = s = np.ones(m)
  gamma = np.ones(p)
  mu = 1 #initial case
  z = np.concatenate((x,gamma,lam,s)) #z0
  M = Create_MKKT_c6(A,G,C,n,m,p,s,lam)#MKKT matrix
  # print(M)

  ######################################### Empieza bucle ########################################


  rl,ra,rc,rs = F(G,C,A,x,g,lam,s,gamma,d,b)
  cont = 0
  dlam = ds = np.zeros(m)
  #dgamma = np.zeros(p)
  t0 = time.time()
  while( np.linalg.norm(rl)>=tol and np.linalg.norm(rc)>= tol and np.linalg.norm(ra)>= tol and abs(mu)>=tol and cont<100):

    new_f = np.zeros(N) #smaller dimension than f
    new_f[:n] = rl
    new_f[n:n+p] = ra
    new_f[n+p:N] = rc - np.multiply(np.divide(np.ones(m),lam),rs) #cambio de signo a rs

    #Solve the LDLT system
    L,D,perm = ldl(M)
    dz = ldlt(L,D,perm,new_f)

    #Step size correction substep
    dlam = dz[n+p:N]
    ds = np.multiply(np.divide(np.ones(m),lam),(rs-np.multiply(s,dlam)))
    alpha = Newton_step(lam,dlam,s,ds)
    mu = (s.T@lam)/m
    mu_1 = ((s+alpha*ds).T)@(lam+alpha*dlam)/m
    sigma = (mu_1/mu)**3

    #Segundo sistema a resolver
    rs += - np.multiply(dlam,ds) + sigma*mu*np.ones(m)
    new_f[n+p:N] = rc - np.multiply(np.divide(np.ones(m),lam),rs)

    #LDLT again
    dz = ldlt(L,D,perm,new_f)

    #Step size correction substep
    dlam = dz[n+p:N]
    ds = np.multiply(np.divide(np.ones(m),lam),(rs-np.multiply(s,dlam)))

    alpha = Newton_step(lam,dlam,s,ds)
    dz_big = np.concatenate((dz,ds))
    #Define new z
    z= z+ 0.95*alpha*dz_big
    x = z[:n]
    gamma = z[n:n+p]
    lam = z[n+p:N]
    s = z[N:N+m]

    #Updating matrix
    M[n+p:N,n+p:N] = -np.diag(np.multiply(np.divide(np.ones(m),lam),s))
    cont += 1

    rl,ra,rc,rs = F(G,C,A,x,g,lam,s,gamma,d,b)


  t1 = time.time()

  comprobar = (1/2)*x.T@G@x+g.T@x
  print("time:", t1-t0, "f(x) = " ,comprobar,cont)


c5(100,200,50,0) #np.linalg.solve
c6(100,200,50,0) #ldlt
c5(1000,2000,500,1) #np.linalg.solve
c6(1000,2000,500,1) #ldlt



