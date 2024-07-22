import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular,qr



#Reading functions
def read_dades(degree):
   
    X = np.genfromtxt('dades.csv', delimiter='  ')
    n = X.shape[0] #rows
    points = np.zeros((n,2)) 
    
    
    for i in range(0,n):
        points[i,0] = X[i,0]
        points[i,1] = X[i,1]  
    if (degree<2): #if we just want to see the data
        return(points)
    else: 
    
        A = np.zeros((n,degree))
        for i in range(0,n):
            A[i,:] = [points[i,0]**d for d in range(0,degree)]
        b = points[:,1]
        
        return(A,b,points)

def read_csv():
    X = np.genfromtxt('dades_regressio.csv', delimiter=',')
    A = X[:,:-1] #not taking into account the last column, because it's b
    b = X[:,-1] #only the last column of X
    return(A,b)



#QR problem 
def QR(A,b):

    r = np.linalg.matrix_rank(A)  #Let's first check the rank of the matrix
    if(r == A.shape[1]): #full rank, (m>=n)
        Q1, R1 = np.linalg.qr(A) 
        x = solve_triangular(R1,Q1.T@b) #default is to use upper triangular
        
    else:         
        #not full rank
        Q, R, P = qr(A, pivoting = True) #complete QR factorization
        R1 = R[:r,:r] 
        c = ((Q.T) @b)[:r] 
        u = solve_triangular(R1,c) #basic solution to obtain u and v 
        v = np.zeros((A.shape[1]-r))
        x = np.eye(A.shape[1])[:,P]@np.concatenate((u,v))
    return(x)

#SVD
def SVD(A,b): 
    return (np.linalg.pinv(A)@b)

########################### MAIN ###############################

#First dataset

print("FIRST DATASET \n")

#Let us see the points
points = read_dades(0)
plt.plot(points[:,0], points[:,1], 'o', markersize=2)
plt.show()

#QR

print("QR: \n")
qr_errors = []
for d in range(2,20):
    A, b, points = read_dades(d)
    x = QR(A,b)
    qr_errors.append(np.linalg.norm(A@x-b))
print("Errors: \n", qr_errors)


#Minimum error, we add 2 because we started at degree 2
best_degree = np.argmin(qr_errors) + 2
print("The best degree, according to the error, is:", best_degree)

#Let us check how the best degree is working
A, b, points = read_dades(best_degree)
x_bq = QR(A,b)
print( "x = \n", x_bq)
print('Best degree:', best_degree)
print('Error:', np.linalg.norm(A@x_bq-b))
print('\n') 



plt.scatter(points[:,0],points[:,1],s=5,c='b')
x = np.linspace(1,8.2,100)
y = 0
for i in range(0,best_degree):
    y = y + x_bq[i]*x**i

plt.plot(x,y,color='red')
plt.title('Least Squares Problem with QR')
plt.show()


#SVD
print("SVD: \n")

svd_errors = []
for d in range(2,20):
    A, b, points = read_dades(d)
    x = SVD(A,b)        
    svd_errors.append(np.linalg.norm(A@x-b))
print("Errors: \n", svd_errors)

#Minimum error, we add 2 because we started at degree 2
best_degree = np.argmin(svd_errors) + 2

#Let us check how the best degree is working
A, b, points = read_dades(best_degree)
x_bs = SVD(A,b)
print( "x = \n", x_bs)
print('Best degree:', best_degree)
print('Error:', np.linalg.norm(A@x_bs-b))
print('\n') 


plt.scatter(points[:,0],points[:,1],s=5,c='b')
x = np.linspace(1,8.2,100)
y = 0
for i in range(0,best_degree):
    y = y + x_bs[i]*x**i

plt.plot(x,y,color='green')
plt.title('Least Squares Problem with SVD')
plt.show()
        

#In order to be able to compare the vectors let's assume that the best degree for svd it's 11 and not 12

print("Choosing degree 11: \n")
#Let us check how the best degree is working
A, b, points = read_dades(11)
x_bs = SVD(A,b)
print( "x = \n", x_bs)
print('Best degree:', 11)
print('Error:', np.linalg.norm(A@x_bs-b))
print('\n') 


plt.scatter(points[:,0],points[:,1],s=5,c='b')
x = np.linspace(1,8.2,100)
y = 0
for i in range(0,11):
    y = y + x_bs[i]*x**i

plt.plot(x,y,color='green')
plt.title('Least Squares Problem with SVD')
plt.show()

print("Difference vector between QR and SVD:\n", x_bq-x_bs)
print("Norm of the previous vector: \n", np.linalg.norm(x_bq-x_bs), " \n")



#Second dataset
print("SECOND DATASET  \n")

#QR
A,b = read_csv()
x1 = QR(A,b)
print("QR: \n", x1)

#SVD
x2 = SVD(A,b)
print("SVD: \n", x2)


print("Difference vector between QR and SVD:\n", x1-x2)
print("Norm of the previous vector: \n", np.linalg.norm(x1-x2))
print("Error with QR:", np.linalg.norm(A@x1-b))
print("Error with LS:", np.linalg.norm(A@x2-b))

    

