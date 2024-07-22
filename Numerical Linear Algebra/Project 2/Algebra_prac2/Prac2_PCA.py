import numpy as np
import matplotlib.pyplot as plt

#Reading functions

#Read example.dat
def read_txt():
    X = np.genfromtxt('example.dat.txt', delimiter = ' ')
    return (X.T) #because it's said that we have 16 observations (columns) and 4 variables (rows)
#Read csv
def read_csv():    
    X = np.genfromtxt('RCsGoff.csv', delimiter = ',')
    return(X[1:,1:]) #We dont need neither de first row nor the first column.
#PCA function
def PCA(data,case): #data can be read from the .text (1) or the .csv and case can be the covariance (1) or correlation matrix
    
    if(data == 1):
        X = read_txt()
    else:
        X = read_csv()
        
    n = X.shape[1]
    m = X.shape[0]
    if(n<=1):
        print("The dimension is too small")
        exit()
     
    print("X dimension:", X.shape)
    
    #Mean and standard deviations vectors
    means = np.mean(X, axis = 1) #axis = 1 ---> rows
    sds = np.std(X,axis = 1)
    
    for i in range(m):
        for j in range(n):  
            X[i][j] = X[i][j] - means[i]
        
    if(case==1): #We now distinguish between the covariance (case 1) and the correlation (case 2) matrix
        Y = X.T*(1/(np.sqrt(n-1)))
        
    else: #Correlation matrix
        
        for i in range(m):
            for j in range(n):  
                X[i][j] = X[i][j]/sds[i]
    
        Y = X.T*(1/(np.sqrt(n-1)))  
    U,S,VH = np.linalg.svd(Y, full_matrices = False) #full_matrices = False because U and V have different shapes, S is just a vector
        
    #Portion of the total variance accumulated in each of the principal components
    var_tot = 0
    var_sol = []
    for i in range(S.shape[0]): #total variance
        var_tot+= S[i]*S[i]
    for i in range(S.shape[0]): #accumulated variance
        var_sol.append(S[i]*S[i]/var_tot*100)
        
    #Standard deviation of each of the principal components
    sd_vectors = np.std(VH@X, axis = 1)    
    #Expression of the original dataset in the new PCA coordinates
    new = (VH)@X    
    return (var_sol, sd_vectors, new,S)

#Rules to discuss the number of principal components needed to explain the data sets

def Kaiser(S):
    cont = 0
    vec = []
    for i in range(len(S)):
        if((S[i]*S[i])>1):
            cont += 1
            vec.append(cont+1) #to know which rows are important
    return(cont,vec)

def rule_34(vec): #vec will be the vector with the variance accumulation 
    cont = 0 
    sum_cont = 0
    while(sum_cont <= 75):
        sum_cont += vec[cont]
        cont += 1
    return(cont, sum_cont)

    
  


########   Main   ####

#Exercise 1 
print("FIRST DATASET \n")
print("COVARIANCE MATRIX information\n")
var_sol1, sd_vectors1, new,S1 = PCA(1,1)
print("The total variance accumulated in each of the principal components is: \n", var_sol1)
print("\n")
print("The standard deviation of each of the principal components is: \n",sd_vectors1)
print("\n")
print("The expression of the original dataset in the new PCA coordinates is:\n", new)
print("\n")

print("CORRELATION MATRIX information\n")
var_sol2, sd_vectors2, new,S2 = PCA(1,12)
print("The total variance accumulated in each of the principal components is: \n", var_sol2)
print("\n")
print("The standard deviation of each of the principal components is: \n",sd_vectors2)
print("\n")
print("The expression of the original dataset in the new PCA coordinates is: \n", new)

#Exercise 2 
print("SECOND DATASET \n")
print("COVARIANCE MATRIX information\n")
var_sol3, sd_vectors3, new,S3= PCA(2,1)
print("The total variance accumulated in each of the principal components is: \n", var_sol3)
print("\n")
print("The standard deviation of each of the principal components is: \n",sd_vectors3)
print("\n")
print("The expression of the original dataset in the new PCA coordinates is:\n", new)
print("\n")
print(new.shape)

print("Plot of PC1 and PC2 (the same that you showed in the project,see the output) \n")
plt.scatter(new[0, :], new[1, :])
plt.xlabel(f"PC1: {var_sol3[0]:.2f}% variance")
plt.ylabel(f"PC2: {var_sol3[1]:.2f}% variance")
plt.show()




#Analysis , KAISER RULE
print("KAISER RULE:\n")

print("FIRST DATASET: \n")
cont1, vec1 = Kaiser(S1)
print("COVARIANCE: ", cont1, vec1 )



cont2, vec2 = Kaiser(S2)
print("CORRELATION: ", cont2, vec2)

print("\n")

print("SECOND DATASET \n")

cont3, vec3 = Kaiser(S3)
print("COVARIANCE: ",cont3, vec3)
print("\n")

#3/4 RULE

print("3/4 RULE: \n")
print("FIRST DATASET: \n")
cont11,sum1 = rule_34(var_sol1)
print("COVARIANCE: ", cont11,"accumulated variance:", sum1)

cont21,sum2 = rule_34(var_sol2)
print("CORRELATION: ",cont21,"accumulated variance:", sum2)

print("\n")

print("SECOND DATASET \n")
cont31,sum3 = rule_34(var_sol3)
print("COVARIANCE: ",cont31,"accumulated variance:", sum3)

# Plot the scree plot
print("\n")
print("Scree plots (see the output)")

plt.plot(range(1, len(S1) + 1), var_sol1, marker='o')
plt.title('Scree Plot for the covariance matrix')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.show()

plt.plot(range(1, len(S2) + 1), var_sol2, marker='o')
plt.title('Scree Plot for the correlation matrix')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.show()

plt.plot(range(1, len(S3) + 1), var_sol3, marker='o')
plt.title('Scree Plot for the covariance matrix')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.show()


print("Plot the information as you asked:")
info = np.column_stack(new.T,var_sol3)
print(info)



















