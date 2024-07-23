import numpy as np
from imageio.v2 import imread,imsave
from numpy.linalg import svd
import matplotlib.pyplot as plt


def SVD(image,k,flag):
    new_image = imread(image)
    compressed_image = np.zeros(new_image.shape)  
    norm_image = np.linalg.norm(new_image)
    
    error = 0
    for i in range(0,3):
        U, S, VH = svd(new_image[:,:,i], full_matrices=False)
        RGB = U[:, :k]@np.diag(S[:k])@VH[:k, :]
        compressed_image[:,:,i] = RGB
        error += np.linalg.norm(new_image[:,:,i] - compressed_image[:,:,i]) / np.linalg.norm(new_image[:,:,i] )
        
    error*=100/3
    compressed_image = np.rint(255*(compressed_image - np.min(compressed_image))/(np.max(compressed_image) - np.min(compressed_image)))
    compressed_norm = np.linalg.norm(compressed_image)
    captured = 100*compressed_norm/norm_image
    if(flag==1): #only if we want to save the image
        # Save the compressed color image
        name = f"compressed_color_{error}_{captured}.jpeg"
        imsave(name, compressed_image.astype(np.uint8))
        
    return(error, captured)    

######## MAIN ########

#HOMER PICTURE
print("\n")
print("First picture")
print("\n")
error1 = []
norm1 = []
i_vec = []
for i in range(5,51,5): 
    i_vec.append(i)
    aux1, aux2 = SVD('homer.jpg',i,1)
    error1.append(aux1)
    norm1.append(aux2)
    
print("error: \n",error1)
print("\n")
print("norm \n ",norm1)
    

plt.plot(i_vec,error1)
plt.xlabel('k')
plt.ylabel('Error')
plt.show()


#LETTER PICTURE
print("\n")
print("Second picture")
print("\n")

error2 = []
norm2 = []
i_vec = []
for i in range(5,51,5): 
  i_vec.append(i)
  aux1,aux2 = SVD('Letters.jpg',i,1)
  error2.append(aux1)
  norm2.append(aux2)
  
print("error: \n",error2)
print("\n")
print("norm \n ",norm2)



plt.plot(i_vec,error2)
plt.xlabel('k')
plt.ylabel('Error')
plt.show()


#NOBEL PICTURE
print("\n")
print("Third picture")
print("\n")
error3 = []
norm3 = []
i_vec = []
for i in range(5,51,5):
    i_vec.append(i)
    aux1, aux2 = SVD('Nobel.jpg',i,1)
    error3.append(aux1)
    norm3.append(aux2)
    
print("error: \n",error3)
print("\n")
print("norm \n ",norm3)

plt.plot(i_vec,error3)
plt.xlabel('k')
plt.ylabel('Error')
plt.show()










    
    
