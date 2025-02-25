import numpy as np
arr1=np.arange(24).reshape(6,4)
print(arr1)
arr2=np.arange(10)
print(arr2)
print(arr2[3])
print(arr2[2:4])
print(arr2[-1])

#Slicing
#Printing 2nd and 3rd column of arr1 
print(arr1[:,1:3])
#Printing 2nd and 3rd column of 2nd and 3rd row of arr1
print(arr1[2:4,1:3])

#Iteration
for i in arr1:
    print(i)
#For printing item by item
for i in np.nditer(arr1):
    print(i)

