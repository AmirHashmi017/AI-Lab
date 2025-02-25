import numpy as np

#Creating Numpy nd Arrays.
arr1=np.array([1,2,3,4,5])
print(arr1)
print(type(arr1))
#2D Numpy ND arrays
arr2D=np.array([[1,2,3],[4,5,6]])
print(arr2D)
#Zeros Function
arr3=np.zeros((3,3))
print(arr3)
#Ones Function
arr4=np.ones((2,1))
print(arr4)
#Identity Function
arr5=np.identity(5)
print(arr5)
#Arange Function showing even numbers from 4 to 16.
arr6=np.arange(4,16,2)
print(arr6)
#Linspace function
arr7=np.linspace(10,20,10)
print(arr7)
#Copy Function
arr8=arr7.copy()
print(arr8)