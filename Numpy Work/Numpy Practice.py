import numpy as np
# arr1=np.zeros(10)
# print(arr1)
# arr2=np.ones(10)
# print(arr2)
# arr3=np.ones(10)*5
# print(arr3)
# arr4=np.arange(10,51)
# print(arr4)
# arr5=np.arange(10,51,2)
# print(arr5)
# arr6=np.arange(0,9).reshape(3,3)
# print(arr6)
# arr7=np.identity(3)
# print(arr7)
# print(np.random.rand())
# print(np.random.randn(25))
# print(np.arange(1,101).reshape(10,10)/100)
# print(np.linspace(0,1,20))
mat = np.arange(1,26).reshape(5,5)
print(mat[2:,1:])
print(mat[3,4])
print(mat[0:3,1:2])
print(mat[4])
print(mat[3:])
print(mat.sum())
print(mat.std())
print(mat.sum(axis=0))