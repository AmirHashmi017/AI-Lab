import pandas as pd
import numpy as np
labels=['a','b','c']
mydata=[1,2,3]
arr=np.array(mydata)
d={'a':10,'b':20,'c':30}

#Declare Series
print(pd.Series(data=mydata))

print(pd.Series(data=mydata,index=labels))


#Similar Working
print(pd.Series(mydata,labels))

#Series with dictionary
print(pd.Series(d))

#Series with Built in functions
print(pd.Series(data=[len,print,sum]))

ser1=pd.Series([1,2,3,4],['USA','Russia','Pakistan','America'])
print(ser1)

ser2=pd.Series([1,2,5,4],['USA','India','Iran','Turkey'])
print(ser2)

#Accessing data from series throgh index or key
print(ser2["Turkey"])

#Add the values with same keys in both series and NAN if not exist in both.
print(ser1+ser2)