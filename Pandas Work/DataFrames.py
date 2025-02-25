import pandas as pd
import numpy as np
from numpy.random import randn
np.random.seed(101)

#Creating DataFrame
df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
print(df)
#Getting through column,A dataframe is a bunch of series
print(df['W'])
#OR
print(df.W)
#Getting multiple columns
print(df[['W','X']])
#Making new column
df['new']=df['W']+df['X']
print(df)
#Dropping Column
df.drop('new',axis=1,inplace=True)
print(df)
#Dropping Rows
df.drop('E',axis=0,inplace=True)
print(df)
print(df.shape)

#Select Rows
print(df.loc['A'])
#Selecting using index
print(df.iloc[0])
#Accessing single vaue
print(df.loc['A','Y'])

print(df.loc[['A','C'],['X','Y']])

#Data Frames Part 02
#Apply selection
booldf=df>0
print(booldf)
print(df[booldf])
#OR
print(df[df>0])

#Applying Codnition to column
print(df[df['W']>0])

print(df[df['W']>0][['X','Y']])

#Multiple Conditions
#AND
print(df[(df['W']>0)&(df['X']<0)])
#OR
print(df[(df['W']>0)|(df['X']<0)])

#Reset Index
print(df.reset_index())

newind='CA NY WY OR'.split()
print(newind)

df['States']=newind
print(df)

print(df.set_index('States'))