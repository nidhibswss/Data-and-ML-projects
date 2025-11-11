#!/usr/bin/env python3


import pandas as pd


# loading the dataset directly from the URL given
dataframe = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)

attributes = ["sepal length", "sepal width","petal length","petal width","class"]
dataframe.columns = attributes



#------------------SEPAL LENGTH--------------------------------------#

print()
#stats for sepal length 
print("summary statistics  for sepal length")


min_sepal_length = dataframe['sepal length'].min()
print('Minimum: {:.1f}'.format(min_sepal_length))

max_sepal_length = dataframe['sepal length'].max()
print('Maximum: {:.1f}'.format(max_sepal_length))

mean_sepal_length =dataframe['sepal length'].mean()
print('Average: {:.2f}'.format(mean_sepal_length))

sd_sepal_length =dataframe['sepal length'].std()
print('Standard Deviation: {:.2f}'.format(sd_sepal_length))

print()
print()


#------------------SEPAL WIDTH--------------------------------------#

#stats for sepal width 
print("summary statistics for sepal width")


min_sepal_width = dataframe['sepal width'].min()
print('Minimum: {:.1f}'.format(min_sepal_width))


max_sepal_width = dataframe['sepal width'].max()
print('Maximum: {:.1f}'.format(max_sepal_width))

mean_sepal_width =dataframe['sepal width'].mean()
print('Average: {:.2f}'.format(mean_sepal_width))

sd_sepal_width =dataframe['sepal width'].std()
print('Standard deviation: {:.2f}'.format(sd_sepal_width))

print()
print()


#------------------PETAL LENGTH --------------------------------------#

#stats for petal length
print("summary statistics for petal length")


min_petal_length = dataframe['petal length'].min()
print('Minimum: {:.1f}'.format(min_petal_length))

max_petal_length = dataframe['petal length'].max()
print('Maximum: {:.1f}'.format(max_petal_length))

mean_petal_length =dataframe['petal length'].mean()
print('Average: {:.2f}'.format(mean_petal_length))

sd_petal_length =dataframe['petal length'].std()
print('Standard Deviation: {:.2f}'.format(sd_petal_length))

print()
print()

#------------------PETAL WIDTH--------------------------------------#

#stats for petal width
print('summary statistics for petal width')


min_petal_width = dataframe['petal width'].min()
print('Minimum: {:.1f}'.format(min_petal_width))

max_petal_width = dataframe['petal width'].max()
print('Maximum: {:.1f}'.format(max_petal_width))

mean_petal_width =dataframe['petal width'].mean()
print('Average: {:.2f}'.format(mean_petal_width))

sd_petal_width =dataframe['petal width'].std()
print('Standard Deviation: {:.2f}'.format(sd_petal_width))

print()
print()


#---counting frequency------#



count_freq = dataframe['class'].value_counts()
print("the frequency for each of its distinct class values:\n" , count_freq)





