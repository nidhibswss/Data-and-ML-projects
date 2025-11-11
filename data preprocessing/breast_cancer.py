#Data preprocessing can significantly improve the quality of the data mining analysis.
# In this lab, i will write Python code to alleviate some data quality issues, such as missing values and duplicates.

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt  #for 8)


#2)load the ‘breast-cancer-wisconsin.data’ file into a DataFrame.
#3)Assign new headers to the DataFrame. The headers should be relevant to the attribute description in the file ‘breast-cancer-wisconsin.names’. 

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", 
                   header=None)


data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']



#4) Drop the 'Sample code number' attribute

new_data = data.drop(columns=['Sample code number'],axis=1)

#5)According to the description of the data, the missing values are encoded as '?' . Replace the '?' to the numpy’s constant ‘NaN’

missing_value = np.nan
new_data.replace('?',missing_value,inplace=True)

 
#6) For each attribute, count and print the number of missing values

#mvc stands for missing value count

mvc_Clump = new_data['Clump Thickness'].isna().sum()
print(f"missing values in Clump Thickness is : {mvc_Clump}")

mvc_CellSize = new_data['Uniformity of Cell Size'].isna().sum()
print(f"missing values in Uniformity of Cell Size is : {mvc_CellSize}")

mvc_CellShape = new_data['Uniformity of Cell Shape'].isna().sum()
print(f"missing values in Uniformity of Cell Shape is : {mvc_CellShape}")

mvc_Marginal = new_data['Marginal Adhesion'].isna().sum()
print(f"missing values in Marginal Adhesion is : {mvc_Marginal}")

mvc_Epilithelial = new_data['Single Epithelial Cell Size'].isna().sum()
print(f"missing values in Single Epithelial Cell Size is : {mvc_Epilithelial}")

mvc_BareNuclei = new_data['Bare Nuclei'].isna().sum()
print(f"missing values in Bare Nuclei is : {mvc_BareNuclei}")

mvc_Chromatin = new_data['Bland Chromatin'].isna().sum()
print(f"missing values in Bland Chromatin is : {mvc_Chromatin}")

mvc_nucleoli = new_data['Normal Nucleoli'].isna().sum()
print(f"missing values in Normal Nucleoli is : {mvc_nucleoli}")

mvc_mitoses = new_data['Mitoses'].isna().sum()
print(f"missing values in Mitoses is : {mvc_mitoses}")

mvc_Class = new_data['Class'].isna().sum()
print(f"missing values in Class is : {mvc_Class}")


#7) Discard the data points that contain missing values. 

cleaned_data = new_data.dropna()
# print(cleaned_data)

# personal observation : from task 5 i noticed 'Bare Nuclei' attribute contained 16 missing values
# after cleaning the data, i can cinclude there is no missing values because size decreased from (699,10) to (683,10) by 16



#8) boxplot to identify the columns in the table that contain outliers

cleaned_data.boxplot(column=[
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses'], figsize=(20, 20))
plt.show()

# answer : the boxplot shows that 5 of the attributes which are :
#    1. Marginal Adhesion
#    2. Single Epithelial Cell Size
#    3. Bland Chromatin
#    4. Normal Nucleoli
#    5. Mitoses
# the above attirbutes contain outliers


#9) Check for duplicate instances

dups = cleaned_data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))


#10) Drop the row duplicates
print()
print('Number of rows before discarding duplicates = %d' % (cleaned_data.shape[0]))
cleaned_data = cleaned_data.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (cleaned_data.shape[0]))



#11) Plot a 10-bin histogram of the attribute values 'Clump Thickness' distribution

cleaned_data['Clump Thickness'].hist(bins=10,edgecolor='black')


plt.show()


# 12) Discretize the Clump Thickness' attribute into 4 bins of equal width.

cleaned_data['Discretized_ClumpThickness'] = pd.cut(cleaned_data['Clump Thickness'], bins=4 )
print(cleaned_data['Discretized_ClumpThickness'].value_counts(sort=False))
#   range :
#   Discretized_ClumpThickness
#   (0.991, 3.25]    131
#   (3.25, 5.5]      140
#   (5.5, 7.75]       52
#   (7.75, 10.0]     126
#   Name: count, dtype: int64


# 13)  Randomly select 1% of the data without replacement, and save these samples into a new variable. 
sample = cleaned_data.sample(frac=0.01, replace=False, random_state=1)
sample


