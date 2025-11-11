import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier #imported this to construct NearestNeighbors classifier
from sklearn.model_selection import cross_val_score #imported this for cross_val_score calculation 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix #imported this for confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay




# 2) Reading the dataset located here 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
# 3) Assign new headers to the DataFrame
data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']

# 4) Drop the 'Sample code number' attribute 
data = data.drop(['Sample code number'],axis=1)

### Missing Values ###
# 5)Convert the '?' to NaN
data = data.replace('?',np.NaN)

# 6) Count the number of missing values in each attribute of the data.
print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))
    
# 7) Discard the data points that contain missing values
data = data.dropna()


### Outliers ### 
# 8)Drawing a boxplot to identify the columns in the table that contain outliers 
# The 'Bare Nuclei' attribute is a string. Convert it to a numerical attribute.
data.iloc[:,5]=pd.to_numeric(data.iloc[:,5])


#PART 1 : 
    #Continue with the preprocessing:
        
#separate the features from the target class, standardize the features. 
    
X = data.drop(columns=['Class']) # removing class column from data and storing remaining feature colums in X 
Y = data['Class'] # Y stores class column 
    
    
scaler = StandardScaler()
standardized_x = scaler.fit_transform(X)

# Modify the target values: from the description, the malignant class labels are indicated with the value 4. The ‘benign’ labels are indicated with the value‘2’. Replace (or ‘map’) the values such that the label ‘4’ becomes the integer ‘1’, and thelabel ‘2’ becomes the integer ‘0’. 
Y = Y.map({4: 1,2: 0})


# PART 2.5 
# construct NearestNeighbors classifier with default value of number of neighbours 5 
nearest_neighbor_classifier = KNeighborsClassifier(n_neighbors=5)

#PART 2.6
# Compute and print out the averages of :
    # the accuracies, 
    # f1-scores, 
    # precision 
    #recall measurements of the nearest neighbor classifier, using 10-fold cross validation. 


# Perform 10-fold cross-validation for accuracy

accuracy_scores = cross_val_score(nearest_neighbor_classifier, standardized_x, Y, cv=10, scoring='accuracy')
f1_scores = cross_val_score(nearest_neighbor_classifier, standardized_x, Y, cv=10, scoring='f1') 
precision_scores = cross_val_score(nearest_neighbor_classifier, standardized_x, Y, cv=10,scoring='precision') 
recall_scores = cross_val_score(nearest_neighbor_classifier, standardized_x, Y, cv=10, scoring='recall') 


#computing average of each of them : 
average_accuracy = np.mean(accuracy_scores)
average_f1_score = np.mean(f1_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
    

#print :
print(f'the average of accuracy score : {average_accuracy:.2f}')
print(f'the average of f1 scores : {average_f1_score:.2f}')
print(f'the average of precision : {average_precision:.2f}')
print(f'the average of recall : {average_recall:.2f}')


#PART 2.7 confusion matrix 

# step 1. creating a training and test set from my preprocessed data
X_train, X_test, Y_train, Y_test = train_test_split(standardized_x, Y, test_size=0.3,random_state=1)

# step 2. train the nearest neighbor classifier on the training set
nearest_neighbor_classifier.fit(X_train,Y_train) 

# step 3. predict the labels of the test set using the trained classifier 
Y_prediction = nearest_neighbor_classifier.predict(X_test)

# step 4. summarize the prediction result using confusion matrix 
confusionmatrix = confusion_matrix(Y_test,Y_prediction)


# step 5. display the confusion matrix : 
matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=nearest_neighbor_classifier.classes_)
matrix_display.plot()
plt.show()




# you may need to close psyder, open it as an administrator, and
# downgrade your matplotlib version:
# pip install matplotlib==3.2.0
boxplot=data.boxplot(figsize=(20,3))

#plt.boxplot(data)
#plt.show()

### Duplicate Data ### 
# 9) Check for duplicate instances.
dups = data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))

# 10) Drop row duplicates
print('Number of rows before discarding duplicates = %d' % (data.shape[0]))
data = data.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (data.shape[0]))

### Discretization ### 
# 11) Plot a 10-bin histogram of the attribute values 'Clump Thickness' distribution
data['Clump Thickness'].hist(bins=10)
#plt.show()

# 12)Discretize the Clump Thickness' attribute into 4 bins of equal width.
data['Clump Thickness'] = pd.cut(data['Clump Thickness'], 4)
data['Clump Thickness'].value_counts(sort=False)

### Sampling ### 
# 13) Randomly select 1% of the data without replacement. The random_state argument of the function specifies the seed value of the random number generator.
sample = data.sample(frac=0.01, replace=False, random_state=1)
sample