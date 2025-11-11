import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

"""
You are provided a dataset of 1500 labeled two-dimensional records.
The file ‘Xdata.npy’ has the attribute values of the 1500 records,
while the file ‘Ydata.npy’ has the corresponding class labels of these 1500 records.
The assigned labels consist of 0 or 1 instances. You are given this dataset without further explanation about the data, its provenance,
or any description of the relevant characteristics.

You are tasked to do your level-best at developing and training a decision tree that will accurately classify unseen records.
You will grow several decision trees with different depths. As the model becomes more complex, you will notice that the training accuracy will improve.
Notwithstanding, the test accuracy will initially improve, up to a maximum depth (that you will have to determine) before decreasing due to model overfitting.
Write a Python script
"""


# 1) Load the data (Y is the class labels of X)
X = np.load('Xdata.npy')
Y = np.load('Ydata.npy')

# 2) Split the training and test data as follows: 
    # 80% of the data for training and 20% for testing. 
    
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
    # Preserve the percentage of samples for each class using the argument 'stratify'. 
    # Use the argument 'random_state' so that the data splitting is the same everytime your code is run.


# 3) Test the fit of different decision tree depths 
# Instruction 1: Use the range function to create different depths options, ranging from 1 to 50, for the decision trees
# Instruction 2: As you iterate through the different tree depth options, please:
    # create a new decision tree using the 'max_depth' argument
    # train your tree
    # apply your tree to predict the 'training' and then the 'test' labels
    # compute the training accuracy
    # compute the test accuracy
    # save the training & testing accuracies and tree depth, so that you can use them in the next steps
  
    
#creating empty lists to store the values of training and testing accuracies and tree depth later
train_list = []
test_list = []

# create a new decision tree using the 'max_depth' argument
# train your tree
for depth in range(1,51):
    model = tree.DecisionTreeClassifier(max_depth=depth,random_state=1)
    model.fit(X_train,Y_train)
    
# apply your tree to predict the 'training' and then the 'test' labels
    Y_prediction_train = model.predict(X_train)
    Y_prediction_test = model.predict(X_test)
    
# compute the training accuracy
    training_accuracy = accuracy_score(Y_train,Y_prediction_train)
    
# compute the test accuracy
    testing_accuracy = accuracy_score(Y_test,Y_prediction_test)
    
# save the training & testing accuracies and tree depth, so that you can use them in the next steps  
#storing training and testing accuracies and tree depth in the list 
    train_list.append(training_accuracy)
    test_list.append(testing_accuracy)
    
   # print(train_list)
   # print(test_list)

range_ = range(1,51)
    
    
    
# 4) Plot of training and test accuracies vs the tree depths  
plt.plot(range_,train_list,'rv-',range_,test_list,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.show()

# 5) Fill out the following blank:
# Model overfitting happens when the tree depth is greater than 10, approximately.
print("Model overfitting happens when the tree depth is greater than 10, approximately.")
