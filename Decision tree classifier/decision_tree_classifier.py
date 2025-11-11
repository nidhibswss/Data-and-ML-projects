
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score



# 1) Read the vertebrate.csv data
data = pd.read_csv("vertebrate.csv")
#print(data)


# 2) The number of records is limited. Convert the data into a binary classification: mammals versus non-mammals
# Hint: ['fishes','birds','amphibians','reptiles'] are considered 'non-mammals'

data["Class"] = data["Class"].replace(["fishes", "reptiles", "amphibians","birds"],"non-mammals")
#print(data) -- shows a new table with above words replaced



# 3)  We want to classify animals based on the attributes: Warm-blooded,Gives Birth,Aquatic Creature,Aerial Creature,Has Legs,Hibernates
# For training, keep only the attributes of interest, and seperate the target class from the class attributes


Y = data["Class"] # (dependent var) setting Y as a targett for training mammal vs non mammal based on X
X = data.drop(["Name","Class"],axis=1) # X is independent variable containing remaining columns after dropping Name and Class


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)


# 4) Create a decision tree classifier object. The impurity measure should be based on entropy.
#                Constrain the generated tree with a maximum depth of 3

model = DecisionTreeClassifier(criterion='entropy',max_depth=3)

# 5) Train the classifier
model = model.fit(X,Y)


# 6)  Suppose we have the following data
testData = [['lizard',0,0,0,0,1,1,'non-mammals'],
           ['monotreme',1,0,0,0,1,1,'mammals'],
           ['dove',1,0,0,1,1,0,'non-mammals'],
           ['whale',1,1,1,0,0,0,'mammals']]


testData = pd.DataFrame(testData, columns=data.columns)

# Prepare the test data and apply the decision tree to classify the test records.
# Extract the class attributes and target class from 'testData'

Ynew = testData["Class"]
Xnew = testData.drop(["Name","Class"],axis=1)


Y_prediction = model.predict(Xnew)

testData["predicted_class"] = Y_prediction 

result = testData[["Name", "Class", "predicted_class"]]

print(result)
# Hint: The classifier should correctly label the vertabrae of 'testData' except for the monotreme


# 7) Compute and print out the accuracy of the classifier on 'testData'


accuracy = accuracy_score(Ynew, Y_prediction) 

print()
print(accuracy)
print()
print(f"Accuracy of the classifier on testData : {accuracy * 100:.2f}%")

# 8) Plot your decision tree

feature_name =["Warm-blooded","Gives Birth","Aquatic Creature","Aerial Creature","Has Legs","Hibernates"]
class_name = ["mammals", "non-mammals"]

tree.plot_tree(model,feature_names=feature_name,class_names=class_name)
plt.title("decision tree model")
plt.show()
