
import pandas as pd
from sklearn.naive_bayes import GaussianNB


#Loading the dataset weather.csv
df = pd.read_csv('weather.csv')
#print(df.columns)

# determining categorical attributes :
    # outlook, windy, play are categorical attributes in our dataset, 
    # so we'll be converting them into dummy variables using panda's function get_dummies 

df = pd.get_dummies(df, columns=["outlook", "windy", "play"], dtype=float)
#print(df.columns)

# **** new printed columns are:Index(['temperature', 'humidity', 'outlook_overcast', 'outlook_rainy',
#       'outlook_sunny', 'windy_False', 'windy_True', 'play_no', 'play_yes'], 



#target attribute play is now split into 2 target attributes - play_no, play_yes
#Droping 'play_no'and separating the features attributes from the target attribute
df.drop(columns=["play_no"], inplace=True)


X = df.drop(columns=["play_yes"])  #feature
Y = df["play_yes"] #target 


#Using the sklearn library to construct and train a Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, Y)


# new day given Outlook = sunny, temp=66, humidity=60, windy=true
# because we converted outlook, windy and play to dummy values, now it is split
# outlook gets split into 3 , windy is split into 2, play is split in 2, temp and humdity stays the same

#rearranging the dataframe according to the serial from step 2 ****
new_day = pd.DataFrame({
    "temperature": [66],  #given temp= 66 for new day
    "humidity": [90],     #given humidity=90 for new day
    "outlook_overcast": [0],  # no overcast, so 0 
    "outlook_rainy": [0],  #no rainy, so 0 
    "outlook_sunny": [1],#given outlook = sunny, so 1 
    "windy_False": [0], #no windy, so 0
    "windy_True": [1]  # given windy=true, so 1 
})
print()

# Get likelihood probabilities

likelihoods = clf.predict_proba(new_day)
print(likelihoods)

print()

play_no = likelihoods[0][0]
play_yes = likelihoods[0][1]

print("Likelihood of play = no:", play_no)
print("Likelihood of play = yes:", play_yes)






