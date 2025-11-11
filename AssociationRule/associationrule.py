import pandas as pd
from apyori import apriori


#loading dataset : 
weather_dataset = pd.read_csv("weather.csv")


#Discretize the continuous data : 

# temperature: create 3 equal-width bins,
# where the original temperature values are replaced by the labels 'cool', 'mild' or, 'hot'

weather_dataset['temperature'] = pd.cut(x=weather_dataset['temperature'], bins=3, labels=['cool', 'mild', 'hot'])

# humidity: create 2 equal-width bins, 
# where the original humidity values are replaced by the labels 'normal' or 'high'

weather_dataset['humidity'] = pd.cut(x=weather_dataset['humidity'], bins=2, labels=['normal', 'high'])


# convert the boolean values from the attribute 'windy' to string ushing map method: 

weather_dataset['windy'] = weather_dataset['windy'].map({True: 'yes', False: 'no'})

print(weather_dataset)

# converting dataset to list 
weather_list = weather_dataset.values.tolist()

print()
print(weather_list)

# apriori  function with given support and confidence values
rules = apriori(weather_list, min_support=0.28, min_confidence=0.5)

print()

# printing out generated rules with respective support and confidence values as given

print("Rules with the given support and confidence values :")

for rule in rules:
    print()
    x = rule.ordered_statistics[0]
    print("Rule: If", list(x.items_base), "then", list(x.items_add))
    print("Support value :", rule.support)
    print("Confidence value :", x.confidence)

    
print()
print("-------------------------TESTING #2 -------------------------------------------" )
print()

#testing another support and confidence value (LOWER) :
    
rule_2 = apriori(weather_list, min_support=0.10, min_confidence=0.40)

# printing out generated rules with respective support and confidence levels
print("Rules 2 testing using low support and confidence  :")
for rule in rule_2:
    print()
    x = rule.ordered_statistics[0]
    print("Rule: If", list(x.items_base), "then", list(x.items_add))
    print("Support value :", rule.support)
    print("Confidence value :", x.confidence)

   
print()
print("-------------------------TESTING #3 -------------------------------------------" )
print()

#testing another support and confidence value (HIGHER) :
    
rule_3 = apriori(weather_list, min_support=0222.40, min_confidence=0.70)

# printing out generated rules with respective support and confidence levels
print("Rules 3 testing using high support and confidence :")
for rule in rule_3:
    print()
    x = rule.ordered_statistics[0]
    print("Rule: If", list(x.items_base), "then", list(x.items_add))
    print("Support value :", rule.support)
    print("Confidence value :", x.confidence)


print()
print("OBSERVATION FROM DIFFERENT TESTINGS : ")
print()

print("-------------------------CONCLUSION-------------------------------------------" )

print("After playing around with different values for suppport and confidence in testing2 and testing3 ,along with the original values; \nwe can conclude that the lower their values are, the more rules we will be able to get. \n and the more their values are, the less rules we will get") 