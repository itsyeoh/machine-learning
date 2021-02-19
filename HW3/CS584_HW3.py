# Jason Yeoh
# A20457826
# CS584 Spring 2020

# Import Libraries
import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder
import math 
from itertools import combinations 
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# Load dataset
history = pd.read_csv('claim_history.csv')
history = history[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()

education_mapper = {'Doctors': 4 , 'Masters': 3, 'Bachelors': 2, 'High School': 1, 'Below High School': 0}
history['MAPPED_EDUCATION'] = history['EDUCATION'].replace(education_mapper)

# Question 1
train, test = model_selection.train_test_split(history, train_size=0.75, random_state=60616, stratify=history['CAR_USE'])

#1A:
num_private, num_commercial = pd.value_counts(train['CAR_USE'])
p_private_train = num_private/(num_commercial + num_private)
p_commercial_train = num_commercial/(num_commercial + num_private)

print(num_private, num_commercial)
print(p_private_train, p_commercial_train)

#1B:
num_private_test, num_commercial_test = pd.value_counts(test['CAR_USE'])
p_private_test = num_private_test/(num_commercial_test + num_private_test)
p_commercial_test = num_commercial_test/(num_commercial_test + num_private_test)

print(num_private_test, num_commercial_test)
print(p_private_test, p_commercial_test)

#1C:
p_train_commercial = (p_commercial_train * 0.75)/(p_commercial_train * 0.75 + p_commercial_test * 0.25)
print('P(Commercial|Train) = {}'.format(p_train_commercial))

#1D:
p_test_private = (p_private_test * 0.25)/(p_private_train * 0.75 + p_private_test * 0.25)
print('P(Commercial|Train) = {}'.format(p_test_private))

# Question 2
def EntropyIntervalSplit(inData, split):
    dataTable = inData
    dataTable['LE_Split'] = (dataTable.iloc[:,0] <= split)

    crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
    nRows = crossTable.shape[0]
    nColumns = crossTable.shape[1]

    tableEntropy = 0
    for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * np.log2(proportion)
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
    tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]

    return(tableEntropy)

def EntropyNominalSplit(inData, split):
    dataTable = inData
    # Divide the inDAta into two branches based on the splitting list parameter
    dataTable['LE_Split'] = list(map(lambda x: True if x in split else False, dataTable.iloc[:,0]))

    crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
    nRows = crossTable.shape[0]
    nColumns = crossTable.shape[1]

    tableEntropy = 0
    for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * np.log2(proportion)
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
    tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]

    return(tableEntropy)

def getOptimalNominalSplit(inData, splits):
    minEntropy = 1.0
    minCombination = []
    length = len(splits) + 1
    
    # Loop through all possible splitting combinations 
    for i in range(1, length):
        for comb in list(combinations(splits, i)):
            currCombination = list(comb)
            currEntropy = EntropyNominalSplit(inData, currCombination)

            if currEntropy < minEntropy:
                minEntropy = currEntropy
                minCombination = currCombination
    
    return (minEntropy, minCombination, list(set(splits)-set(minCombination)))

def getOptimalIntervalSplit(inData, splits):
    minEntropy = 1.0
    minSplit = -1.0
    length = len(splits)
    
    # Loop through all possible split values
    for i in range(length):
        currEntropy = EntropyIntervalSplit(inData, splits[i])
        
        if currEntropy < minEntropy:
            minEntropy = currEntropy
            minSplit = splits[i]
    
    return (minEntropy, minSplit) 

def getEntropy(p_private, p_commercial):
    return -(p_private * np.log2(p_private) + p_commercial * np.log2(p_commercial))

def printCounts(data, decisionRule):
    frequency_table = pd.value_counts(data['CAR_USE'])
    num_p = frequency_table['Private']
    num_c = frequency_table['Commercial']
    nums = num_p + num_c
    entropy =  getEntropy(num_p/nums, num_c/nums)
    
    print('Decision Rule: {}'.format(decisionRule))
    print('Private Count: {}'.format(num_p))
    print('Commercial Count: {}'.format(num_c))
    print('Total Count: {}'.format(nums))
    print('Entropy: {}\n'.format(entropy))   
    
def getProbabilities(data):
    frequency_table = pd.value_counts(data['CAR_USE'])
    num_p = frequency_table['Private']
    num_c = frequency_table['Commercial']
    nums = num_p + num_c
    return (num_p/nums, num_c/nums)


#2A:
root_entropy = getEntropy(p_private_train, p_commercial_train)
print('Root Entropy: {}'.format(root_entropy))

#2B:
occupations = ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional',
              'Student', 'Unknown']
car_types = ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van']
mapped_education_splits = [0.5, 1.5, 2.5, 3.5, 4.5]

print( getOptimalNominalSplit(train[['CAR_TYPE', 'CAR_USE']], car_types) )
print( getOptimalNominalSplit(train[['OCCUPATION', 'CAR_USE']], occupations) )
print( getOptimalIntervalSplit(train[['MAPPED_EDUCATION', 'CAR_USE']], mapped_education_splits) )

#2C:
occupations_left = ['Blue Collar', 'Student', 'Unknown']
occupations_right = ['Lawyer', 'Manager', 'Home Maker', 'Professional', 'Doctor', 'Clerical']

left_data = train[train['OCCUPATION'].isin(occupations_left)]
right_data = train[train['OCCUPATION'].isin(occupations_right)]

# LEFT BRANCH
print( getOptimalNominalSplit(left_data[['OCCUPATION', 'CAR_USE']], occupations_left) )
print( getOptimalNominalSplit(left_data[['CAR_TYPE', 'CAR_USE']], car_types) )
print( getOptimalIntervalSplit(left_data[['MAPPED_EDUCATION', 'CAR_USE']], mapped_education_splits) )
print('\n')

# RIGHT BRANCH
print( getOptimalNominalSplit(right_data[['OCCUPATION', 'CAR_USE']], occupations_right) )
print( getOptimalNominalSplit(right_data[['CAR_TYPE', 'CAR_USE']], car_types) )
print( getOptimalIntervalSplit(right_data[['MAPPED_EDUCATION', 'CAR_USE']], mapped_education_splits) )

#2D:
carTypes_left = ['Minivan', 'SUV', 'Sports Car']
carTypes_right = ['Panel Truck', 'Van', 'Pickup']

ll_data = left_data[left_data['MAPPED_EDUCATION'] <= 0.5]
lr_data = left_data[left_data['MAPPED_EDUCATION'] > 0.5]
rl_data = right_data[right_data['CAR_TYPE'].isin(carTypes_left)]
rr_data = right_data[right_data['CAR_TYPE'].isin(carTypes_right)]

printCounts(ll_data, 'Education <= Below High School')
printCounts(lr_data, 'Education > Below High School')
printCounts(rl_data, 'CarType = [Minivan, SUV, Sports Car]')
printCounts(rr_data, 'CarType = [Pickup, Van, Panel Truck]')

#2E:
p_priv_ll, p_comm_ll = getProbabilities(ll_data)
p_priv_lr, p_comm_lr = getProbabilities(lr_data)
p_priv_rl, p_comm_rl = getProbabilities(rl_data)
p_priv_rr, p_comm_rr = getProbabilities(rr_data)

# revised_data = pd.concat([ll_data, lr_data, rl_data, rr_data], axis=0)
def getPredictedProbability(data):
    if data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if data['MAPPED_EDUCATION'] <= 0.5:
            return p_comm_ll
        else:
            return p_comm_lr
    else:
        if data['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return p_comm_rl
        else:
            return p_comm_rr

train['PROBABILITY'] = train.iloc[:].apply(lambda x: getPredictedProbability(x), axis=1)
threshold = num_commercial/(num_private + num_commercial)

# Get misclassification rate
Y_train = np.array(train['CAR_USE'])
predProbY_train = np.array(train['PROBABILITY'])
predY_train = np.empty_like(Y_train)
nYt = Y_train.shape[0]

for i in range(nYt):
    if predProbY_train[i] > threshold:
        predY_train[i] = 'Commercial'
    else:
        predY_train[i] = 'Private'
        
fpr, tpr, thresholds = metrics.roc_curve(Y_train, predProbY_train, pos_label = 'Commercial')

cutoff = np.where(thresholds > 1.0, np.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()

print(thresholds)
print(tpr-fpr)

# Question 3
#3A:
# Event = Commercial, Non-event = Private
threshold = num_commercial/(num_private + num_commercial)

def goodmanKruskalsGamma(data):
    event = list(data[data['CAR_USE'] == 'Commercial']['PROBABILITY'])
    non_event = list(data[data['CAR_USE'] == 'Private']['PROBABILITY'])

    C = 0
    D = 0
    T = 0

    for e in event:
        for ne in non_event:
            if e > ne:
                C += 1
            elif e == ne:
                T += 1
            else:
                D += 1

    C, D, T
    return((C-D)/(C+D))

test['PROBABILITY'] = test.iloc[:].apply(lambda x: getPredictedProbability(x), axis=1)

# Get misclassification rate
Y = np.array(test['CAR_USE'])
predProbY = np.array(test['PROBABILITY'])
predY = np.empty_like(Y)
nY = Y.shape[0]

for i in range(nY):
    if predProbY[i] > threshold:
        predY[i] = 'Commercial'
    else:
        predY[i] = 'Private'

RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Commercial'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = np.sqrt(RASE/nY)

# Calculate the Root Mean Squared Error
Y_true = 1.0 * np.isin(Y, ['Commercial'])
RMSE = metrics.mean_squared_error(Y_true, predProbY)
RMSE = np.sqrt(RMSE)

AUC = metrics.roc_auc_score(Y_true, predProbY)
Gini = 2 * AUC - 1
accuracy = metrics.accuracy_score(Y, predY)
        
print('                  Accuracy: {:.13f}' .format(accuracy))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
print('          Area Under Curve: {:.13f}' .format(AUC))
print('Root Average Squared Error: {:.13f}' .format(RASE))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE))
print('                      Gini: {:.13f}' .format(Gini))
print('     Goodman-Kruskal Gamma: {:.13f}' .format(goodmanKruskalsGamma(test)))

ks_threshold = 0.53419726
ks_predY = np.empty_like(Y)
for i in range(nY):
    if predProbY[i] > ks_threshold:
        ks_predY[i] = 'Commercial'
    else:
        ks_predY[i] = 'Private'
        
print(' KS-Misclassification Rate: {:.13f}' .format(1.0 - metrics.accuracy_score(Y, ks_predY)))

# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = np.append([0], fpr)
Sensitivity = np.append([0], tpr)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
plt.show()
