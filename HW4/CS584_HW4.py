# Jason Yeoh
# A20457826 
# CS584 Spring 2020

import itertools
import numpy
import pandas as pd
import scipy
import sympy 
import sklearn.naive_bayes as naive_bayes
import statsmodels.api as stats
import warnings
warnings.filterwarnings('ignore')

# Load dataset
purchase = pd.read_csv("Purchase_Likelihood.csv")
# purchase.head()

## 
## QUESTION 1
##

# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


# 
y = purchase['insurance'].astype('category')

x1 = pd.get_dummies(purchase[['group_size']].astype('category'))
x2 = pd.get_dummies(purchase[['homeowner']].astype('category'))
x3 = pd.get_dummies(purchase[['married_couple']].astype('category'))

designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')


#Intercept + group_size
designX = stats.add_constant(x1, prepend=True)
LLK1, DF1, fullParams1 = build_mnlogit(designX, y, debug = 'N')
testDev = 2 * (LLK1 - LLK0)
testDF = DF1 - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + group_size + homeowner
designX = x1
designX = designX.join(x2)
designX = stats.add_constant(designX, prepend=True)
LLK2, DF2, fullParams2 = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK2 - LLK1)
testDF = DF2 - DF1
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + group_size + homeowner + married_couple
designX = x1
designX = designX.join(x2)
designX = designX.join(x3)
designX = stats.add_constant(designX, prepend=True)
LLK3, DF3, fullParams3 = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK3 - LLK2)
testDF = DF3 - DF2
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + group_size + homeowner + married_couple + (group_size * homeowner)
designX = x1
designX = designX.join(x2)
designX = designX.join(x3)
x12 = create_interaction(x1, x2)
designX = designX.join(x12)
designX = stats.add_constant(designX, prepend=True)
LLK4, DF4, fullParams4 = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK4 - LLK3)
testDF = DF4 - DF3
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + group_size + homeowner + married_couple + (group_size * homeowner) + (group_size * married_couple)
designX = x1
designX = designX.join(x2)
designX = designX.join(x3)
x12 = create_interaction(x1, x2)
designX = designX.join(x12)
x13 = create_interaction(x1, x3)
designX = designX.join(x13)

designX = stats.add_constant(designX, prepend=True)
LLK5, DF5, fullParams5 = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK5 - LLK4)
testDF = DF5 - DF4
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

#Intercept + group_size + homeowner + married_couple + (group_size * homeowner) + (group_size * married_couple)
# + (homeowner * married_couple)
designX = x1
designX = designX.join(x2)
designX = designX.join(x3)
x12 = create_interaction(x1, x2)
designX = designX.join(x12)
x13 = create_interaction(x1, x3)
designX = designX.join(x13)
x23 = create_interaction(x2, x3)
designX = designX.join(x23)

designX = stats.add_constant(designX, prepend=True)
LLK6, DF6, fullParams6 = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK6 - LLK5)
testDF = DF6 - DF5
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Check for non-aliased columns
fullParams6[fullParams6['0_y'] == 0.0]

# Negative log10 values
print(-numpy.log10(1.4597001210408566e-295))


#
# QUESTION 2
#
x1 = pd.get_dummies(purchase[['group_size']].astype('category'))
x2 = pd.get_dummies(purchase[['homeowner']].astype('category'))
x3 = pd.get_dummies(purchase[['married_couple']].astype('category'))
y = pd.get_dummies(purchase[['insurance']].astype('category'))

X = x1
X = X.join(x2)
X = X.join(x3)
X = X.join(create_interaction(x1, x2))
X = X.join(create_interaction(x1, x3))
X = X.join(create_interaction(x2, x3))
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

thisFit.predict(X)

lst=([1,0,0],[1,0,1],[1,1,0],[1,1,1],
[2,0,0],[2,0,1],[2,1,0],[2,1,1],
[3,0,0],[3,0,1],[3,1,0],[3,1,1],
[4,0,0],[4,0,1],[4,1,0],[4,1,1])

catPred = ['group_size', 'homeowner', 'married_couple']

xTest = pd.DataFrame(lst, columns=catPred)
x1Test = pd.get_dummies(xTest[['group_size']].astype('category'))
x2Test = pd.get_dummies(xTest[['homeowner']].astype('category'))
x3Test = pd.get_dummies(xTest[['married_couple']].astype('category'))

X_test = x1Test
X_test = X_test.join(x2Test)
X_test = X_test.join(x3Test)
X_test = X_test.join(create_interaction(x1Test, x2Test))
X_test = X_test.join(create_interaction(x1Test, x3Test))
X_test = X_test.join(create_interaction(x2Test, x3Test))
X_test = stats.add_constant(X_test, prepend=True)
X_test
yTest_predProb = thisFit.predict(X_test)
yTest_predProb.columns = ['P(insurance=0)', 'P(insurance=1)', 'P(insurance=2)']
test = pd.concat([xTest, yTest_predProb], axis = 1)
print(test)


#2B: P(insurance=1)/P(insurance=0)
print('Max Odd Value: P(insurance=1)/P(insurance=0)\n')
print('{}'.format(test['P(insurance=1)']/test['P(insurance=0)']))

# [7]: 2 1 1

#2C: 
P_G3 = purchase[purchase['group_size'] == 3].groupby('insurance').size()[2] / purchase[purchase['group_size'] == 3].groupby('insurance').size()[0]
P_G1 = purchase[purchase['group_size'] == 1].groupby('insurance').size()[2] / purchase[purchase['group_size'] == 1].groupby('insurance').size()[0]
P_G = P_G3/P_G1
print(P_G)

#2D:
P_H1 = purchase[purchase['homeowner'] == 1].groupby('insurance').size()[0] / purchase[purchase['homeowner'] == 1].groupby('insurance').size()[1]
P_H0 = purchase[purchase['homeowner'] == 0].groupby('insurance').size()[0] / purchase[purchase['homeowner'] == 0].groupby('insurance').size()[1]
P_H = P_H1/P_H0
print(P_H)


#
# QUESTION 3
#
purchase.groupby('insurance').count()['group_size']
print(143691/len(purchase))
print(426067/len(purchase))
print(95491/len(purchase))

purchase.groupby(['insurance', 'group_size']).count()

purchase.groupby(['insurance', 'homeowner']).count()

purchase.groupby(['insurance', 'married_couple']).count()

# Define a function that performs the Chi-square test
def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    if (debug == 'Y'):
        print('Observed Count:\n', obsCount)
        print('Column Total:\n', cTotal)
        print('Row Total:\n', rTotal)
        print('Overall Total:\n', nTotal)
        print('Expected Count:\n', expCount)
        print('\n')
       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = numpy.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)


catPred = ['group_size', 'homeowner', 'married_couple']

testResult = pd.DataFrame(index = catPred,
                              columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in catPred:
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(purchase[pred], purchase['insurance'], debug = 'Y')
    testResult.loc[pred] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]

testResult.sort_values(by='Significance', ascending=False)

X = purchase[catPred]
y = purchase['insurance']
classifier = naive_bayes.MultinomialNB().fit(X, y)

print('Class Count:\n', classifier.class_count_)
print('Log Class Probability:\n', classifier.class_log_prior_ )
print('Feature Count (after adding alpha):\n', classifier.feature_count_)
print('Log Feature Probability:\n', classifier.feature_log_prob_)

predProb = classifier.predict_proba(X)
print('Predicted Conditional Probability (Training):', predProb)

lst=([1,0,0],[1,0,1],[1,1,0],[1,1,1],
[2,0,0],[2,0,1],[2,1,0],[2,1,1],
[3,0,0],[3,0,1],[3,1,0],[3,1,1],
[4,0,0],[4,0,1],[4,1,0],[4,1,1])

xTest = pd.DataFrame(lst, columns=catPred)
yTest_predProb = pd.DataFrame(classifier.predict_proba(xTest), columns = ['P(insurance=0)', 'P(insurance=1)', 'P(insurance=2)'])
yTest_score = pd.concat([xTest, yTest_predProb], axis = 1)
print(yTest_score)


(yTest_score['P(insurance=1)']/yTest_score['P(insurance=0)']).sort_values()


