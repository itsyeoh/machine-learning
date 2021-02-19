# Jason Yeoh
# CS584 HW5
# Spring 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")

# Load dataset
spiral = pd.read_csv('SpiralWithCluster.csv')

# 1A. Count the percentage of spectral cluster equal to 1.
spiral[spiral['SpectralCluster'] == 1].count()/spiral.count()

X = spiral[['x', 'y']]
y = spiral['SpectralCluster']

# 1B - 1C
def build_NN(activation): 
    result = pd.DataFrame(columns = ['Activation function', 'nLayer', 'nHiddenNeuron', 'Iterations', 'Loss', 'Misclassification', 'Output Layer'])
    
    for nLayer in range(1, 6):
        for nHiddenNeuron in range(1, 11):
            nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = activation, verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20200408)
            nnObj.fit(X, y)
            y_pred = nnObj.predict(X)
            
            loss = nnObj.loss_
            misclass = 1.0 - accuracy_score(y, y_pred.round())
            iters = nnObj.n_iter_
            
            result = result.append(pd.DataFrame([[activation, nLayer, nHiddenNeuron, iters, loss, misclass, nnObj.out_activation_]],
                                                columns = ['Activation function', 'nLayer', 'nHiddenNeuron',  'Iterations', 'Loss', 'Misclassification', 'Output Layer']))
    
    lowestLoss = result['Loss'] == result['Loss'].min()
    lowestMisclass = result['Misclassification'] == result['Misclassification'].min()
    
    return result[lowestLoss] 
#     return result[lowestLoss & lowestMisclass])

nn_grid = []

for activation in ['identity', 'logistic', 'relu', 'tanh']:
    nn_grid.append( build_NN(activation) )
    
nn_grid = pd.concat(nn_grid)
nn_grid


# 1D:
lowestLoss = nn_grid['Loss'] == nn_grid['Loss'].min()
lowestMisclass = nn_grid['Misclassification'] == nn_grid['Misclassification'].min()

nn_grid[lowestLoss & lowestMisclass]

# 1E:
nnObj = nn.MLPClassifier(hidden_layer_sizes = (10,)*4,
                            activation = 'relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20200408)

nnObj.fit(X, y)
y_pred = nnObj.predict(X).round()

spiral_nn = pd.concat([X, pd.DataFrame(y_pred, columns=['SpectralCluster'])], axis=1)
spiral_nn

sns.set(rc={'figure.figsize':(10, 7)})
sns.scatterplot(x="x", y="y", hue="SpectralCluster", data=spiral_nn).set_title('Optimal MLP (10 neurons, 4 layers)')

# 1F:
y_predProb = nnObj.predict_proba(X)

print('Count: {}'.format(y_predProb[:,1].sum()))
print('Mean: {}'.format(y_predProb[:,1].mean()))
print('std: {}'.format(y_predProb[:,1].std()))


# 2A:
svm_model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20200408, max_iter = -1)
thisFit = svm_model.fit(X, y) 
y_pred = thisFit.predict(X)

spiral['_PredictedClass_'] = y_pred

svm_Mean = spiral.groupby('_PredictedClass_').mean()
# print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# 2B:
print('Misclassification Rate = ', 1.0 - accuracy_score(y, y_pred))

# 2C:
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-3, 3)
yy = a * xx - (thisFit.intercept_[0]) / w[1]

b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

cc = thisFit.support_vectors_

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = spiral[spiral['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'X', s = 100)
plt.plot(xx, yy, color = 'black', linestyle = '-')
plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machine')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# 2D:
# Convert to the polar coordinates
spiral['radius'] = np.sqrt(spiral['x']**2 + spiral['y']**2)
spiral['theta'] = np.arctan2(spiral['y'], spiral['x'])

def customArcTan(z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

spiral['theta'] = spiral['theta'].apply(customArcTan)

carray = ['red', 'blue']
# Scatterplot that uses prior information of the grouping variable
plt.figure(figsize=(10,10))
for i in range(2):
    subData = spiral[spiral['SpectralCluster'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
# plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Prior Group Information (Theta v. Radius)')
plt.xlabel('Radius')
plt.ylabel('Theta (angle in radians)')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# 2E:
X_new = spiral[['radius', 'theta']]

dbscan = DBSCAN().fit(X_new)
spiral['Group'] = dbscan.labels_ + 1

carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = spiral[spiral['Group'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Prior Group Information (Theta v. Radius)')
plt.xlabel('Radius')
plt.ylabel('Theta (angle in radians)')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# 2F:
filter0 = (spiral['Group'] == 0) | (spiral['Group'] == 1)
filter1 = (spiral['Group'] == 1) | (spiral['Group'] == 2)
filter2 = (spiral['Group'] == 2) | (spiral['Group'] == 3)

SVM0_X, SVM0_y = spiral[filter0][['radius', 'theta']], spiral[filter0]['Group']
SVM1_X, SVM1_y = spiral[filter1][['radius', 'theta']], spiral[filter1]['Group']
SVM2_X, SVM2_y = spiral[filter2][['radius', 'theta']], spiral[filter2]['Group']

SVM0_model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr', 
                     random_state = 20200408, max_iter = -1).fit(SVM0_X, SVM0_y)
SVM1_model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr', 
                     random_state = 20200408, max_iter = -1).fit(SVM1_X, SVM1_y)
SVM2_model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr', 
                     random_state = 20200408, max_iter = -1).fit(SVM2_X, SVM2_y)

print('SVM0: Intercept = {}, Coefficients = {}'.format( SVM0_model.intercept_, SVM0_model.coef_))
print('SVM1: Intercept = {}, Coefficients = {}'.format( SVM1_model.intercept_, SVM1_model.coef_))


# 2G:
# Setting up hyperplanes for the three SVM models
w0 = SVM0_model.coef_[0]
a0 = -w0[0] / w0[1]
xx0 = np.linspace(0, 5)
yy0 = a0 * xx0 - (SVM0_model.intercept_[0]) / w0[1]

w1 = SVM1_model.coef_[0]
a1 = -w1[0] / w1[1]
xx1 = np.linspace(0, 5)
yy1 = a1 * xx1 - (SVM1_model.intercept_[0]) / w1[1]

w2 = SVM2_model.coef_[0]
a2 = -w2[0] / w2[1]
xx2 = np.linspace(0, 5)
yy2 = a2 * xx2 - (SVM2_model.intercept_[0]) / w2[1]


b0 = SVM0_model.support_vectors_[0]
yy0_down = a0 * xx0 + (b0[1] - a0 * b0[0])
b0 = SVM0_model.support_vectors_[-1]
yy0_up = a0 * xx0 + (b0[1] - a0 * b0[0])

b1 = SVM1_model.support_vectors_[0]
yy1_down = a1 * xx1 + (b1[1] - a1 * b1[0])
b1 = SVM1_model.support_vectors_[-1]
yy1_up = a1 * xx1 + (b1[1] - a1 * b1[0])

b2 = SVM2_model.support_vectors_[0]
yy2_down = a2 * xx2 + (b2[1] - a2 * b2[0])
b2 = SVM2_model.support_vectors_[-1]
yy2_up = a2 * xx2 + (b2[1] - a2 * b2[0])


# cc = thisFit.support_vectors_

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
plt.figure(figsize=(10,10))
for i in range(4):
    subData = spiral[spiral['Group'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.plot(xx0, yy0, color = 'black', linestyle = '-')
plt.plot(xx1, yy1, color = 'brown', linestyle = '-')
plt.plot(xx2, yy2, color = 'blue', linestyle = '-')

plt.plot(xx0, yy0_up, color = 'black', linestyle = '--')
plt.plot(xx0, yy0_down, color = 'black', linestyle = '--')

plt.plot(xx1, yy1_up, color = 'brown', linestyle = '--')
plt.plot(xx1, yy1_down, color = 'brown', linestyle = '--')

plt.plot(xx2, yy2_up, color = 'blue', linestyle = '--')
plt.plot(xx2, yy2_down, color = 'blue', linestyle = '--')


plt.grid(True)
plt.title('Kernel Trick with 3 SVM hyperplanes')
plt.xlabel('radius')
plt.ylabel('theta')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
print('SVM2: Intercept = {}, Coefficients = {}'.format( SVM2_model.intercept_, SVM2_model.coef_))


# 2H:
h0_xx = xx0 * np.cos(yy0)
h0_yy = xx0 * np.sin(yy0)
h1_xx = xx1 * np.cos(yy1)
h1_yy = xx1 * np.sin(yy1)
h2_xx = xx2 * np.cos(yy2)
h2_yy = xx2 * np.sin(yy2)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = spiral[spiral['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
# plt.plot(h0_xx, h0_yy, color = 'black', linestyle = '--')
plt.plot(h1_xx, h1_yy, color = 'green', linestyle = '--')
plt.plot(h2_xx, h2_yy, color = 'blue', linestyle = '--')
plt.grid(True)
plt.title('SVM with Hypercurve Segments')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()