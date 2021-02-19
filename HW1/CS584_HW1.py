# ================================
# Jason Yeoh (A20457826)
# CS584 HW1 (Spring 2020)
# Prof. Lam
# ================================ 

#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

sample = pd.read_csv('NormalSample.csv')
fraud = pd.read_csv('fraud.csv')

#
# QUESTION 1
#
def Q1A():
	print("2(IQR) * N^-(1/3)")

def Q1B():
	a = sample.min()['x']
	b = sample.max()['x']
	print('Min:', a, '\nMax:', b)
	return a, b

def Q1C(): 
	min, max = Q1B()
	a = math.floor(min)
	b = math.ceil(max)
	print('a=', a, 'b=', b)
	return a, b

# h = 0.25, 0.5, 1, 2
def Q1DEFG():
	a, b = Q1C()
	for h in [0.25, 0.5, 1, 2]:
	    fig, axs = plt.subplots(ncols=2, figsize=(10,4))
	    sns.distplot(sample['x'], bins=[a+(h*i) for i in range(int((b-a)/h) + 1)], kde=False, rug=False, ax=axs[0])
	    sns.distplot(sample['x'], bins=[a+(h*i) for i in range(int((b-a)/h) + 1)], kde=True, rug=False, ax=axs[1])
	    axs[0].set(xlabel='x', ylabel='count', title='Histogram (h=%.2f)'%h)
	    axs[1].set(xlabel='x', ylabel='density', title='Histogram with Density Estimator (h=%.2f)'%h)
	    plt.tight_layout()
	    plt.show()
#
# QUESTION 2
#
def Q2A():
	_, _, _, min, q1, median, q3, max = sample.describe()['x']

	iqr = q3-q1
	iqr_low = q1 - 1.5*iqr
	iqr_high = q3 + 1.5*iqr

	print('Min: ', min)
	print('Q1:  ', q1)
	print('Med: ', median)
	print('Q3:  ', q3)
	print('Max: ', max)
	print('=============')
	print('IQR: ', iqr)
	print('1.5*IQR Range: [', iqr_low, '-', iqr_high, ']')
	return iqr_low, iqr_high

def Q2B():
	print( sample.groupby(['group']).describe()['x'], '\n')

	# Group 0
	IQR_g0 = 30.6 - 29.4
	IQR_g0_lo = 29.4 - 1.5*IQR_g0
	IQR_g0_hi = 30.6 + 1.5*IQR_g0

	#Group 1
	IQR_g1 = 32.7 - 31.4
	IQR_g1_lo = 31.4 - 1.5*IQR_g1
	IQR_g1_hi = 32.7 + 1.5*IQR_g1

	print('[Group 0]')
	print('IQR: %.3f' % IQR_g0)
	print('1.5*IQR Range: [%.3f - %.3f]\n' % (IQR_g0_lo, IQR_g0_hi))

	print('[Group 1]')
	print('IQR: %.3f' % IQR_g1)
	print('1.5*IQR Range: [%.3f - %.3f]\n' % (IQR_g1_lo, IQR_g1_hi))
	return IQR_g0_lo, IQR_g0_hi, IQR_g1_lo, IQR_g1_hi

def Q2C():
	print('Yes, because it correctly represents two low outliers (<27.4) and no high outliers (>35.4).')
	sns.boxplot(data=sample['x'])
	plt.show()

def Q2D():
	g0, g1 = [x for _, x in sample.groupby('group')['x']]
	all = sample['x']

	g0 = pd.DataFrame(g0).assign(Group='0')
	g1 = pd.DataFrame(g1).assign(Group='1')
	all = pd.DataFrame(all).assign(Group='all')

	cgrp = pd.concat([all, g0, g1])
	mgrp = pd.melt(cgrp, id_vars=['Group'], var_name=['Number'])

	sns.boxplot(x="Group", y="value", data=mgrp)
	plt.show()

	# Outliers
	iqr_low, iqr_high = Q2A()
	IQR_g0_lo, IQR_g0_hi, IQR_g1_lo, IQR_g1_hi = Q2B()

	print('\n[OUTLIERS FOR THE ENTIRE DATA]')
	print( sample[(sample['x'] < iqr_low) | (sample['x'] > iqr_high)]['x'] )

	print('\n[OUTLIERS FOR GROUP 0]')
	print( g0[(g0['x'] < IQR_g0_lo) | (g0['x'] > IQR_g0_hi)]['x'] )

	print('\n[OUTLIERS FOR GROUP 1]')
	print( g1[(g1['x'] < IQR_g1_lo) | (g1['x'] > IQR_g1_hi)]['x'] )

def Q3A():
	num_fraud = fraud[fraud['FRAUD'] == 1].count()['FRAUD']
	num_investigations = len(fraud)

	print('Percentage of fraudulent investigations: %.4f' % (num_fraud/num_investigations))

def Q3B():
	fraud_interval = fraud.drop(['FRAUD', 'CASE_ID'], axis=1)
	fraud_long = pd.melt(fraud, id_vars=['FRAUD'])
	fraud_long.drop(fraud_long[fraud_long['variable'] == 'CASE_ID'].index)

	totalSpend = fraud_long[ fraud_long['variable'] == 'TOTAL_SPEND']
	doctorVisits = fraud_long[ fraud_long['variable'] == 'DOCTOR_VISITS']
	numClaims = fraud_long[ fraud_long['variable'] == 'NUM_CLAIMS']
	memberDuration = fraud_long[ fraud_long['variable'] == 'MEMBER_DURATION']
	optomPresc = fraud_long[ fraud_long['variable'] == 'OPTOM_PRESC']
	numMembers = fraud_long[ fraud_long['variable'] == 'NUM_MEMBERS']

	variables = [totalSpend, doctorVisits, numClaims, memberDuration, optomPresc, numMembers]
	for i, var in enumerate(variables):
		sns.boxplot(x='value', y='variable', hue='FRAUD', data=var)
		plt.show()

def Q3C():
	fraud_interval = fraud.drop(['FRAUD', 'CASE_ID'], axis=1)
	x = fraud_interval.to_numpy()
	xtx = x.transpose().dot(x)
	evals, evecs = LA.eigh(xtx)
	print("Eigenvalues of x = \n", evals)
	print("Eigenvectors of x = \n",evecs)

	# Here is the transformation matrix
	transf = np.dot(evecs, LA.inv(np.sqrt(np.diagflat(evals))));
	print("Transformation Matrix = \n", transf)

	# Here is the transformed X
	transf_x = np.dot(x, transf);
	print("The Transformed x = \n", transf_x)

	# Check columns of transformed X
	xtx = np.dot(transf_x.T, transf_x)
	print("Expect an Identity Matrix = \n", xtx)
	return transf_x, transf

def Q3D():
	transf_x, transf = Q3C()
	X = transf_x  # transformed X from part c
	y = fraud['FRAUD'].to_numpy()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
	neigh.fit(X, y)
	neigh.score(X, y)
	return X, y

def Q3E():
	transf_x, transf = Q3C()
	X, y = Q3D()
	kNN_model = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
	nbrs = kNN_model.fit(X, y)
	distances, indices = nbrs.kneighbors(X)

	focal = [[7500, 15, 4, 127, 2, 2]]

	transf_focal = focal * transf

	myNeighbors_t = nbrs.kneighbors(transf_focal, return_distance = False)
	print("My Neighbors = \n", myNeighbors_t)
	return transf_focal

def Q3F():
	neigh_prediction = neigh.predict(X_test)
	print( 'Prediction %:', sum(neigh_prediction == 1) / len(neigh_prediction) )

Q1A()
Q1B()
Q1C()
Q1DEFG()
Q2A()
Q2B()
Q2C()
Q3A()
Q3B()
Q3C()
Q3D()
Q3E()