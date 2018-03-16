##############################################################################################################
############				ARTIFICIAL INTELLIGENCE CS561 ASSIGNMENT 1                 	######################
############								GROUP 23									######################
############	This code contains the implementation of Hill Climbing Technique		######################
############	algorithm of feature selection. This only has the main function and 	######################
############	all the preprocessing and evaluation code is in a separate file			######################
##############################################################################################################


import numpy as np
import sklearn
import copy
import sys
import pandas as pd 
import matplotlib.pyplot as plt
from utils import *

from random import randint
import random
from datetime import datetime


## load the csv file 
## Parameters : (input file name ,  separator)
## this will return a Pandas dataframe where the data has been properly processed
## 				and a Target vector
df,Y = load_dataset("winequality-white.csv",';')

## get the features list
Remaining = df.columns.tolist()
Optimal = []

## We need to track the global best subset of features
global_min = sys.float_info.max
global_max_set = []
MSE=[]
min_k=0

iteration=-1


## We will start with a random subset of features
random.seed(datetime.now())
size = randint(0, len(Remaining))
Optimal = [ Remaining[i] for i in sorted(random.sample(xrange(len(Remaining)), size))]

#remove features in feature_subset from Remaining
for f in Optimal:
	Remaining.remove(f)


opt = True
############################## HILL CLIMBING TECHNIQUE #######################################
while opt:
	iteration=iteration+1
	min_score = sys.float_info.max
	f_selected = None

	## flag to tell whether to remove or add a feature
	toAdd = 0
	
	# First we will adding a new feature
	for f in Remaining:
		## add the feature
		Optimal.append(f)

		# extract the subset of features and evaluate on the chosed model
		X = df[Optimal]
		mse = evaluateObj(X,Y)

		# if this change gave us some improvement then flag the current feature as the best (for now)
		if mse <= min_score:
			min_score = mse
			f_selected = f
			toAdd = 1
		Optimal.remove(f)


	# Now we will consider removing an existing feature
	if(len(Optimal)>1):
		for f in Optimal:
			#remove the feature
			Optimal.remove(f)
			
			# extract the subset of features and evaluate on the chosed model
			X = df[Optimal]
			mse = evaluateObj(X,Y)

			# if this change gave us some improvement then flag the current feature as the best (for now)
			if mse <= min_score:
				min_score = mse
				f_selected = f
				toAdd = -1
			Optimal.append(f)

	## We get the best susbset by adding the new feature
	if toAdd is 1 :
		Optimal.append(f_selected)
		Remaining.remove(f_selected)
		print (len(Optimal), "Selected ",f_selected, min_score)
	## we get the best subset by removing a current feature
	if toAdd is -1 :
		Optimal.remove(f_selected)
		Remaining.append(f_selected)
		print (len(Optimal), "Removed ",f_selected, min_score)
	

	## update the global optimal set
	if min_score < global_min:
		global_min = min_score
		global_max_set= copy.copy(Optimal)
		min_k = iteration+1
		MSE.append(min_score)
	
	# if we get to a local minima we stop
	else:
		opt = False
		break


############################ END OF HILL CLIMBING TECHNIQUE ###############################
 
print("\n\nThe optimal subset of features is :")
print(global_max_set)


## Plot the MSE vs no of features graph
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(range(1,iteration + 1), MSE, marker='o', linestyle='-', color='r')

## we want to annotate the global minimum on the graph
ax.annotate('global minimum', xy=(min_k, global_min), xytext=(0.7,0.6 ), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->",facecolor='black'))
plt.xlabel("No. of iterations")
plt.ylabel("Mean Squared Error")
plt.title("No. of iterations VS Mean Squared Error")
plt.show()





