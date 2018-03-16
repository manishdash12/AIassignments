##############################################################################################################
############				ARTIFICIAL INTELLIGENCE CS561 ASSIGNMENT 1                 	######################
############								GROUP 23									######################
############	This code contains the implementation of Sequential Selection First		######################
############	algorithm of feature selection. This only has the main function and 	######################
############	all the preprocessing and evaluation code is in a separate file			######################
##############################################################################################################

import numpy as np
import sklearn
import copy
import sys
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import *


## load the csv file 
## Parameters : (input file name ,  separator)
## this will return a Pandas dataframe where the data has been properly processed
## 				and a Target vector
df,Y = load_dataset("winequality-red.csv",';')

## get the features list
Remaining = df.columns.tolist()
Optimal = []

## We need to track the global best subset of features
global_min = sys.float_info.max
global_max_set = []
MSE=[]
min_k=0

iteration=-1
############################## SEQUENTIAL FORWARD SELECTION #######################################
while Remaining:
	min_score = sys.maxint
	f_selected= None
	iteration = iteration +1

	## each iteration we will consider a new feature to be included into the feature list
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
		Optimal.remove(f)

	## the above loop identifies the best feature to be selected in this iteration
	## modify the Optimal list of features
	Optimal.append(f_selected)
	Remaining.remove(f_selected)

	## record the MSE for plotting later
	MSE.append(min_score)

	print ("iteration ",iteration+1, Optimal)
	## update the global optimal set
	if min_score <= global_min:
		global_min = min_score
		global_max_set= copy.copy(Optimal)
		min_k = iteration+1

############################ END OF SEQUENTIAL FORWARD SELECTION ###############################
 
print("\n\nThe optimal subset of features is :")
print(global_max_set)


## Plot the MSE vs no of features graph
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(range(1,df.shape[1] + 1), MSE, marker='o', linestyle='-', color='r')

## we want to annotate the global minimum on the graph
ax.annotate('global minimum', xy=(min_k, global_min), xytext=(0.7,0.6 ), textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->",facecolor='black'))
plt.xlabel("No. of features: k")
plt.ylabel("Mean Squared Error")
plt.title("No. of features VS Mean Squared Error")
plt.show()


