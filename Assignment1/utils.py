##############################################################################################################
############				ARTIFICIAL INTELLIGENCE CS561 ASSIGNMENT 1                 	######################
############								GROUP 23									######################
############	This code contains the implementation of utility functions used in the 	######################
############	main assigment. This has 2 main functions:								######################
############		1. load_dataset:  This function takes any csv file and the separator######################
############						and processes it to handle the categorical data 	######################
############		2. evaluateObj : This function has model evaluation for the feature ######################
############						subset that we propose								######################
##############################################################################################################


import sklearn
import pandas as pd 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

def getFieldnames(csvFile,separator):
    """
    Read the first row and store values in a tuple
    """
    with open(csvFile) as csvfile:
        firstRow = csvfile.readlines(1)
        fieldnames = tuple(firstRow[0].strip('\n').split(separator))
    return fieldnames

def load_dataset(input_file,separator):
	"""
    Read the csv file and handle the categorical data (if any)
    Returns : (pandas.DataFrame , numpy.array)
    """

    ## first get the names of the fields in the csv file
	colNames = getFieldnames(input_file,separator)

	## We will use Pandas DataFrame to read the csv file
	## Skip the first line as it has the names of fields
	df = pd.read_csv(input_file,skiprows=[0], sep=separator,names=colNames,header=None)

	## the Pandas has 3 types of fields : int64, float and object
	## we need to convert these "object" columns into categorical data
	object_columns = df.select_dtypes(['object']).columns

	# we tell pandas that we want to treat these columns as categorical data
	for col in list(object_columns):
	    df[col] = df[col].astype('category')

	# Categorical dtypes in Pandas can be easily converted into their codes using the inbuilt lambda function
	df[object_columns] = df[object_columns].apply(lambda x: x.cat.codes)


	# If there are NaN values in the dataset then replace these with the mean of the columns
	df.fillna(df.mean(), inplace = True)

	

	### We also want to normalise them
	#copy the headers of the dataframe
	y = list(df.columns)

	# extract the values as a numpy array from the dataframe to normalise them
	x = df.values

	# the preprocessing lib has a utility to normalise data in np arrays
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled)

	# Copies the column headers back to df
	df.columns = y

	# Y contains data of the output column
	Y = df.iloc[:,-1]
	df = df.drop(colNames[-1],axis=1)

	return df,Y



def evaluateObj(X,Y):
	"""
    This function represents the model evaluation
    Inputs : (pandas.DataFrame , numpy.array)
    output : mean square error
    """		

    ## we split the data into training and testing splits
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.33,random_state=5)

	## we use a simple linear regressor to evaluate out feature subsets
	lm = LinearRegression()
	lm.fit(X_train, Y_train)	## train the model
	Y_pred = lm.predict(X_test)	## compute the predictions for training data
	
	#calculate the MSE
	mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
	return mse