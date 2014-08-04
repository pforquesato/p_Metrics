import pandas as pd
import numpy as np
import time
from scipy.stats import t
from tabulate import tabulate
import scipy.sparse as sparse
import p_Bootstrap

class p_OLS_raw(object):
	'''
	Class for Ordinary Least Squares linear regression on properly prepared 
	dependent variable (y) and independent variables (X) matrixes. For direct use from
	pandas DataFrame, see the p_OLS inherited class.
	
	__author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
	
	...
	
	Arguments
	---------
	
	yMatrix			: numpy.array (Nx1)
					  A Nx1 matrix with N observations of the dependent variable (y).
	xMatrix			: numpy.array(NxK)
					  A NxK matrix with N observations of the K independent variables (X).
	seMethod		: str
					  One of possible methods of calculating standard errors. Options are None, which
					  uses standard simplified formula for SE, 'robust' for White-Huber
					  heteroskedasticity-robust SE and 'bootstrap' for bootstrap-calculated SE.
					  Defaults to 'robust', following discussion in Cameron & Triverdi (2005), p.74-75.
	clusterDummies	: DataFrame
					  A pandas DataFrame with the cluster variable in dummy format, if it exists. 
					  Defaults to None.
					 
	'''	
	
	def __init__(self, yMatrix, xMatrix, seMethod, clusterDummies):
		'''
		Initializes p_OLS_raw and calculates variables of interest.
		'''
		# Here we do all the algebra in _p_OlsRun. After it,
		# we calculate the OLS statistics.
			
		# Always useful to have No. of observations and variables.
		self.N = len(xMatrix[:, 0])
		self.K = len(xMatrix[0, :])
		
		# Step 1:
		# We run OLS by calling self._p_OlsRun
		self._p_OlsRun(yMatrix, xMatrix, seMethod, clusterDummies)
		
		# Step 2:
		# Having the OLS results, we calculate some extra
		# useful statistics.
		
		# The T-value is simply coefficient divided by standard error.
		self.tValue = np.divide(self.beta, self.stdError)
		
		# The p-value is the survival function of T-value * 2 (bi-modal), 
		# given N - K degrees of freedom.
		self.pValue = t.sf(np.abs(self.tValue), self.N - self.K) * 2
	
		# To calculate R square, we divide the sum of squared errors
		# by the total sum of squares, and take the complement.
		
		# The Sum of Error is the dot square of the residuals.
		ssError = np.dot(self.residuals.T, self.residuals)[0, 0]
		
		# The Total Sum is (y - y*).T * (y - y*), where y* is the
		# mean.
		yMatrixMean = yMatrix.mean()
		yMatrixMean = np.repeat(yMatrixMean, self.N)
		yMatrixMean.shape = (self.N, 1)
		yMatrixBar = yMatrix - yMatrixMean
		ssTotal = np.dot(yMatrixBar.T, yMatrixBar)[0, 0]
		self.rSquare = 1 - ssError/ssTotal
		
		# Finally, the Adj. R Square adjusts it by the number of 
		# independent variables.
		self.adjRSquare = 1 - (1 - self.rSquare) * (float(self.N - 1)/(self.N - self.K - 1))

		
	def _p_OlsRun(self, y, X, seMethod, clusterDummies):
		'''
		Runs Ordinary Least Squares linear regression on chosen variables of a pandas DataFrame.
		
		__author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
		
		...
		
		Arguments
		---------
		
		y				: numpy.array (Nx1)
						  A Nx1 matrix with N observations of the dependent variable (y).
		X				: numpy.array(NxK)
						  A NxK matrix with N observations of the K independent variables (X).
		intercept		: bool
						  Whether to add an intercept to regression. Defaults to True.
		seMethod		: str
						  One of possible methods of calculating standard errors. Options are None, which
						  uses standard simplified formula for SE, 'robust' for White-Huber
						  heteroskedasticity-robust SE and 'bootstrap' for bootstrap-calculated SE.
						  Defaults to 'robust', following discussion in Cameron & Triverdi (2005), p.74-75.
		clusterDummies	: DataFrame
						  A pandas DataFrame with the cluster variable in dummy format, if it exists. 
						  Defaults to None.						 
		'''			
		# Here is where the magic happens (or the sausage making,
		#  depending on point of view).
		
		# Step 1:
		# To calculate the coefficients (beta), we use the usual 
		# OLS formula: beta = (X'X)^-1 * X'y  (see CT05, p.71).
		XX = np.dot(X.T, X)
		
		# First it is useful to check if the matrix is singular.
		if np.linalg.det(XX) != 0:
			XX_inv = np.linalg.inv(XX)
		else:
			raise ValueError("X'X matrix is singular!")
		Xy = np.dot(X.T, y)
		beta = np.dot(XX_inv, Xy)
		
		# The residuals are therefore the difference between
		# actual values y and predicted values X * beta^
		residuals = y - np.dot(X, beta)
		
		# Step 2:
		# Now we calculate the Covariance Matrix (VarCoVar)
		# with method depending on chosen SE Method
		if clusterDummies is not None and seMethod is 'robust':				
			# Cluster-robust Heterokedasticity-robust Method
			# We implement it according to CT05, p. 834.
			# The idea is to keep correlation within cluster unrestricted
			# (as well as heterokedasticity). The formula is:
			# [Sum(c)Xc'Xc]^-1 * Sum(c)Xc'uc uc' Xc * [Sum(c)Xc'Xc]^-1
			
			# First we get the number of clusters types
			numberClusters = len(clusterDummies.columns)
			
			# As with dummies, it is useful to make sure the factor
			# has more than one value, otherwise the matrix will be singular.
			if numberClusters == 1:
				raise Exception("The X'X matrix is singular!")
			
			# To implement the formula above, we need to multiply Xc'Xc
			# for each cluster, and then sum. To do so, we concatenate
			# the Xc matrixes along the number of clusters.
			X_c = [None] * numberClusters
			u_c = [None] * numberClusters
			XX_c = [None] * numberClusters
			XuuX_c = [None] * numberClusters
			
			# Step SE1:
			# Before we start, we need to separate the X matrix among
			# the different clusters, and the same for the residuals uc
			for i in xrange(numberClusters):
				# For each cluster, we select X  and u for which this 
				# cluster's dummies are one.
				X_c[i] = X[np.array(clusterDummies.iloc[:, i] == 1), :]
				u_c[i] = residuals[np.array(clusterDummies.iloc[:, i] == 1), :]
				
				# And then we multiply them, for each cluster.
				XX_c[i] = np.dot(X_c[i].T, X_c[i])
				XuuX_c[i] = np.dot( np.dot(X_c[i].T, u_c[i]), np.dot(u_c[i].T, X_c[i]))
			
			# We sum each list of matrixes along the clusters
			XuuX_c = np.sum(XuuX_c, axis=0)
			
			# And as in the formula, we invert [Sum(c)Xc'Xc]
			XX_c_inv = np.linalg.inv(np.sum(XX_c, axis=0))
			
			# And finally we get the VarCoVar matrix.
			varCoVar = float(self.N - 1)/(self.N - self.K) * float(numberClusters)/(numberClusters - 1) * \
						np.dot(XX_c_inv, np.dot(XuuX_c, XX_c_inv)) 
			
			# From the VarCoVar we can get the standard errors
			# simply picking from the diagonal.
			stdError = np.sqrt(np.diagonal(varCoVar))
			stdError.shape = (self.K, 1)
			
		elif seMethod is None or seMethod is 'None':
			# Homoskedasticity Restricted Std. Errors.
			# These SE are only consistent if the errors are
			# homosketastic. They are included only for completeness,
			# as they rarely apply. See CT05, p. 73.			
			# The formula is: s^2 * (X'X)^-1
			
			sSquare = np.square(residuals).sum()
			sigma = float(sSquare * self.N)
			varCoVar = sSquare/(self.N - self.K) * XX_inv
				
			# From the VarCoVar we can get the standard errors
			# simply picking from the diagonal.
			stdError = np.sqrt(np.diagonal(varCoVar))
			stdError.shape = (self.K, 1)
			
		elif seMethod is 'robust':
			# Heteroskedasticity-robust Std. Errors.
			# We implement it according to CT05, p. 72-73
			# The formula is:
			# (X'X)^-1 * X' Sigma X * (X'X)^-1
			
			# First we build the Sigma^ matrix by making it a 
			# diagonal matrix with the squared residuals.
			uSquare = np.square(residuals)
			uSquare.shape = (self.N,)
			
			# We use a sparse matrix to keep sigma, as all non-diagonal
			# elements are zero (and the matrix can get quite big!).
			sigma = sparse.diags(uSquare, 0)
			sigma_XX = np.dot(X.T, sigma.dot(X))
			varCoVar = float(self.N)/(self.N - self.K) * np.dot(np.dot(XX_inv, sigma_XX), XX_inv)
			
			# From the VarCoVar we can get the standard errors
			# simply picking from the diagonal.
			stdError = np.sqrt(np.diagonal(varCoVar))
			stdError.shape = (self.K, 1)
	
		elif seMethod is 'bootstrap':
			# The p_Bootstrap function accepts a W matrix to be 
			# bootstrapped. We build it here...
			W = np.concatenate([y, X], axis=1)
			
			# ... and then send to the function. Check it for details.
			varCoVar = p_Bootstrap.p_Bootstrap(W, b_cluster=clusters)
			
			# From the VarCoVar we can get the standard errors
			# simply picking from the diagonal.			
			stdError = np.sqrt(np.diagonal(varCoVar))
			stdError.shape = (self.K, 1)
		
		else:
			raise ValueError('Variable seMethod was not given an acceptable string. Try: None, "bootstrap" or "robust".')
		
		# Finally, we define p_OLS_raw attributes with the important variables.
		# They will be inherited by p_OLS class and become our results.
		self.beta = beta; self.stdError = stdError; self.residuals = residuals
		self.fitted = np.dot(X, beta); self.varCoVar = varCoVar
		


class p_OLS(p_OLS_raw):
	'''
	Class for Ordinary Least Squares linear regression results on chosen variables of a pandas DataFrame.
	
	__author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
	
	...
	
	Arguments
	---------
	
	dataFrame		: DataFrame
					  The Pandas DataFrame to which y_variable and x_variable belongs.
	yVariable		: str
					  A string with the name of the variable (column) in df to be used as
					  the dependent variable.
	xVariable		: list
					  List of strings with names of variables (columns) in df to be used as
					  independent variables. Factors can be marked by adding 
					  'factor:' in front of the variable. Interactions can be added using '*'
					  between variable names.
	intercept		: bool
					  Whether to add an intercept to regression. Defaults to True.
	seMethod		: str
					  One of possible methods of calculating standard errors. Options are None, which
					  uses standard simplified formula for SE, 'robust' for White-Huber
					  heteroskedasticity-robust SE and 'bootstrap' for bootstrap-calculated SE.
					  Defaults to 'robust', following discussion in Cameron & Triverdi (2005), p.74-75.
	cluster			: str
					  Name of the variable that defines the clusters, if those exist. Defaults to None.
					  If a string is given, then Cluster-robust standard errors are calculated, and any value
					  given to se_method is ignored, except if given 'bootstrap', when Bootstrap clustered SE
					  are calculated instead (NOT WORKING ATM).  Designed for SMALL clusters, such that Variance 
					  Matrix can be properly estimated.	
	autoPrint		: bool
					  Whether should automatically print the results. Defaults to True.
	'''
	
	def __init__(self, dataFrame, yVariable, xVariable, intercept=True, \
					seMethod='robust', cluster=None, autoPrint=True):
		'''
		Initializes p_OLS, formats DataFrame and prepares matrixes for initializating p_OLS_raw.
		'''
		# The plan is to fix the DataFrame and variables, so that we can call p_OLS_raw
		# which effectively implements the OLS regression.

		# Step 1:
		# First we initialize variables, making sure to copy and not
		# link to variables.
		self.dataFrame = pd.DataFrame(dataFrame)
		self.yVariable = yVariable
		self.xVariable = list(xVariable)
		self.seMethod = seMethod
		self.cluster = cluster
		
		# Further, we add intercept (a column of 1s) if wanted 
		# (that is, if intercept is set to true).
		if intercept:
			self.dataFrame['Intercept'] = 1
			self.xVariable = ['Intercept'] + self.xVariable
		
		# Then we call _p_Dummify function to fix DataFrame
		#  for running p_OLS_raw. See method for details.
		self.dataFrame, self.xVariable = self._p_Dummify(self.dataFrame, \
			self.xVariable, self.yVariable, self.cluster)
		
		# Step 2:
		# It will be useful to have defined the No. observations and
		# No. of (x) variables.
		self.N = len(self.dataFrame)
		self.K = len(self.xVariable)
		
		# p_OLS_raw runs on y and X matrixes, so we need to build them,
		# making sure they are 2-dimensional numpy arrays for matrix algebra.
		X = self.dataFrame.loc[:, self.xVariable].as_matrix()
		X.shape = (self.N, self.K)
		y = self.dataFrame.loc[:, self.yVariable].as_matrix()
		y.shape = (self.N, 1)
		
		# If we have clusters, we need to build dummies for it so
		# we can calculate std. errors later.
		if self.cluster is not None:
			clusterDummies = pd.get_dummies(self.dataFrame[self.cluster], prefix='clstr')
		else:
			clusterDummies = None
		
		# Step 3:
		# Now we can call a p_OLS_raw object, and inherit its results.
		p_OLS_raw.__init__(self, y, X, self.seMethod, clusterDummies)
		
		# If autoPrint is True, then run outPrint
		if autoPrint:
			self.outPrint()

	
	def outPrint(self, printOpt='table', output=None):
		'''
		Prints the OLS parameters and statistics to either terminal or file (available in latex).
		
		Arguments
		---------			
		printOpt			: str
							  How should output be returned. Defaults to 'table', with 
							  results printed as a table. Other option is 'latex' for latex formatting.
		output				: str
							  The name of the file where the output should be printed. Defaults to
							  None, which means output is printed in console.
		'''		
		# This function prints the OLS results in a table format.
		
		# Step 1:
		# First we prepare the Header and the table with the results in a format
		# that the package tabulate can accept.
		headers = ['Variable', 'Coefficient', 'Std. Errors', 'T Value', 'P Value']
		table = list()
		for i, var in enumerate(self.xVariable):
			table.append([var, self.beta[i, 0], self.stdError[i, 0], self.tValue[i, 0], self.pValue[i, 0]])
		
		# Prepare for printing the method we used to calculate std. errors.
		if self.cluster is not None:
			if self.seMethod is not 'bootstrap':
				tpSE = 'Cluster-Robust'
			else:
				tpSE = 'Cluster-Bootstrapped SE'
		elif self.seMethod is 'robust':
			tpSE = 'Heteroskedasticity-Robust'
		elif self.seMethod is 'bootstrap':
			tpSE = 'Bootstrapped SE'
		elif self.seMethod is None or self.seMethod is 'None':
			tpSE = 'Homoskedastic Restricted'
		else:
			raise ValueError('Variable se_method did not receive an acceptable value. Try "bootstrap", "robust" or None.')
		
		# Step 2:
		# Above the table we print general statistics of the model (N, R^2, etc.)
		info = [['Dep. Variable:', self.yVariable], ['Model:', 'OLS'], ['Standard Errors:', tpSE], \
				['No. Observations:', str(self.N)], ['No. Variables:', str(self.K)],
				['R Squared:', '%.3f' % self.rSquare], ['Adj. R Squared', '%.3f' % self.adjRSquare]]		
		
		# Step 3: Print out the output.
		if printOpt not in ['table', 'latex']:
			# First we check for wrong printOpt input.	
			raise ValueError('Variable printOpt was not given an acceptable string. Try: "table" or "latex".')
		else:
			# Otherwise:
			if output is None:
				if printOpt is 'table':
					print tabulate(info, floatfmt='.4f', tablefmt='rst')
					print tabulate(table, headers=headers, floatfmt=".4f", tablefmt='rst')
			
				elif printOpt is 'latex':
					print tabulate(info, floatfmt='.4f', tablefmt='latex')
					print tabulate(table, headers=headers, tablefmt="latex", floatfmt=".4f")
					
			else:
				# If output is not None, we print to a file. First we open it...
				f = open(output + '.txt', 'w')
				
				# ... then we print...
				if printOpt is 'table':
					print >> f, tabulate(info, floatfmt='.4f', tablefmt='rst')
					print >> f, tabulate(table, headers=headers, floatfmt=".4f", tablefmt='rst')
			
				elif printOpt is 'latex':
					print >> f, tabulate(info, floatfmt='.4f', tablefmt='latex')	
					print >> f, tabulate(table, headers=headers, tablefmt="latex", floatfmt=".4f")
				
				# ... then we close it.	
				f.close()	
		

	def _p_Dummify(self, dataFrame, xVariable, yVariable, cluster):
		'''
		Prepares DataFrame and variable lists for transforming them into matrixes
		used for OLS algebra. Most work involves transforming factor variables into dummies.
		'''
		# Here we fix the DataFrame to be able to run OLS.
		
		# Step 1:
		# We accept factors without denomination when their type is 
		# pandas object. To do so, we simply put 'factor:' in front ourselves.
		for varIndex, varName in enumerate(xVariable):
			if varName in dataFrame.columns:
				if dataFrame[varName].dtype == 'O' and not varName.startswith('factor:'):
					x_variable[index] = 'factor:' + var_name
		
		'''
		# Step 1B:			
		# Check for interactions
		for index, var_name in enumerate(x_variable):
			if '*' in var_name:
				# So var_name is an interaction
				# First get the underlying variables
				splitted_var = var_name.split('*')
				for split_indx in range(len(splitted_var)):
					splitted_var[split_indx] = splitted_var[split_indx].replace(' ', '')
				
				# Now before we start, we make sure all variables are in
				# x_variable, as it is not kosher to use interactions without
				# the base variable.
				for split_indx in range(len(splitted_var)):								
					if splitted_var[split_indx] not in x_variable:
						x_variable.append(splitted_var[split_indx])
				
				# OK. So now, is any of them a dummy? If not, it is easier.
				if not any([x.startswith('factor:') for x in splitted_var]):			
					splitted_len = len(splitted_var) # Get No. of interacted variables.
					# And then just multiply the variables.
					df[var_name] = df.loc[:, splitted_var[0]]
					for i in range(1, splitted_len):
						df[var_name] = np.multiply(df[var_name], df.loc[:, splitted_var[i]])
				
				# Now, if some are dummies, that is more complicated,
				# because now we have multiple interactions.
				else:
					SplitFactorList = list()
					for SplitVar in splitted_var:
						if splitted_var[split_indx].startswith('factor:'):
							splitted_var[split_indx] = splitted_var[split_indx].replace('factor:', '')
							SplitFactorList.append(splitted_var[split_indx])
					
					# Now we have our list of factors. We need for each one
					# to get the underlying dummies.
					for element in SplitFactorList:
						SplitDummies = pd.get_dummies(df[element], prefix=element)
						splitted_var.remove(element)
						
						# It is always good to check if there is more than one element
						# in dummy, otherwise matrix will be singular.
						if len(dummies.columns) == 1:
							raise Exception('This factor has only one unique value!')
			
						else:
							# Remember to delete one dummy, to avoid the 
							# "Dummy variable trap".
							SplitDummies = dummies.iloc[:, 1:]
							df = df.join(SplitDummies) # Add them to DataFrame.
					
					# Now we need to build one interaction for each
					# dummy variable.
					for dummy in SplitDummies:
						dummyName = '*'.join([dumy] + splitted_var) 
						
						# Add this new interaction to variables list.
						x_variable.append(dummyName)
						# ... and to DataFrame.
						df[dummyName] = df[dummy] # We add the dummy
						# And multiply by splitted_var
					for i in range(1, splitted_len):
						df[var_name] = np.multiply(df[var_name], df.loc[:, splitted_var[i]])
		'''
		# Dealing with factors is tricky. We create a list of dummies
		# and make sure the list of variables is always pointing to the
		# right name.
		dummyList = list()
		for varIndex, varName in enumerate(xVariable):
			# Scan through all X variables, if any is a factor...
			if varName.startswith('factor:'):
				# ... we need to remove the 'factor' from the name
				# (since it doesn't exist in the DataFrame)
				# and mark it as dummy (by adding to dummyList).
				xVariable[varIndex] = varName.replace('factor:', '')
				dummyList.append(xVariable[varIndex])
		
		# Step 2:
		# Before creating dummies, we remove all NA and 
		# save space by removing not needed variables
		# (also important to keep all numeric).
		if cluster is not None and cluster not in xVariable:
			# Makes sure we keep the cluster in dataFrame...
			dataFrame = dataFrame.loc[:, xVariable + [yVariable] + [cluster]].dropna()
		else:
			dataFrame = dataFrame.loc[:, xVariable + [yVariable]].dropna()
		
		# Step 3:
		# Now we create the dummies to substitute for the factor.
		for vfactor in dummyList:
			# Pandas makes the trouble of actually creating the dummies
			# a breeze. But we still need to fix namings.			
			dummies = pd.get_dummies(dataFrame[vfactor], prefix=vfactor)
			xVariable.remove(vfactor) # Remove factor from X
	
			# Always helpful to check if factor has more than one value,
			# to avoid singular matrixes.
			if len(dummies.columns) == 1:
				raise Exception('This factor has only one unique value!')
			
			else:
				# Remove one of the dummies, to avoid the 
				# 'Dummy variable trap'.				
				dummies = dummies.iloc[:, 1:]
				# And then add them to DataFrame and X variable names.
				dataFrame = dataFrame.join(dummies)
				xVariable.extend(dummies.columns)
		
		return dataFrame, xVariable



if __name__ == '__main__':
	
	# paths
	folder_path = '/home/pedro/Dropbox/doutorado/4o ano/2014 research/networks/'
	df_path = folder_path + 'prepdata/data/MN/final/'

	# read csv file
	df = pd.read_csv(df_path + 'dfMN.csv', low_memory=False)
	
	yv = 'dem_votes'
	xv = ['x_rich', 'x_poorcl', 'factor:YEAR'] # , 'factor:YEAR*x_rich'

	test = p_OLS(df, yv, xv) #, cluster='VTD') #, se_method='bootstrap')
	#test.outPrint()		
