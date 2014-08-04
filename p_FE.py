import pandas as pd
import numpy as np
import time
from scipy.stats import t, f
from tabulate import tabulate
import scipy.sparse as sparse
from scipy.stats.stats import pearsonr

import p_Bootstrap
import p_OLS

class p_FE(p_OLS.p_OLS):
	'''
	Implements linear regression models (FE and IVFE) on chosen variables of a 
	pandas DataFrame or Panel.
	
	__author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
	
	...
	
	Arguments
	---------
	
	dataFrame	: DataFrame
			  The Pandas DataFrame to which yVariable and xVariable 
			  belongs.
	yVariable	: str
			  A string with the name of the variable (column) in dataFrame
			  to be used as the dependent variable. 
	xVariable	: list
			  List of strings with names of variables (columns) in dataFrame
			  to be used as independent variables. Factors have to be marked 
			  by adding 'factor:' in front of the variable.
	effect		: str
			  String denoting Panel method to be applied. Defaults to "within",
			  but can also be used "first differences", "random" for Random 
			  Effects and 'ols' for Pooled OLS.
	indexes		: tuple (str)
			  Name of variables denoting individual (0) and time (1) indexes 
			  (MIND THE ORDER). If providing a pandas Panel with individual-time 
			  multi-indexing, use None (also Default). 
	endogenous	: list
			  List of strings with names of variables (columns) in dataFrame
			  considered endogenous and thus being instrumented by variables 
			  in instruments. These variables should NOT be added to xVariable. 
			  Defaults to None.
	instruments	: list
			  List of strings with names of variables (columns) in df to be 
			  used as instruments. These should include ONLY variables not already 
			  included in xVariable. Variables in xVariable which are not in 
			  endogenous are automatically instrumented by themselves. 
			  Defaults to None.
	intercept	: bool
			  Whether to add an intercept to regression. Defaults to True.
	seMethod	: str
			  One of possible methods of calculating standard errors. Options 
			  are None, which uses standard simplified formula for SE, 'robust' 
			  for White-Huber heteroskedasticity-robust SE and 'bootstrap' for 
			  bootstrap-calculated SE. Defaults to 'robust', following discussion 
			  in Cameron & Triverdi (2005), p.74-75.
				 
	'''			
	def __init__(self, dataFrame, yVariable, xVariable, indexes, \
		effect='within', endogenous=None, instruments=None, \
		intercept=False, seMethod='robust', autoPrint=True):
		'''
		Changes variables in pandas Panel according to effects (that is, 'within', 
		'between', 'random' or 'ols'), and then call inheriting object p_OLS.
		'''
		# So the plan here is to do the necessary transformations in data (e.g. within), 
		# and then call p_OLS (i.e. run OLS), with the appropriate individual clusters
		# and correcting the std. errors.
		
		# Step 1:
		# We set attributes, taking care to copy not only
		# link to variables.
		self.yVariable = yVariable
		self.xVariable = list(xVariable)
		self.effect = effect
		if endogenous is not None:
			self.endogenous = list(endogenous)
		else:
			self.endogenous = None
		if instruments is not None:
			self.instruments = list(instruments)
		else:
			self.instruments = None
		self.seMethod = seMethod
		self.indexes = indexes
		
		# Step 2:
		# It is useful to use the 'indexes' as pandas index of the
		# DataFrame. Here we also allow the indexes to be already
		# the DataFrame ones (and then indexes is None).
		if self.indexes is not None:
			self.dataFrame = dataFrame.set_index(self.indexes)
		else:
			self.dataFrame = pd.DataFrame(dataFrame)
			self.indexes = [self.dataFrame.index.names[0], \
				self.dataFrame.index.names[1]]
		
		# Call _p_DummifyFE method to fix the DataFrame for
		# running OLS. See the method for details.
		self.dataFrame, self.xVariable =  self._p_DummifyFE(self.dataFrame, \
			self.xVariable, self.yVariable)
		
		# According to pandas documentation, some methods don't function
		# properly if the indexes are not properly sorted. We do that here thus.
		self.dataFrame.sort_index(0, inplace=True)
		
		# Step 3:
		# Now we change the DataFrame according to the method used.
		if self.effect is 'within':
			# The within method consists of using variables modified as
			# x^ = x - x*, where x* is the mean for each individual.
			pWithin = lambda x: (x - x.mean())
			self.dataFrame = self.dataFrame.groupby(level=0).transform(pWithin)
			
			# (Ironically) for some things it is necessary to 
			# undo the indexing. We do so below (and for each effect).
			self.dataFrame.reset_index(inplace=True)
		
		elif self.effect is 'first differences':
			# The First Differences effect consists in taking discrete
			# differences of variables on time, for each individual.
			# This is equivalent to differentiating all and removing first
			# periods.
			self.dataFrame = self.dataFrame.diff()
			self.dataFrame.reset_index(inplace=True) # Reset Index
			self.dataFrame.loc[ self.dataFrame[self.indexes[0]] != \
				self.dataFrame[self.indexes[0]].shift( 1 ) ] = np.nan
		
		elif self.effect is 'random':
			# Random Effects are not yet implemented.
			raise NotImplementedError('Sorry!')
			
		elif self.effect is 'ols':
			# Naturally for Pooled OLS no transformation is required,
			# except to also reset the index.
			self.dataFrame.reset_index(inplace=True)
			
		else:
			raise ValueError('Variable effect was not given a valid value.' 
				'Try "within", "first differences" or "random".')

		# Now we obtain No. observations, periods and variables.
		# We must be careful not to override p_OLS variables.
		self.NF = len(self.dataFrame[self.indexes[0]].unique())
		self.TF = len(self.dataFrame[self.indexes[1]].unique())
		self.KF = len(self.xVariable)
		
		# Step 4:	
		# Now we call p_OLS on the transformed DataFrame, using cluster
		# on the individual to control for temporal error correlation.
		# See CT05, p.707.
		if self.endogenous is None:
			p_OLS.p_OLS.__init__(self, self.dataFrame, self.yVariable, \
				self.xVariable, seMethod=self.seMethod, \
				cluster=self.indexes[0], autoPrint=False)
		else:
			raise NotImplementedError('Sorry!')
		
		
		# For within FE, we need to adjust the std. errors, T and P values
		# for the loss of degrees of freedom (CT05, p.727).
		if effect is 'within':
			self.stdError = np.sqrt((self.NF * self.TF - self.KF) / \
				float(self.NF * (self.TF - 1) - self.KF)) * self.stdError
			self.tValue = np.divide(self.beta, self.stdError)
			self.pValue = t.sf(np.abs(self.tValue), self.NF - self.KF) * 2
		
		# autoPrint does what its name says.	
		if autoPrint:
			self.outPrint()

			
	def outPrint(self, printOpt='table', output=None):
		'''
		Prints the FE parameters and statistics to either terminal or 
		file (available in latex).
		
		Arguments
		---------			
		printOpt	: str
				  How should output be returned. Defaults to 'table', 
				  with results printed as a table. Other option is 
				  'latex' for latex formatting.
		output		: str
				  The name of the file where the output should be printed. 
				  Defaults to None, which means output is printed in console.
		'''
		# This function prints the OLS/FE results in a table format.
		
		# Step 1:
		# First we prepare the Header and the table with the results 
		# in a format that the package tabulate can accept.
		headers = ['Variable', 'Coefficient', 'Std. Errors', \
			'T Value', 'P Value']
		table = list()
		for i, var in enumerate(self.xVariable):
			table.append([var, self.beta[i, 0], self.stdError[i, 0], \
				self.tValue[i, 0], self.pValue[i, 0]])
		
		# Prepare for printing the method we used to calculate std. errors.
		if self.seMethod is 'robust':
			tpSE = 'Panel Adjusted Heteroskedasticity-Robust'
		elif self.seMethod is 'bootstrap':
			tpSE = 'Panel Adjusted Bootstrapped SE'
		elif self.seMethod is None or self.seMethod is 'None':
			tpSE = 'Panel Adjusted Homoskedastic Restricted'
		else:
			raise ValueError('Variable seMethod did not receive an '
				'acceptable value. Try "bootstrap", "robust" or None.')
		
		# Now we prepare the method ('effect') used to calculate the Panel model. 
		if self.effect is 'ols':
			tpEffect = 'Pooled OLS'
		else:
			tpEffect = self.effect.capitalize() + ' FE Model'
		
		# Step 2:
		# Above the table we print general statistics of the model (N, R^2, etc.)
		info = [['Dep. Variable:', self.yVariable], ['Model:', tpEffect], 
			['Standard Errors:', tpSE],  ['No. Individuals:', str(self.NF)], 
			['No. Periods:', str(self.TF)], ['Total No. Observations:', str(self.N)], 
			['No. Variables:', str(self.KF)], ['R Squared:', '%.3f' % self.rSquare], 
			['Adj. R Squared', '%.3f' % self.adjRSquare]]		
		
		# Step 3: Print out the output.
		if printOpt not in ['table', 'latex']:
			# First we check for wrong printOpt input.	
			raise ValueError('Variable print_opt was not given an '
				'acceptable string. Try "print" or "to_latex".')
		else:
			# Otherwise:
			if output is None:
				if printOpt is 'table':
					print tabulate(info, floatfmt='.4f', tablefmt='rst')
					print tabulate(table, headers=headers, \
						floatfmt=".4f", tablefmt='rst')
			
				else:
					print tabulate(info, floatfmt='.4f', tablefmt='latex')
					print tabulate(table, headers=headers, \
						tablefmt="latex", floatfmt=".4f")
					
			else:
				# If output is not None, we print to a file. First we open it...
				f = open(output + '.txt', 'w')
				
				# ... then we print...
				if printOpt is 'table':
					print >> f, tabulate(info, floatfmt='.4f', tablefmt='rst')
					print >> f, tabulate(table, headers=headers, \
						floatfmt=".4f", tablefmt='rst')
			
				else:
					print >> f, tabulate(info, floatfmt='.4f', tablefmt='latex')	
					print >> f, tabulate(table, headers=headers, \
						tablefmt="latex", floatfmt=".4f")
				
				# ... then we close it.	
				f.close()	


	def _p_DummifyFE(self, dataFrame, xVariable, yVariable):
		'''
		Prepares DataFrame and variable lists for transforming them 
		into matrixes used for FE/OLS algebra. Most work involves
		transforming factor variables into dummies.
		'''
		# Here we fix the DataFrame to be able to run OLS.
		
		# Step 1:
		# We accept factors without denomination when their type is 
		# pandas object. To do so, we simply put 'factor:' in front ourselves.
		for varIndex, varName in enumerate(xVariable):
			if varName in dataFrame.columns:
				if dataFrame[varName].dtype == 'O' and not \
					varName.startswith('factor:'):
					xVariable[varIndex] = 'factor:' + varName
		
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
	

	# Add some test data here.
