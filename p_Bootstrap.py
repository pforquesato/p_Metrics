import numpy as np
from scipy.stats import t
from random import randint
import scipy.sparse as sparse

def p_Bootstrap(W, repNumber=200, statistic='se', bCluster=None, method='nonparametric'):
	'''
	Implements Bootsrap method on data, obtaining the target statistic.
	
	__author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
	
	...
	
    Arguments
	---------
	
	W		: np.array
			  A two-dimensional array W with the data to be sampled. 
	rep_number	: int
			  The number of replications of data sampling performed by the 
			  bootstrapping procedure. Defaults to 200, following Efron 
			  and Tibsharani (1993), p. 52.
	statistic	: str
			  A string with the statistic to be bootstrapped. Defaults to 
			  Std. Errors 'se', and so far only implemented statistic.
	cluster		: bool
			  Whether to calculate Cluster-robust Bootstrapped Standard 
			  Errors. Default to False. 
	method		: str
			  One of possible methods of performing bootstrap. Defaults 
			  to 'nonparametric', other possibilities are 'parametric' 
			  and 'residual'. For more information, see Cameron & 
			  Triverdi (2005), p.360-361.
	'''
	# List where the bootstrapped repetitions will be sent
	bootList = list()
	
	# Sets up parameters
	N = len(W[:, 0])
	K = len(W[0, 1:])
	
	if bCluster is not None:
		# Transform factor into dummies
		numberClusters = len(bCluster.columns)
						
		# Check if cluster has one class
		if numberClusters == 1:
			raise Exception('This factor has only one unique value!')
	
		nSampling = N / numberClusters
		#repNumber = 400
		
	for rep in xrange(repNumber):
		# See Cameron & Triverdi (2005), p.360 for the bootstrap algorithm
				
		# Calculate the target statistic
		if statistic is 'se' and method is 'nonparametric':
			if bCluster is None:
				# Gets indexes of a sample with replacement
				sampleIndx = [randint(0, N - 1) for i in xrange(N)]
		
				# Get the sample database
				smple = W[sampleIndx, :]
				
				# Define X and y for the sample
				X_b = smple[:, 1:]
				y_b = smple[:, 0]
				
				# Calculate the OLS parameters
				XX_b = np.dot(X_b.T, X_b)
				if np.linalg.det(XX_b) != 0:
					XX_b_inv = np.linalg.inv(XX_b)
				else:
					raise ValueError("X'X (bootstrap) is singular!")
				Xy_b = np.dot(X_b.T, y_b)
				betaB = np.dot(XX_b_inv, Xy_b)
				
				# Add them to boot_list
				bootList.append(betaB)  
			
			else:
				# With cluster, we sample with replacement each cluster and then
				# use all observations inside the cluster. 	# See Cameron & Triverdi (2005), 
				# p.708 for an application to panel data estimation.
				
				# Sample from clusters
				sampleIndx = [randint(0, numberClusters - 1) for i in xrange(nSampling)]
				
				# Get the sample data			
				chosen = list()
				for rndom in sampleIndx:
					chosen.append(W[np.array(bCluster.iloc[:, rndom] == 1), :])
					
				chosenW = np.concatenate(chosen)
				
				# And now it is just as before
				# Define X and y for the sample
				X_b = chosenW[:, 1:]
				y_b = chosenW[:, 0]
				
				# Calculate the OLS parameters
				XX_b = np.dot(X_b.T, X_b)
				if np.linalg.det(XX_b) != 0:
					XX_b_inv = np.linalg.inv(XX_b)
				else:
					raise ValueError("X'X (bootstrap) is singular!")
				Xy_b = np.dot(X_b.T, y_b)
				betaB = np.dot(XX_b_inv, Xy_b)
				
				# Add them to boot_list
				bootList.append(betaB)  	
							
		else:
			raise NotImplementedError('Sorry!')
	
	if statistic is 'se':
		# See Cameron & Triverdi (2005), p. 362 for details
		thetaHat = np.sum(bootList, axis=0) / float(repNumber)
		
		for i in xrange(repNumber):
			bootList[i].shape = (K, 1)
			thetaHat.shape = (K, 1)
			bootDiff = bootList[i] - thetaHat
			bootList[i] = np.dot(bootDiff, bootDiff.T)
		
		varCoVar = np.sum(bootList, axis=0) / float(repNumber - 1)
	
	# Returns result
	return varCoVar
	
	# This is the end.
