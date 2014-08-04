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
	boot_list = list()
	
	# Sets up parameters
	N = len(W[:, 0])
	K = len(W[0, 1:])
	
	if b_cluster is not None:
		# Transform factor into dummies
		numberClusters = len(bCluster.columns)
						
		# Check if cluster has one class
		if numberClusters == 1:
			raise Exception('This factor has only one unique value!')
	
		n_sampling = N / number_clusters
		#rep_number = 400
		
	for rep in xrange(rep_number):
		# See Cameron & Triverdi (2005), p.360 for the bootstrap algorithm
				
		# Calculate the target statistic
		if statistic is 'se' and method is 'nonparametric':
			if b_cluster is None:
				# Gets indexes of a sample with replacement
				sample_indx = [randint(0, N - 1) for i in xrange(N)]
		
				# Get the sample database
				smple = W[sample_indx, :]
				
				# Define X and y for the sample
				X_b = smple[:, 1:]
				y_b = smple[:, 0]
				
				# Calculate the OLS parameters
				XX_b = np.dot(X_b.T, X_b)
				if np.linalg.det(XX_b) != 0:
					XX_b_inv = np.linalg.inv(XX_b)
				else:
					raise ValueError('Independent variables are colinear!')
				Xy_b = np.dot(X_b.T, y_b)
				Beta_b = np.dot(XX_b_inv, Xy_b)
				
				# Add them to boot_list
				boot_list.append(Beta_b)  
			
			else:
				# With cluster, we sample with replacement each cluster and then
				# use all observations inside the cluster. 	# See Cameron & Triverdi (2005), 
				# p.708 for an application to panel data estimation.
				
				# Sample from clusters
				sample_indx = [randint(0, number_clusters - 1) for i in xrange(n_sampling)]
				
				# Get the sample data			
				chosen = list()
				for rndom in sample_indx:
					chosen.append(W[np.array(clusters.iloc[:, rndom] == 1), :])
					
				chosen_W = np.concatenate(chosen)
				
				# And now it is just as before
				# Define X and y for the sample
				X_b = chosen_W[:, 1:]
				y_b = chosen_W[:, 0]
				
				# Calculate the OLS parameters
				XX_b = np.dot(X_b.T, X_b)
				if np.linalg.det(XX_b) != 0:
					XX_b_inv = np.linalg.inv(XX_b)
				else:
					raise ValueError('Independent variables are colinear!')
				Xy_b = np.dot(X_b.T, y_b)
				Beta_b = np.dot(XX_b_inv, Xy_b)
				
				# Add them to boot_list
				boot_list.append(Beta_b)  	
							
		else:
			raise NotImplementedError('Sorry!')
	
	if statistic == 'se':
		# See Cameron & Triverdi (2005), p. 362 for details
		Theta_hat = np.sum(boot_list, axis=0) / float(rep_number)
		
		for i in xrange(rep_number):
			boot_list[i].shape = (K, 1)
			Theta_hat.shape = (K, 1)
			boot_diff = boot_list[i] - Theta_hat
			boot_list[i] = np.dot(boot_diff, boot_diff.T)
		
		VarCoVar = np.sum(boot_list, axis=0) / float(rep_number - 1)
	
	# Returns result
	return VarCoVar
	
	# This is the end.

if __name__ == '__main__':
	
	# paths
	folder_path = '/home/pedro/Dropbox/doutorado/4o ano/2014 research/networks/'
	df_path = folder_path + 'prepdata/data/MN/final/'

	# read csv file
	df = pd.read_csv(df_path + 'dfMN.csv', low_memory=False)
	#df = df.iloc[0:10]
	cluster = 'VTD'
	
	y_variable = 'dem_votes'
	x_variable = ['x_rich', 'x_poorcl', 'nbx_rich']
	df = df.loc[:, x_variable + [y_variable] + [cluster]].dropna()

	# Adds intercept
	x_variable = ['Intercept'] + x_variable
	df['Intercept'] = 1
		
	# Sets some basic parameters
	N = len(df)
	K = len(x_variable)	
			
	# Creates X and y matrixes	
	X = df.loc[:, x_variable].as_matrix()
	X.shape = (N, K)
	y = df.loc[:, y_variable].as_matrix()
	y.shape = (N, 1)

	w = np.concatenate([y, X], axis=1)
	W=w
	rep_number = 200
	statistic='se'
	b_cluster=None
	method='nonparametric'
	x = p_bootstrap(df, W) #, b_cluster=cluster)
