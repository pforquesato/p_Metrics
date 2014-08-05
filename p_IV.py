import pandas as pd
import numpy as np
import time
from scipy.stats import t, f
from tabulate import tabulate
import scipy.sparse as sparse
from scipy.stats.stats import pearsonr

import p_Bootstrap
import p_OLS


class p_IV_FS(object):
    '''
    Implements a series of tests of First Stage significance. If run separately, 
    also can provide First Stage results.
    
    __author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
    
    ...
    
    Arguments
    ---------
    
    dataFrame        : DataFrame
                      The Pandas DataFrame to which all variables belongs.
    endogenous        : str
                      A string with the name of the variables (columns) in df to be used as
                      the endogenous variables. They should NOT be included in x_variable.
    instruments        : str
                      A string with the name of the variables (columns) in df to be used as
                      instruments. They should NOT include the exogenous elements of 
                      x_variable (they are added automatically).
    xVariable        : list
                      List of strings with names of variables (columns) in df to be used as
                      independent variables. Factors can be marked by adding 
                      'factor:' in front of the variable.
    seMethod        : str
                      One of possible methods of calculating standard errors. Options 
                      are None, which uses standard simplified formula for SE, 'robust'
                      for White-Huber heteroskedasticity-robust SE and 'bootstrap' for
                      bootstrap-calculated SE. Defaults to 'robust', following discussion
                      in Cameron & Triverdi (2005), p.74-75.
    cluster            : str
                      Name of the variable that defines the clusters, if those exist. 
                      Defaults to None. If a string is given, then Cluster-robust standard 
                      errors are calculated, and any value given to se_method is ignored, 
                      except if given 'bootstrap', when Bootstrap clustered SE are calculated 
                      instead.  Designed for -small- clusters, such that Variance Matrix can 
                      be properly estimated.
        
    '''
    def __init__(self, dataFrame, endogenous, instruments, xVariable, seMethod, cluster):
        '''
        Initializes p_IV_FS and calculates variables of interest.
        '''
        # The idea here is that the DataFrame should already be prepared
        # by p_IV (we might want to change this in future). So we only need
        # to get attributes and run _p_PartialR() or _p_SheaR() and _p_Ftest()
        # First stage results will be preceded by 'fs_'
        
        # Step 1:
        # First we initialize variables, making sure to copy and not
        # link to variables.
        self.seMethod = seMethod
        self.endogenous = list(endogenous)
        self.instruments = list(instruments)
        self.xVariable = list(xVariable)
        self.dataFrame = pd.DataFrame(dataFrame)
        self.Cluster = cluster
        
        # Asking endogenous and instruments not to be included in 
        # xVariable simplifies the process of building all lists of
        # variables of interest. In particular the total set of 
        # instruments and total set of X variables.
        self.zVariable = self.xVariable + self.instruments
        self.augXVariable = self.xVariable + self.endogenous
        
        # It will be useful to have defined the No. observations and
        # No. of (x and z) variables.
        self.N = len(self.dataFrame)
        self.Kx = len(self.augXVariable)
        self.Kz = len(self.zVariable)
        
        # First we want to find the Partial R square for the instruments
        # If there is one endogenous variable, then we use the simple
        # R square formula in CT05, p.104. Otherwise, there are many
        # alternatives. We use Shea's Partial R Square (CT05, p.105).
        # See the methods for more details.
        if len(self.endogenous) == 1:
            self._p_PartialR()
        else:
            self._p_SheaR()
            
        # Then we run a F test (Wald test) of whether all instruments
        # coefficient equals to zero (similarly whether the model with
        # instruments fits better than the model without).See the 
        # methods for more details.
        self._p_Ftest()    
        
                    
    def _p_PartialR(self):
        '''
        Calculates Partial R square according to Bound, Jaeger and Baker (1995) 
        for 1 endogenous variable.
        '''
        # Here we use Bound, Jaeger and Baker (1995) Partial R Sq. 
        # As seen in CT05, p.104. The idea is to regress
        # (x1 - x1~) on (z - z~), where x1~ and z~ are the fitted
        # values from regressions of x1 and z on x2 (the exogenous part
        # of X), or equivalently, (x1 - x1~) are the residuals (as we
        # do below).
                        
        # To use p_OLS_raw, we need to define the matrixes. We 
        # will need x1, x2 and Z. And make sure they are 2-dimensional
        # numpy arrays (for matrix algebra).
        X2 = self.dataFrame.loc[:, self.xVariable].as_matrix()
        X2.shape = (self.N, len(self.xVariable))
        Z = self.dataFrame.loc[:, self.zVariable].as_matrix()
        Z.shape = (self.N, self.K_z)
        x1 = self.dataFrame.loc[:, self.endogenous].as_matrix()
        x1.shape = (self.N, 1)
        
        # Gets (x1 - x1~), the residuals of x1 (endogenous) on x2 (exogenous)        
        xTilde = p_OLS.p_OLS_raw(x1, X2, self.seMethod, self.cluster).residuals
        
        # Gets (z - z~), the residuals of instruments Z on x2 (exogenous).
        # Remember that even if there is only one endogenous variable, there
        # can be multiple instruments. So we need to run multiple 
        # regressions, one for each instrument.
        zTilde = list()
        for inst in self.instruments:
            z1 = self.dataFrame.loc[:, inst].as_matrix()
            z1.shape = (self.N, 1)        
            zTilde.append(p_OLS.p_OLS_raw(z1, X2, self.seMethod, self.cluster).residuals)
        
        # If there is only one instrument, then its residuals are it.
        # If there are more, we need to concatenate them and form a matrix.
        if len(zTilde) == 1:
            zTilde = zTilde[0]
        else:
            zTilde = np.concatenate(zTilde, axis=1)
        
        # Finally, we obtain what we sought: a regression of xTilde on zTilde.
        tildeOLS = p_OLS.p_OLS_raw(xTilde, zTilde, self.SE_method, self.Cluster)
        
        # And its R Square is the Partial R Square we were looking for.
        # (see p_IV_raw implementation for details).
        ssError = np.dot(tildeOLS.residuals.T, tildeOLS.residuals)[0, 0]
        xTildeMean = xTilde.mean()
        xTildeMean = np.repeat(xTildeMean, self.N)
        xTildeMean.shape = (self.N, 1)
        xTildeBar = xTilde - xTildeMean
        ssTotal = np.dot(xTildeBar.T, xTildeBar)[0, 0]
        self.fs_partialRSquare = [1 - ssError/ssTotal]


    def _p_SheaR(self):
        '''
        Calculates Shea (1997) Partial R square for more than 1 endogenous variables.
        '''
        # If there is more than one endogenous variable, the Partial
        # R Square formula doesn't apply. We then implement
        # Shea (1997)'s Partial R square, see CT05, p.105.
        # The idea is for each endogenous variable to get the square
        # sample correlation between (x1 - x1~) and (x1^ - x1^~), where
        # (x1 - x1~) is the residual from regressing x1 in X2, and 
        # (x1^ - x1^~) is the residual from regressing x1^ on X2^,
        # where xk^ is the fitted value of regressing xk (k=1,2) on Z.
        
        # Now, since there are multiple endogenous variables, we get
        # multiple Partial R Squares.
        partialRSquare = list()
        
        for endog in self.endogenous:
            # Step 1:
            # Here we have to be careful because X2 now includes
            # all other endogenous variables. It will be useful, however,
            # to still define only the exogenous ones.
            X2Exog = self.dataFrame.loc[:, self.xVariable].as_matrix()
            X2Exog.shape = (self.N, len(self.xVariable))
            x1 = self.dataFrame.loc[:, endog].as_matrix()
            x1.shape = (self.N, 1)
            Z = self.dataFrame.loc[:, self.zVariable].as_matrix()
            Z.shape = (self.N, self.Kz)
            
            # So we create a X2 without our target endogenous variable,
            # but with all others.
            endogenousWO = [x for x in self.endogenous if x is not endog]
            augXMinus = self.xVariable + endogenousWO                        
            X2 = self.dataFrame.loc[:, augXMinus].as_matrix()
            X2.shape = (self.N, len(augXMinus))
            
            # x1Tilde (x1 - x1~) again is the residual from x1 on X2,
            # but now some variables in X2 are endogenous.            
            x1Tilde = p_OLS.p_OLS_raw(x1, X2, self.seMethod, self.cluster).residuals
            
            # Step 2:
            # Now (x1^ - x1^~) is more complicated. First we need to get
            # x1^ and X2^, the regressions of xk^ on Z.
                        
            # x1^ (x1Hat) is the fitted value of x1 on Z.            
            x1Hat = p_OLS.p_OLS_raw(x1, Z, self.seMethod, self.cluster).fitted
            
            # Now I need to do this for all X2. But to simplify, note that
            # I only actually need this for endogenous variables. Naturally
            # if x is included in Z, then LP(x, Z) = x. So, for the
            # endogenous we get fitted values...        
            x2HatList = list()
            for endogWO in endogenousWO:
                xEndog = self.dataFrame.loc[:, endogWO].as_matrix()
                xEndog.shape = (self.N, 1)
                x2HatList.append(p_OLS.p_OLS_raw(xEndog, Z, self.seMethod, self.cluster).fitted)            
            
            # ... and for the rest we simply add them to the matrix.
            x2HatList = np.concatenate(x2HatList, axis=1)
            x2Hat = np.concatenate([x2HatList, X2Exog], axis=1)
            
            # Finally we run OLS on them and get the residuals, (x1^ - x1^~)            
            x1DoubleHat = p_OLS.p_OLS_raw(x1Hat, x2Hat, self.seMethod, self.cluster).residuals
            
            # And Shea Partial R Square is just the squared correlation 
            # between (x1 - x1~) and (x1^ - x1^~).
            partialRSquare.append(pearsonr(x1DoubleHat, x1Tilde)[0]**2)
        
        # When this is done for all variables, we make it an attribute.
        self.fs_partialRSquare = partialRSquare
    

    def _p_Ftest(self):
        '''
        Implements F test for instruments not relevant and other statistics.
        '''
        # The idea here is for each endogenous variable, to run a
        # normal first stage regression (that is, regress the endogenous
        # variable on the instruments) and get all the attributes from
        # it, including a F test of whether all instruments coefficient
        # equals zero (or, equivalently, whether the model without
        # instruments explain the endogenous variable as well as the
        # model with). For a discussion of Partial F-Statistics, see
        # CT05, p. 105.
        
        # Step 1:
        # Create instruments and exogenous variable matrixes
        # outside of the loop, in order to save resources.
        zInst = self.dataFrame.loc[:, self.instruments].as_matrix()
        zInst.shape = (self.N, len(self.instruments))
        X2 = self.dataFrame.loc[:, self.xVariable].as_matrix()
        X2.shape = (self.N, len(self.xVariable))
        
        # Since we have many endogenous variables, our statistics will 
        # be lists along the number of endogenous variables.
        beta = list(); tValue = list(); pValue = list()
        fTest = list(); fTestValue = list(); stdError = list()
        
        # For each endogenous variable, we run the regression.
        for endog in self.endogenous:
            
            # Our objective is to test p1 = 0 in the regression:
            # x1 = p1 * zInst + p2 * X2, where z is the instruments
            # and X2 the exogenous variables in X.
            # X2 and zInst were already created outside, now we need x1.
            x1 = self.dataFrame.loc[:, endog].as_matrix()
            x1.shape = (self.N, 1)
            
            # And then we run the First Stage regression using the p_OLS_raw
            # object.
            firstStage = p_OLS.p_OLS_raw(x1, np.concatenate([zInst, X2], axis=1), \
                self.seMethod, self.cluster)
        
            # Now we calculate Wald's F test on whether p1 = 0. To do so,
            # we define R = [Ikz, 0], r = [0], and we test whether
            # R * beta - r = 0. See CT05, p.224 for a discussion of Wald
            # Tests. The Wald Test then is:
            # W = h^' [R^ V^ R^']^-1 h^, where h^= h(beta^) and R^
            # is its derivative. Here, since we have a linear constraint,
            # h^(beta) = p1 and R^ = R. Finally, the Wald asymptotic 
            # F-statistic is W/h (where h is number of constraints), which
            # is asymptotically F(h, N - K) distributed.
            
            # First we build R matrix. It is an Identity matrix for 
            # the instruments, and then zero for exogenous variables in X.
            instLen = len(self.instruments)
            bigR = np.concatenate( [np.eye(instLen), \
                np.zeros(shape=(instLen, len(self.xVariable)), dtype=float)], axis=1)
            
            # Now h^ will simply be bigR * beta.
            hHat = np.dot(bigR, firstStage.beta)
            
            # Step 2:
            # And we can start calculating the Wald Statistic.
            
            # First we calculate the middle part, [R^ V^ R^']^-1 ...
            middlePart = np.linalg.inv( np.dot( bigR, np.dot( firstStage.varCoVar, bigR.T ) ) )
            
            # ... and then we multiply the sides.
            wTestSingle = np.dot( hHat.T, np.dot( middlePart, hHat ) )
            
            # As we said, the F test is simply the wTest divided by
            # number of restrictions...
            fTestSingle = 1.0/instLen * wTestSingle
            
            # ... and it (asymptotically) follows a F distribution with 
            # (h, N - K) degrees of freedom.
            fTestValue.append(f.sf(fTestSingle, instLen, self.N - self.Kz))
            fTest.append(fTestSingle)
            
            # And now we calculate other useful statistics...
            tValueSingle = np.divide(firstStage.beta, firstStage.stdError)
            pValueSingle = t.sf(np.abs(tValueSingle), self.N - self.Kz) * 2
            
            # ... and add them to the lists (we are only interested
            # in the instruments).
            beta.append(firstStage.beta[0:instLen])
            stdError.append(firstStage.stdError[0:instLen])
            tValue.append(tValueSingle[0:instLen])
            pValue.append(pValueSingle[0:instLen])
                    
        # Finally, when this is done for all endogenous variables, we 
        # save it as object attributes.    
        self.fs_beta = beta; self.fs_stdError =  stdError; self.fs_tValue = tValue 
        self.fs_pValue = pValue; self.fs_fTest = fTest; self.fs_fTestValue = fTestValue 


    
class p_IV_raw(object):
    '''
    Class for Instrumental Variables linear regression on properly prepared 
    dependent variable (y) and independent variables (X and Z) matrixes. 
    For direct use from pandas DataFrame, see the p_IV inherited class.
    
    __author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
    
    ...
    
    Arguments
    ---------
    
    yMatrix            : numpy.array (Nx1)
                      A Nx1 matrix with N observations of the dependent variable (y).
    xMatrix            : numpy.array(NxK_x)
                      A NxK_x matrix with N observations of the K_x independent 
                      variables (X), including endogenous ones.
    zMatrix            : numpy.array(NxK_z)
                      A NxK_z matrix with N observations of the K_z independent 
                      variables (Z), excluding endogenous variables and including 
                      instruments.
    seMethod        : str
                      One of possible methods of calculating standard errors. Options 
                      are None, which uses standard simplified formula for SE, 'robust'
                      for White-Huber heteroskedasticity-robust SE and 'bootstrap' for 
                      bootstrap-calculated SE. Defaults to 'robust', following discussion 
                      in Cameron & Triverdi (2005), p.74-75.
    clusterDummies    : DataFrame
                      A pandas DataFrame with the cluster variable in dummy format, 
                      if it exists. Defaults to None.
                     
    '''    
    
    def __init__(self, yMatrix, xMatrix, zMatrix, seMethod, clusterDummies):
        '''
        Initializes p_IV_raw and calculates variables of interest.
        '''        
        # Here we do all the algebra in _p_IVreg or _p_TSLS. After it,
        # we calculate the IV statistics.
        
        # Always useful to have No. of observations and variables (x and z).
        self.N = len(xMatrix[:, 0])
        self.Kx = len(xMatrix[0, :])
        self.Kz = len(zMatrix[0, :])
        
        # Step 1:
        # If No. instruments == No. endogenous, the model is just-identified
        # and thus we can use the simpler IV estimator. We do so in
        # _p_IVreg and get the attributes. See the method for details.
        if self.Kx == self.Kz:
            self._p_IVreg(yMatrix, xMatrix, zMatrix, seMethod, clusterDummies)
            
        # Whereas if No. Instruments > No. endogenous, then the model 
        # is overidentified and we need to apply Two Stages 
        # Least Square (2SLS). See CT05, p. 101.
        else:
            self._p_TSLS(yMatrix, xMatrix, zMatrix, seMethod, clusterDummies)
        
        # Step 2:
        # After acquiring the attributes from TSLS or IV, we use them to
        # build useful statistics.
        
        # The T-value is simply coefficient divided by standard error.
        self.tValue = np.divide(self.beta, self.stdError)
    
        # The p-value is the survival function of T-value * 2 (bi-modal), 
        # given N - Kx degrees of freedom.
        self.pValue = t.sf(np.abs(self.tValue), self.N - self.Kx) * 2
    
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
        self.adjRSquare = 1 - (1 - self.rSquare) * (float(self.N - 1)/(self.N - self.Kx - 1))

    
    def _p_IVreg(self, y, X, Z, seMethod, clusterDummies):
        '''
        Runs Instrumental Variables estimation on specified y, X and Z matrixes.
        
        __author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
        
        ...
        
        Arguments
        ---------
        
        y                : numpy.array (Nx1)
                          A Nx1 matrix with N observations of the dependent variable (y).
        X                : numpy.array(NxKx)
                          A NxKx matrix with N observations of the Kx independent
                          (possibly endogenous) variables (X).
        Z                : numpy.array(NxKz)
                          A NxKz matrix with N observations of the Kz instruments (Z).
        intercept        : bool
                          Whether to add an intercept to regression. Defaults to True.
        seMethod        : str
                          One of possible methods of calculating standard errors. Options 
                          are None, which uses standard simplified formula for SE, 'robust'
                          for White-Huber heteroskedasticity-robust SE and 'bootstrap' for 
                          bootstrap-calculated SE. Defaults to 'robust', following discussion 
                          in Cameron & Triverdi (2005), p.74-75.
        clusterDummies    : DataFrame
                          A pandas DataFrame with the cluster variable in dummy format, 
                          if it exists. Defaults to None.                         
        '''            
        # Is go time! This is the IV estimator, see CT05, p. 98.
        
        # Step 1:
        # To calculate the coefficients (beta), we use the IV 
        # estimator: beta = (Z'X)^-1 * Z'y  (see CT05, p.100).
        ZX = np.dot(Z.T, X)
        
        # First it is useful to check if the matrix is singular.
        if np.linalg.det(ZX) != 0:
            ZX_inv = np.linalg.inv(ZX)
        else:
            raise ValueError("Z'X matrix is singular!")        
        Zy = np.dot(Z.T, y)        
        beta = np.dot(ZX_inv, Zy)
        
        # The residuals are therefore the difference between
        # actual values y and predicted values X * beta^
        residuals = y - np.dot(X, beta)

        # Step 2:
        # Now we calculate the Covariance Matrix (VarCoVar)
        # with method depending on chosen SE Method        
        if clusterDummies is not None:
            # We still haven't implemented cluster SE for IV.
            raise NotImplementedError('Sorry!')
            
        else:
            if seMethod is 'robust':
                # Heteroskedasticity-robust Std. Errors.
                # We implement it according to CT05, p. 101
                # The formula is:
                # (Z'X)^-1 * Z' Sigma Z * (Z'X)^-1

                # First we build the Sigma^ matrix by making it a 
                # diagonal matrix with the squared residuals.
                uSquare = np.square(residuals)
                uSquare.shape = (self.N, )    
                
                # We use a sparse matrix to keep sigma, as all non-diagonal
                # elements are zero (and the matrix can get quite big!).    
                sigma = sparse.diags(uSquare, 0)
                sigma_ZZ = np.dot(Z.T, sigma.dot(Z))
                varCoVar = float(self.N)/(self.N - self.Kx) * \
                    np.dot(np.dot(ZX_inv, sigma_ZZ), ZX_inv)
        
                # From the VarCoVar we can get the standard errors
                # simply picking from the diagonal.
                stdError = np.sqrt(np.diagonal(varCoVar))
                stdError.shape = (self.Kx, 1)
                
            elif seMethod is None or seMethod=='None':
                # Homoskedasticity Restricted Std. Errors.
                # These SE are only consistent if the errors are
                # homosketastic. They are included only for completeness,
                # as they rarely apply.            
                # The formula is: s^2 * (Z'X)^-1 * Z'Z * (Z'X)^-1

                sSquare = np.square(residuals).sum()
                sigma = float(sSquare * self.N)
                ZZ = np.dot(Z.T, Z)
                varCoVar = sSquare/(self.N - self.Kx) * np.dot(ZX_inv, np.dot(ZZ, ZX_inv))
                
                # From the VarCoVar we can get the standard errors
                # simply picking from the diagonal.
                stdError = np.sqrt(np.diagonal(varCoVar))
                stdError.shape = (self.Kx, 1)
                
            elif seMethod is 'bootstrap':
                # We still haven't implemented bootstrap SE for IV
                # (although it shouldn't be very difficult).
                raise NotImplementedError('Sorry!')
            
            else:
                raise ValueError('Variable seMethod was not given a valid input.' 
                    'Please try: "bootstrap", "robust" or None.')
        
        # Finally, we define p_IV_raw attributes with the important variables.
        # They will be inherited by p_IV class and become our results.
        self.beta = beta; self.stdError = stdError; self.varCoVar = varCoVar
        self.residuals = residuals; self.fitted = np.dot(X, beta)


    def _p_TSLS(self, y, X, Z, seMethod, clusterDummies):
        '''
        Runs Two Stages Least Squares estimation on specified y, X and Z matrixes.
        
        __author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
        
        ...
        
        Arguments
        ---------
        
        y                : numpy.array (Nx1)
                          A Nx1 matrix with N observations of the dependent variable (y).
        X                : numpy.array(NxKx)
                          A NxKx matrix with N observations of the Kx independent 
                          (possibly endogenous) variables (X).
        Z                : numpy.array(NxKz)
                          A NxKz matrix with N observations of the Kz instruments (Z).
        intercept        : bool
                          Whether to add an intercept to regression. Defaults to True.
        seMethod        : str
                          One of possible methods of calculating standard errors. Options 
                          are None, which uses standard simplified formula for SE, 'robust' 
                          for White-Huber heteroskedasticity-robust SE and 'bootstrap' for 
                          bootstrap-calculated SE. Defaults to 'robust', following discussion 
                          in Cameron & Triverdi (2005), p.74-75.
        clusterDummies    : DataFrame
                          A pandas DataFrame with the cluster variable in dummy format,
                          if it exists. Defaults to None.                         
        '''            
        # Is go time! This is the TSLS estimator, see CT05, p. 101.
        
        # Step 1:
        # To calculate the coefficients (beta), we use the TSLS 
        # estimator: (quite obtuse)
        # beta = [X'Z(Z'Z)^-1 Z'X]^-1 [X'Z (Z'Z)^-1 Z'y]
        ZX = np.dot(Z.T, X)
        XZ = np.dot(X.T, Z)
        Zy = np.dot(Z.T, y)
        ZZ = np.dot(Z.T, Z)
        
        # First it is useful to check if the matrix is singular.
        if np.linalg.det(ZZ) != 0:
            ZZ_inv = np.linalg.inv(ZZ)
        else:
            raise ValueError("Z'Z matrix is singular!")

        # We should also check if [X'Z(Z'Z)^-1 Z'X] is singular
        bigMatrix = np.dot(XZ, np.dot(ZZ_inv, ZX)) 
        if np.linalg.det(bigMatrix) != 0:
            bigInv = np.linalg.inv(bigMatrix)
        else:
            raise ValueError("X'Z(Z'Z)^-1 Z'X matrix is singular!")
        
        # Now we just finish the matrix algebra.
        bigY = np.dot(XZ, np.dot(ZZ_inv, Zy))    
        beta = np.dot(bigInv, bigY)
        
        # The residuals are therefore the difference between
        # actual values y and predicted values X * beta^
        residuals = y - np.dot(X, beta)
        
        if clusterDummies is not None:
            # We still haven't implemented cluster SE for IV.
            raise NotImplementedError('Sorry!')
        else:
            if seMethod is 'robust':
                # Heteroskedasticity-robust Std. Errors.
                # We implement it according to CT05, p. 102
                # The formula is even more upsetting:
                # V = N * [X'Z (Z'Z)^-1 Z'X]^-1 [X'Z (Z'Z)^-1 Z' Sigma
                # Z (Z'Z)^-1 Z'X] [X'Z (Z'Z)^-1 Z'X]^-1
                
                # First we build the Sigma^ matrix by making it a 
                # diagonal matrix with the squared residuals.                
                uSquare = np.square(residuals)
                uSquare.shape = (self.N, )
                
                # We use a sparse matrix to keep sigma, as all non-diagonal
                # elements are zero (and the matrix can get quite big!).
                sigma = sparse.diags(uSquare, 0)
                sigma_ZZ = np.dot(Z.T, sigma.dot(Z))
                
                # Now we build the huge monsters. Luckily the leftmost
                # and rightmost parts we already computed: they are
                # the first part of beta formula (bigInv).
                # Only remains to calculate the middle.                
                zMiddle = np.dot( np.dot(XZ, ZZ_inv), np.dot(sigma_ZZ, np.dot(ZZ_inv, ZX) ))
                
                # Finally...
                varCoVar = float(self.N)/(self.N - self.Kx) * \
                    np.dot(np.dot(bigInv, zMiddle), bigInv)
        
                # From the VarCoVar we can get the standard errors
                # simply picking from the diagonal.
                stdError = np.sqrt(np.diagonal(varCoVar))
                stdError.shape = (self.Kx, 1)
                
            elif seMethod is None or seMethod is 'None':
                # Homoskedasticity Restricted Std. Errors.
                # These SE are only consistent if the errors are
                # homosketastic. They are included only for completeness,
                # as they rarely apply. See CT05, p. 102.            
                # The formula is: s^2 * [X'Z (Z'Z)^-1 Z'X]^-1
                
                sSquare = np.square(residuals).sum()
                sigma = float(sSquare * self.N)
                varCoVar = sSquare/(self.N - self.Kx) * bigInv
            
                # From the VarCoVar we can get the standard errors
                # simply picking from the diagonal.
                stdError = np.sqrt(np.diagonal(varCoVar))
                stdError.shape = (self.Kx, 1)
                
            elif se_method is 'bootstrap':
                # We still haven't implemented bootstrap SE for IV
                # (although it shouldn't be very difficult).
                raise NotImplementedError('Sorry!')        
        
        # Finally, we define p_IV_raw attributes with the important variables.
        # They will be inherited by p_IV class and become our results.
        self.beta = beta; self.stdError = stdError; self.varCoVar = varCoVar
        self.residuals = residuals; self.fitted = np.dot(X, beta)
        
        
        
class p_IV(p_IV_raw, p_IV_FS):
    '''
    Runs Instrumental Variables Linear Regression approach on chosen 
    variables of a pandas DataFrame.  See p_iv_first_stage for first stage 
    results and statistics.
    
    __author__ = 'Pedro Forquesato <pedro.forquesato@puc-rio.br>'
    
    ...
    
    Arguments
    ---------
    
    dataFrame        : DataFrame
                      The Pandas dataframe to which yVariable, xVariable, 
                      endogenous and instruments belong.
    yVariable        : str
                      A string with the name of the variable (column) in dataFrame
                      to be used as the dependent variable. 
    xVariable        : list
                      List of strings with names of variables (columns) in dataFrame
                      to be used as independent variables. Factors have to be marked
                      by adding 'factor:' in front of the variable.
    endogenous        : list
                      List of strings with names of variables (columns) in dataFrame
                      considered endogenous and thus being instrumented by variables 
                      in instruments. These variables should NOT be added to xVariable. 
                      Factors can be marked by adding 'factor:' in front of the variable.
    instruments        : list
                      List of strings with names of variables (columns) in dataFrame
                      to be used as instruments. These should include ONLY variables
                      not already included in xVariable. Variables in xVariable which 
                      are not in endogenous are automatically instrumented by
                      themselves. Factors can be marked by adding 'factor:' in front 
                      of the variable.
    intercept        : bool
                      Whether to add an intercept to regression. Defaults to True.
    seMethod        : str
                      One of possible methods of calculating standard errors. Options 
                      are None, which uses standard simplified formula for SE, 'robust'
                      for White-Huber heteroskedasticity-robust SE and 'bootstrap' for 
                      bootstrap-calculated SE. Defaults to 'robust', following discussion 
                      in Cameron & Triverdi (2005), p.74-75.
    cluster            : str
                      Name of the variable that defines the clusters, if those exist. 
                      Defaults to None. If a string is given, then Cluster-robust 
                      standard errors are calculated, and any value given to se_method 
                      is ignored, except if given 'bootstrap', when Bootstrap clustered SE
                      are calculated instead.  Designed for SMALL clusters, such that Variance 
                      Matrix can be properly estimated.
    firstStage        : bool or None
                      Boolean denoting whether First Stage statistics should be calculated. 
                      Defaults to True. To change whether the results are shown, change 
                      firstStage input in outPrint method.
    autoPrint        : bool
                      Whether should automatically print results. Defaults to True.
                     
    '''
    
    def __init__(self, dataFrame, yVariable, xVariable, endogenous, instruments, \
                    intercept=True, seMethod='robust', cluster=None, firstStage=True, autoPrint=True):
        '''
        Initializes p_IV, formats DataFrame and prepares matrixes for initializating p_IV_raw.
        '''    
        # The plan is to fix the DataFrame and variables, so that we can call p_IV_raw
        # which effectively implements the IV regression.

        # Step 1:
        # First we initialize variables, making sure to copy and not
        # link to variables.
        self.dataFrame = pd.DataFrame(dataFrame)
        self.yVariable = yVariable
        self.xVariable = list(xVariable)
        self.endogenous = list(endogenous)
        self.instruments = list(instruments)
        self.seMethod = seMethod
        self.cluster = cluster
        self.firstStage = firstStage
        
        # Further, we add intercept (a column of 1s) if wanted 
        # (that is, if intercept is set to true)
        if intercept:
            self.dataFrame['Intercept'] = 1
            self.xVariable = ['Intercept'] + self.xVariable
        
        # Then we call _p_Dummify function to fix DataFrame
        #  for running p_IV_raw. See method for details.
        self.dataFrame, self.xVariable, self.endogenous, self.instruments = \
            self._p_Dummify(self.dataFrame, self.yVariable, self.xVariable, \
            self.endogenous, self.instruments, self.cluster)
        
        # Asking endogenous and instruments not to be included in 
        # xVariable simplifies the process of building all lists of
        # variables of interest. In particular the total set of 
        # instruments and total set of X variables.
        self.zVariable = self.xVariable + self.instruments
        self.augXVariable = self.xVariable + self.endogenous
        
        # Step 2:
        # It will be useful to have defined the No. observations and
        # No. of (x and z) variables.
        self.N = len(self.dataFrame)
        self.Kx = len(self.augXVariable)
        self.Kz = len(self.zVariable)
        
        # If we have clusters, we need to build dummies for it so
        # we can calculate std. errors later.
        if self.cluster is not None:
            clusterDummies = pd.get_dummies(self.dataFrame[self.cluster], \
                prefix='clstr')
        else:
            clusterDummies = None
        
        # Before actually running the model, we need to check if it is
        # not underidentified, and thus not estimable.
        if self.Kz < self.Kx:
            raise ValueError('This model is underidentified! There are more'
                'endogenous variables than instruments.')
        
        # Step 3:
        # p_IV_raw runs on y, X and Z matrixes, so we need to build them,
        # making sure they are 2-dimensional numpy arrays for matrix algebra.
        # matrix.shape will also call error in case the matrix size  
        # is wrong, making debugging easier.
        X = self.dataFrame.loc[:, self.augXVariable].as_matrix()
        X.shape = (self.N, self.Kx)
        y = self.dataFrame.loc[:, self.yVariable].as_matrix()
        y.shape = (self.N, 1)
        Z = self.dataFrame.loc[:, self.zVariable].as_matrix()
        Z.shape = (self.N, self.Kz)
        
        # If firstStage is true, then inherits from firstStage class
        # all first stage statistics.
        if self.firstStage:
            p_IV_FS.__init__(self, self.dataFrame, self.endogenous, self.instruments, \
                self.xVariable, self.seMethod, self.cluster)        
                        
        # Calls inheriting class p_IV_raw, and inherit its attributes.
        p_IV_raw.__init__(self, y, X, Z, self.seMethod, clusterDummies)
        
        # If auto_print is True, then print out results (automatically).
        if autoPrint:
            self.outPrint()

    
    def _p_Dummify(self, dataFrame, yVariable, xVariable, endogenous, instruments, cluster):
        '''
        Prepares DataFrame and variable lists for transforming them into 
        matrixes used for IV algebra. Most work involves transforming factor 
        variables into dummies.
        '''
        # Here we fix the DataFrame to be able to run IV.        
        
        # Step 1:
        # Before we start, we check if inputs are correct. They should 
        # all be mutually exclusive.
        for endog in endogenous:
            if endog in xVariable:
                raise ValueError('Endogenous variables must be given'
                    'separately from xVariable!')
        for exog in instruments:
            if exog in xVariable:
                raise ValueError('Instruments cannot be part of xVariable!')
        
        # We accept factors without denomination when their type is 
        # pandas object. To do so, we simply put 'factor:' in front ourselves.
        for varIndex, varName in enumerate(xVariable):
            if varName in dataFrame.columns:
                if dataFrame[varName].dtype == 'O':
                    xVariable[varIndex] = 'factor:' + varName
        for varIndex, varName in enumerate(instruments):
            if varName in dataFrame.columns:
                if dataFrame[varName].dtype == 'O':
                    instruments[varIndex] = 'factor:' + varName
        for varIndex, varName in enumerate(endogenous):
            if varName in dataFrame.columns:
                if dataFrame[varName].dtype == 'O':
                    endogenous[varIndex] = 'factor:' + varName
    
        # Dealing with factors is tricky. We create a list of dummies
        # for each variables list (xVariable, endogenous and instruments)
        # and make sure the list of variables is always pointing to the
        # right name.
        xDummyList = list()
        for varIndex, varName in enumerate(xVariable):
            # Scan through all X variables, if any is a factor...
            if varName.startswith('factor:'):        
                # ... we need to remove the 'factor' from the name
                # (since it doesn't exist in the DataFrame)
                # and mark it as dummy (by adding to dummyList).
                xVariable[varIndex] = varName.replace('factor:', '')
                xDummyList.append(xVariable[varIndex])            
        # The same for instrument...
        instDummyList = list()
        for varIndex, varName in enumerate(instruments):
            if varName.startswith('factor:'):        
                instruments[varIndex] = varName.replace('factor:', '')
                instDummyList.append(instruments[varIndex])                
        # ... and for endogenous.
        endogDummyList = list()
        for varIndex, varName in enumerate(endogenous):
            if varName.startswith('factor:'):        
                endogenous[varIndex] = varName.replace('factor:', '')
                endogDummyList.append(instruments[varIndex])    
                    
        # Step 2:
        # Before creating dummies, we remove all NA and 
        # save space by removing not needed variables
        # (also important to keep all numeric).
        if cluster is not None and cluster not in xVariable \
            and cluster not in zVariable:
            # Makes sure we keep the cluster in dataFrame...
            dataFrame = dataFrame.loc[:, xVariable  + [yVariable] + \
            instruments + endogenous + [cluster]].dropna()
        else:
            dataFrame = dataFrame.loc[:, xVariable + [yVariable] + \
                instruments + endogenous].dropna()
        
        # Step 3:
        # Now we create the dummies to substitute for the factor.
        for vfactor in xDummyList:
            # Pandas makes the trouble of actually creating the dummies
            # a breeze. But we still need to fix namings.            
            xDummies = pd.get_dummies(dataFrame[vfactor], prefix=vfactor)
            xVariable.remove(vfactor) # Remove factor from X
            
            # Always helpful to check if factor has more than one value,
            # to avoid singular matrixes.
            if len(xDummies.columns) == 1:
                raise Exception('This factor has only one unique value!')
            
            else:
                # Remove one of the dummies, to avoid the 
                # 'Dummy variable trap'.        
                xDummies = xDummies.iloc[:, 1:]
                # And then add them to DataFrame and X variable names.
                dataFrame = dataFrame.join(xDummies)
                xVariable.extend(xDummies.columns)    
        # We do the same for instruments...
        for vfactor in instDummyList:
            instDummies = pd.get_dummies(dataFrame[vfactor], prefix=vfactor)
            instruments.remove(vfactor)
            if len(instDummies.columns) == 1:
                raise Exception('This factor has only one unique value!')    
            else:
                instDummies = instDummies.iloc[:, 1:]
                dataFrame = dataFrame.join(instDummies)
                instruments.extend(instDummies.columns)
        # ... and for endogenous.
        for vfactor in endogDummyList:
            endogDummies = pd.get_dummies(dataFrame[vfactor], prefix=vfactor)
            endogenous.remove(vfactor)
            if len(endogDummies.columns) == 1:
                raise Exception('This factor has only one unique value!')    
            else:
                endogDummies = endogDummies.iloc[:, 1:]
                dataFrame = dataFrame.join(endogDummies)
                endogenous.extend(endogDummies.columns)
    
        return dataFrame, xVariable, endogenous, instruments

        
    def outPrint(self, printOpt='table', output=None, firstStage = None):
        '''
        Prints the OLS parameters and statistics to either terminal or 
        file (available in latex).
        
        Arguments
        ---------            
        printOpt            : str
                              How should output be returned. Defaults to 
                              'table', with results printed as a table. Other option 
                              is 'latex' for latex formatting.
        output                : str
                              The name of the file where the output should be printed. 
                              Defaults to None, which means output is printed in console.
        firstStage            : bool or None
                              Boolean denoting whether First Stage statistics should 
                              be shown. Defaults to None, using the p_IV initialized 
                              value self.firstStage.
        '''
        # This function prints the IV results in a table format.
        
        # Before we start, if firstStage is None, then use the attribute.
        if firstStage is None:
            firstStage = self.firstStage
            
        # Step 1:
        # First we prepare the Header and the table with the results in a format
        # that the package tabulate can accept.                
        headers = ['Variable', 'Coefficient', 'Std. Errors', 'T Value', 'P Value']
        table = list()
        for i, var in enumerate(self.augXVariable):
            table.append([var, self.beta[i, 0], self.stdError[i, 0], \
                self.tValue[i, 0], self.pValue[i, 0]])
        
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
        else:
            # We already raised exception for wrong seMethod in p_IV_raw.
            tpSE = 'Homoskedastic Restricted'
        
        # Step 2:
        # Above the table we print general statistics of the model (N, R^2, etc.)
        info = [['Dep. Variable:', self.yVariable], ['Model:', 'Instrumental Variables'], 
                ['Standard Errors:', tpSE], ['No. Observations:', str(self.N)], 
                ['No. Variables:', str(self.Kx)], ['R Square:', '%.3f' % self.rSquare], 
                ['Adj. R Square', '%.3f' % self.adjRSquare]]
        
        # Step 3:
        # Now we prepare a table for each endogenous variable with
        # First Stage statistics that we obtained in p_IV_FS (if 
        # we did ask it to be computed and printed out).
        if firstStage and self.firstStage:
            firstStageOut = list()
            for i in range(len(self.endogenous)):
                firstStageOut.append([['Endogenous Variable:', self.endogenous[i]],
                            ['First Stage:', ', '.join(self.instruments)], 
                            ['Partial R Square:', '%.3f' % self.fs_partialRSquare[i]],
                            ['Coefficient:', ', '.join(['%.3f' % self.fs_beta[i][k, 0] \
                                for k in range(len(self.instruments))])], 
                            ['T Value:', ', '.join(['%.1f' % self.fs_tValue[i][k, 0] \
                                for k in range(len(self.instruments))])],
                            ['P Value:', ', '.join(['%.3f' % self.fs_pValue[i][k, 0] \
                                for k in range(len(self.instruments))])],
                            ['F Test:', '%.1f' % self.fs_fTest[i]], 
                            ['F Test P-value:', '%.3f' % self.fs_fTestValue[i]]])
        
        # Step 4: Print out the output.
        if printOpt not in ['table', 'latex']:
            # First we check for wrong printOpt input.        
            raise ValueError('Variable printOpt was not given an acceptable string.'
                'Try: "table" or "latex".')
        else:
            # Otherwise:
            if output is None:
                if printOpt is 'table':
                    if firstStage and self.firstStage:
                        for i in range(len(self.endogenous)):
                            print tabulate(firstStageOut[i], floatfmt='.4f', \
                                tablefmt='rst')
                    print tabulate(info, floatfmt='.4f', tablefmt='rst')
                    print tabulate(table, headers=headers, floatfmt=".4f", \
                        tablefmt='rst')
            
                elif print_opt is 'to_latex':
                    if firstStage and self.firstStage:
                        for i in range(len(self.endogenous)):
                            print tabulate(firstStageOut[i], floatfmt='.4f', \
                                tablefmt='latex')
                    print tabulate(info, floatfmt='.4f', tablefmt='latex')
                    print tabulate(table, headers=headers, tablefmt="latex", \
                        floatfmt=".4f")
                    
            else:
                # If output is not None, we print to a file. First we open it...
                f = open(output + '.txt', 'w')
                
                # ... then we print...
                if printOpt is 'table':
                    if firstStage and self.firstStage:
                        for i in range(len(self.endogenous)):
                            print >> f, tabulate(firstStageOut[i], floatfmt='.4f', \
                                tablefmt='rst')
                    print >> f, tabulate(info, floatfmt='.4f', tablefmt='rst')
                    print >> f, tabulate(table, headers=headers, floatfmt=".4f", \
                        tablefmt='rst')
            
                elif printOpt is 'latex':
                    if Fstage and self.FirstStage:
                        for i in range(len(self.endogenous)):
                            print >> f, tabulate(firstStageOut[i], floatfmt='.4f', \
                                tablefmt='latex')
                    print >> f, tabulate(info, floatfmt='.4f', tablefmt='latex')    
                    print >> f, tabulate(table, headers=headers, tablefmt="latex", \
                        floatfmt=".4f")
                
                # ... then we close it.        
                f.close()    

if __name__ == '__main__':
    pass
    # Put a test data here. 
