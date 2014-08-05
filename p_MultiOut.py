from tabulate import tabulate
import pandas as pd

import p_OLS
import p_IV
import p_FE

def p_MultiOut(listReg, varsToPut, headers=None, printOpt='table', \
                    output=None, withStars=True, starModels = ['***', '**', '*', '.'], \
                    rowHeaders=None):
    '''
    Prints a table with the results of multiple regressions to either terminal or 
    file (available in latex).
    
    Arguments
    ---------            
    listReg      : list
                   A list of the regressions to be added to the output. 
                   They must be p_OLS, p_IV or p_FE objects.
    varsTP       : list
                   List of variables that should be included in the table. 
                   They don't have to belong to X in all regressions.
    headers      : list or None
                   List of headers for the columns. If None (Default) is given, 
                   a header with (1), (2), etc. is created.
    printOpt     : str
                   How should output be returned. Defaults to 'to_print', with 
                   results printed as a table. Other option is 'latex' for 
                   latex formatting.
    output       : str
                   The name of the file where the output should be printed. 
                   Defaults to None, which means output is printed in console.
    withStars    : bool
                   Whether or not the table should contain stars representing 
                   P-values.
    starModels   : list
                   List of symbols that should represent (respectively) 0.001, 
                   0.01, 0.05 and 0.1 P-values. 
                   Defaults to ('***', '**', '*', '.'), as in R.
    rowHeaders   : list or None
                   List of headers for rows (i.e. variables). If None is 
                   given (Default), the names of the variables will be used.
    '''
    # The objective of this function is to print the result of multiple
    # regressions in a single table, as it is usually done in 
    # economics.
    
    # Step 1:
    # First we get the number of regressions to put on the table.
    nRegs = len(listReg)
    
    # The input headers allow us to decide the name of the columns
    # in the table (whereas each column is a regression).
    # If no value is given, the default is to enumerate them from (1)
    # to (M).
    if headers is None:
        headers = ['(' + str(i + 1) + ')' for i in range(len(listReg))]
    
    # We also allow the possibility of changing row names (i.e.
    # names of variables, in case the original names are not
    # very communicative (e.g. VAR1).
    # If no row headers are given, however, we use variable names.
    if rowHeaders is None:
        rowHeaders = varsToPut
        
    # LATER WE SHOULD DEAL WITH FACTORS
    # As of the present, the user must add factor names when calling
    # varsToPut
    
    # Now depending on the method used, the list of variable names
    # might be a different one. We check that here.
    regName = list()
    for reg in listReg:
        if reg.__class__.__name__ is 'p_OLS' or \
            reg.__class__.__name__ is 'p_FE':
            regName.append(reg.xVariable)
        elif reg.__class__.__name__ is 'p_IV':
            regName.append(reg.augXVariable)
        else:
            raise ValueError('Unfortunately this function is only available' + \
                ' for p_metrics objects.') 
                
    #  Another point to note is that some regressions
    # might not have the variables at all. To deal with this,
    # we create varInOut, which is a bool that for each variable
    # in varsToPut says if it exists in that regression or not...
    varInOut = list()
    for i, reg in enumerate(listReg):
        varInOut.append([varsToPut[x] in regName[i] \
            for x in range(len(varsToPut))])
    
    # ... we use it to be able to fill completely varsIndx below,
    # even for variables that don't exist in the regression (putting
    # it None). varsIndx gives the index of each variable in that
    # regression, as in different regressions those same
    # variables will have different indexes. 
    varsIndx = list()
    for regI in range(len(listReg)):
        # So for each regression, we add to varsIndx the list of indexes
        # of each variable such that varInOut is True, and None
        # otherwise.
        varsIndx.append([regName[regI].index(varsToPut[i]) \
            if varInOut[regI][i] else None for i in range(len(varsToPut))])        
    
    # Step 2:
    # Now we are ready to start building the table itself.
    
    # First, we want the table to present what kind of regression
    # it is.
    methodList = list()
    for reg in listReg:
        if reg.__class__.__name__ is 'p_OLS' or \
            (reg.__class__.__name__ is 'p_FE' and reg.effect is 'ols'):
            methodList.append('OLS')
        elif reg.__class__.__name__ is 'p_IV':
            methodList.append('IV')
        elif reg.__class__.__name__ is 'p_FE' and reg.effect is not 'ols':
            methodList.append('FE')
                
    # Now we build the table, with regressions as columns, and for each
    # variable a row for coefficient and another for the std. error
    table = list()
    for vtpI in xrange(len(varsToPut)):
        # For each variable, we append the name and coefficients 
        # of the regression, for the regressions that have this variable.
        table.append([rowHeaders[vtpI]] + ['%.4f' % reg.beta[varsIndx[i][vtpI], 0] \
            if varsIndx[i][vtpI] is not None else '' for (i, reg) in enumerate(listReg)])
                            
        # Now we build the stars to mark p-values, if so desired (i.e.
        # if withStars is True).
        if withStars:
            # Get P-values of the regressions that have this variable.
            # Putting 1 in regressions that doesn't have it ensure that
            # it will append '' (i.e. nothing).
            pValues = [reg.pValue[varsIndx[i][vtpI], 0] if \
                varsIndx[i][vtpI] is not None else 1 for i, reg in enumerate(listReg)]
           
            # Now we set the p-values to stars according to value
            # and to the starModels given as input.
            pStars = list()
            for pv in pValues:
                if pv < 0.001:
                    pStars.append(starModels[0])
                elif pv < 0.01:
                    pStars.append(starModels[1])
                elif pv < 0.05:
                    pStars.append(starModels[2])
                elif pv < 0.1:
                    pStars.append(starModels[3])
                else:
                    pStars.append('')
            
            # Finally, we are going to add another row with the std 
            # errors. In this case, we also add the stars signifying
            # p-value zone.
            table.append([''] + ['(' + '%.4f' % reg.stdError[varsIndx[i][vtpI], 0] + 
                 ')' + pStars[i] if varsIndx[i][vtpI] is not None \
                 else '' for i, reg in enumerate(listReg)])
        else:                     
            # Otherwise, if withStars is False, then we just add the std
            # errors (between parenthesis).
            table.append([''] + ['(' + '%.4f' % reg.stdError[varsIndx[i][vtpI], 0] + 
                ')' if varsIndx[i][vtpI] is not None \
                else '' for i, reg in enumerate(listReg)])
    
    # Under the main table, we also want to add other useful statistics,
    # as No Observations, (Adj.) R square, the (FE) method and
    # the method of calculating SE.                              
    table.append(['N'] + [str(reg.N) for reg in listReg])
    table.append(['Adj. R Square'] + ['%.4f' % reg.adjRSquare for reg in listReg])
    table.append(['Method'] + methodList)
    table.append(['Cluster'] + [str(reg.cluster).capitalize() for reg in listReg])
    table.append(['Std Errors'] + [str(reg.seMethod).capitalize() for reg in listReg])
    
    # Step 3:
    # Finally, we print the results using tabulate.
    if output is None:
        if printOpt is 'table':
            print tabulate(table, headers=headers, floatfmt=".4f", tablefmt='rst')
        elif printOpt is 'latex':
            print tabulate(table, headers=headers, floatfmt=".4f", tablefmt='latex')
        else:
            raise ValueError('Variable printOpt input is not acceptable. Try "table" or "latex".')
    else:
        # If output is not None, we print to a file. First we open it...
        f = open(output + '.txt', 'w')
        
        # ... then we print...
        if print_opt == 'table':
            print >> f, tabulate(table, headers=headers, floatfmt=".4f", tablefmt='rst')
        elif print_opt == 'latex':
            print >> f, tabulate(table, headers=headers, floatfmt=".4f", tablefmt='latex')
        else:
            raise ValueError('Variable printOpt input is not acceptable. Try "table" or "latex".')
        
        # ... and then we close it.  
        f.close()
   
