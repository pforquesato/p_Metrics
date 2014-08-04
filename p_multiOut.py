from tabulate import tabulate
import pandas as pd

import p_OLS
import p_IV
import p_FE

def p_multiOut(list_reg, vars_tp, headers=None, print_opt='print', \
					output=None, withstars=True, star_models = ['***', '**', '*', '.'], \
					row_headers=None):
	'''
	Prints a table with the results of multiple regressions to either terminal or 
	file (available in latex).
	
	Arguments
	---------			
	list_reg			: list
						  A list of the regressions to be added to the output. They must
						  be p_OLS, p_IV or p_FE objects.
	vars_tp				: list
						  List of variables that should be included in the table. They don't have
						  to belong to X in all regressions.
	headers				: list or None
						  List of headers for the columns. If None (Default) is given, 
						  a header with (1), (2), etc. is created.
	print_opt			: str
						  How should output be returned. Defaults to 'to_print', with 
						  results printed as a table. Other option is 'latex' for latex formatting.
	output				: str
						  The name of the file where the output should be printed. Defaults to
						  None, which means output is printed in console.
	withstars			: bool
						  Whether or not the table should contain stars representing P-values.
	star_models			: list
						  List of symbols that should represent (respectively) 0.001, 0.01, 0.05 and
						  0.1 P-values. Defaults to ('***', '**', '*', '.'), as in R.
	row_headers			: list or None
						  List of headers for rows (i.e. variables). If None is given (Default),
						  the names of the variables will be used.
	'''
	# Get basic parameters
	n_regs = len(list_reg)
	
	# Change headers
	if headers is None:
		headers = ['(' + str(i + 1) + ')' for i in range(len(list_reg))]
	
	# Change row headers
	if row_headers is None:
		row_headers = vars_tp
		
	# Later need to deal with factors
		
	# Get list of (x) variable names
	reg_name = list()
	for reg in list_reg:
		if type(reg) is p_OLS.p_OLS:
			 reg_name.append(reg.Xvar)
		elif type(reg) is p_IV.p_IV:
			reg_name.append(reg.AugXvar)
		elif type(reg) is p_FE.p_FE:
			reg_name.append(reg.xVar)
				
	# Now we select the variables to put in output 
	var_in_out = list()
	for i, reg in enumerate(list_reg):
		var_in_out.append([vars_tp[x] in reg_name[i] for x in range(len(vars_tp))])
		
	# Now get indexes of these variables
	vars_indx = list()
	for reg_i in range(len(list_reg)):
		vars_indx.append([reg_name[reg_i].index(vars_tp[i]) if var_in_out[reg_i][i] else None for i in range(len(vars_tp))])		
	
	# Creates a list of method
	method_list = list()
	for reg in list_reg:
		if type(reg) == p_OLS.p_OLS:
			method_list.append('OLS')
		elif type(reg) == p_IV.p_IV:
			method_list.append('IV')
		elif type(reg) == p_FE.p_FE:
			method_list.append('FE')
				
	# and make a table
	table = list()
	for vtp_i in xrange(len(vars_tp)):
		# Append Coefficients
		table.append([row_headers[vtp_i]] + ['%.4f' % reg.Beta[vars_indx[i][vtp_i], 0] if vars_indx[i][vtp_i] is not None \
							else '' for (i, reg) in enumerate(list_reg)])
							
		# Make stars
		if withstars:
			# Get P values
			p_values = [reg.P_value[vars_indx[i][vtp_i], 0] if vars_indx[i][vtp_i] is not None \
							else 1 for i, reg in enumerate(list_reg)]
			# Now set them to stars according to value
			p_stars = list()
			for pv in p_values:
				if pv < 0.001:
					p_stars.append(star_models[0])
				elif pv < 0.01:
					p_stars.append(star_models[1])
				elif pv < 0.05:
					p_stars.append(star_models[2])
				elif pv < 0.1:
					p_stars.append(star_models[3])
				else:
					p_stars.append('')
			
			# Append Std. Errors
			table.append([''] + ['(' + '%.4f' % reg.StdError[vars_indx[i][vtp_i], 0] + ')' + p_stars[i] \
						if vars_indx[i][vtp_i] is not None else '' for i, reg in enumerate(list_reg)])
		else: 					
			# Append Std. Errors
			table.append([''] + ['(' + '%.4f' % reg.StdError[vars_indx[i][vtp_i], 0] + ')' if vars_indx[i][vtp_i] is not None \
									else '' for i, reg in enumerate(list_reg)])
	
	# Append other statistics								
	table.append(['N'] + [str(reg.N) for reg in list_reg])
	table.append(['Adj. R Square'] + ['%.4f' % reg.Adj_R_square for reg in list_reg])
	table.append(['Method'] + method_list)
	table.append(['Std Errors'] + [str(reg.SE_method).capitalize() for reg in list_reg])
	
	# Prints it
	if output is None:
		if print_opt == 'print':
			print tabulate(table, headers=headers, floatfmt=".4f", tablefmt='rst')
		elif print_opt == 'latex':
			print tabulate(table, headers=headers, floatfmt=".4f", tablefmt='latex')
		else:
			raise ValueError('print_opt not acceptable. Try "print" or "latex".')
	else:
		f = open(output + '.txt', 'w')
		if print_opt == 'print':
			print >> f, tabulate(table, headers=headers, floatfmt=".4f", tablefmt='rst')
		elif print_opt == 'latex':
			print >> f, tabulate(table, headers=headers, floatfmt=".4f", tablefmt='latex')
		else:
			raise ValueError('print_opt not acceptable. Try "print" or "latex".')
		f.close()

if __name__ == '__main__':
	
	# paths
	folder_path = '/home/pedro/Dropbox/doutorado/4o ano/2014 research/networks/'
	df_path = folder_path + 'prepdata/data/MN/final/'

	# read csv file
	df = pd.read_csv(df_path + 'dfMN.csv', low_memory=False)
	
	yv = 'dem_votes'
	xv = ['x_rich', 'x_poorcl', 'factor:YEAR', 'COUNTY']

	t1 = p_OLS.p_OLS(df, yv, xv, auto_print=False)
	t2 = p_OLS.p_OLS(df, yv, xv[0:3], auto_print=False)
	t3 = p_OLS.p_OLS(df, yv, xv[0:2], auto_print=False)
	t4 = p_IV.p_IV(df, yv, ['factor:YEAR'], ['x_rich', 'x_poorcl'], ['nbx_rich', 'nbx_poorcl'], auto_print=False)
	t5 = p_FE.p_FE(df, yv, xv[0:2], indexes=['VTD', 'YEAR'], auto_print=False)
	
	sm = ['*', '.', '', '']
	p_multiOut([t1, t2, t3, t4, t5], ['x_rich', 'x_poorcl', 'YEAR_2008', 'YEAR_2012'], star_models = sm)
	
