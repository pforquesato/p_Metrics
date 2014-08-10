p_Metrics
========

Python implementation of basic econometric models. Currently offers ordinary least squares (OLS), instrumental variables (IV) and panel models, with heteroskedasticity-robust variance matrixes (as default), as well as the possibility of bootstrapped and/or cluster-robust standard errors (and in particular, panel robust SE).

This module is -very- new and incipient, and thus there is absolutely no warranty the methods are correct (although many have been tested using other statistical programs) and many details are not yet implemented. There is also no documentation, although the code itself is heavily commented and should be easy to understand.

This is just a personal project since I grew unhappy with other open-source alternatives for microeconometrics. I put the code here in GitHub in the hopes that it might someday be useful to someone. Any mistakes, problems or tips please let me know.

Of August 9, 2014, our status is:

- Ordinary Least Square, including Heteroskedasticity-robust and clustered SE were tested with STATA 13, and found to be correct. Bootstrap and Bootstrapped cluster gave very small deviations (against STATA 13). Since Bootstrap is a random method (and the deviation is very small), there is not (direct) way of knowing if this indicates a problem with the code or not.

- Instrumental Variables, both IV and 2SLS methods, give correct coefficient estimates (compared to STATA 13), but SE seem to be overestimated by about 10-20%. This could be some difference in calculating degrees of freedom, since the difference is not that big.

- As for Panel Models, Pooled OLS gives results consistent with the observations above, within effects give same coefficient as in STATA 13, but again SE seem to be ~10-20% higher in our module. First differences results are not consistent with STATA 13 and therefore shouldn't be used.
