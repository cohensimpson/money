#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/


import os
os.chdir("/Users/cohen/Desktop IconFree/GitHub/money")


# This analysis uses R, initialised in Python with "rpy2", to impute missing data.
# You must have both R and Python installed on your machine to use rpy2. 
# Once R is installed, install the "rpy2" module by running `pip install rpy2`
# or your preferred equivalent. 
# You will then need to install the R libraries "VIM" and "sbgcop" (see Line 31-35). 
# For help calling R in Python, see:
# https://rviews.rstudio.com/2022/05/25/calling-r-from-python-with-rpy2/
# https://rpy2.github.io/doc/v3.5.x/html/introduction.html
# https://rpy2.github.io/doc/v3.5.x/html/generated_rst/pandas.html
# https://rpy2.github.io/doc.html


import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import globalenv # R Environment to perform analysis.
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 20)
pd.set_option("display.min_rows", 20)




# Select CRAN mirror for R library downloads.
# 81 == Bristol, UKs
# utils.chooseCRANmirror(ind = 81)
# utils.install_packages("sbgcop")
# utils.install_packages("VIM")


# Import required R libraries
utils = importr("utils")
base = importr("base")
# graphics = importr('graphics')
vim = importr("VIM") # Used to visualise missing data
# https://cran.r-project.org/web/packages/sbgcop/index.html
sbgcop = importr("sbgcop") # Used for the Bayesian copula imputation




# Impute Attribute Data *USING R* 
# First, make a deep copy of the Pandas DataFrame
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
survey_responses_imputation = survey_responses.copy(deep = True)

# Retain only binary, ordinal, and continuous variables
del survey_responses_imputation["village_ID"]
del survey_responses_imputation["village_ID_txt"]
del survey_responses_imputation["HH_ID"]

# Convert Pandas DataFrame to R DataFrame
# https://rpy2.github.io/doc/latest/html/pandas.html
with localconverter(robjects.default_converter + pandas2ri.converter):
    survey_responses_imputation = robjects.conversion.py2rpy(survey_responses_imputation)


# Compare number of missing values in the Pandas DataFrame
# and the R DataFrame for sanity
print(vim.aggr(survey_responses_imputation))
print(utils.head(survey_responses_imputation))

survey_responses.info()
print(
    survey_responses["female"][survey_responses["female"].isna()], # 154 Missing
    survey_responses["income"][survey_responses["income"].isna()], # 8 Missing
    survey_responses["hasPhone"][survey_responses["hasPhone"].isna()], # 2 Missing
    survey_responses["edu_full"][survey_responses["edu_full"].isna()], # 4 Missing
    survey_responses["HH_Head"][survey_responses["HH_Head"].isna()], # 275 Missing
    sep = "\n\n"
)


# Perform Bayesian copula imputation
sbgcop_mcmc_results = sbgcop.sbgcop_mcmc(
    Y = survey_responses_imputation, 
    nsamp = 200, # 200000
    impute = True,
    seed = 20200127, 
    verb = True
)
print(base.summary(sbgcop_mcmc_results))
print(utils.head(sbgcop_mcmc_results[1]))


# Unfold the output from sbgcop.sbgcop_mcmc
# sbgcop.sbgcop_mcmc returns an R object containing (in order):
# (1) C.psamp = an array of size p x p x nsamp/odens, consisting of
# posterior samples of the correlation matrix.
# (2) Y.pmean = the original datamatrix with imputed values replacing missing data.
# (3) Y.impute = an array of size n x p x nsamp/odens, consisting of copies of the
# original data matrix, with posterior samples of missing values included.
# (4) LPC = the log-probability of the latent variables at each saved sample.
# For the analysis "Y.pmean" is needed.
survey_responses_imputation = sbgcop_mcmc_results[list(sbgcop_mcmc_results.names).index("Y.pmean")]


# Convert original dataset with missing replaced with the posterior mean
# into a R dataframe for conversion into a Pandas dataframe.
survey_responses_imputation = base.data_frame(survey_responses_imputation)
with localconverter(robjects.default_converter + pandas2ri.converter):
    survey_responses_imputation = robjects.conversion.rpy2py(survey_responses_imputation)


# Retain original version of the data with missing values before updating (in place).
survey_responses_within_missing = survey_responses.copy(deep = True)

missing_female = survey_responses_within_missing["female"].isna()  # 154 Missing
missing_income = survey_responses_within_missing["income"].isna() # 8 Missing
missing_hasPhone = survey_responses_within_missing["hasPhone"].isna() # 2 Missing
missing_edu_full = survey_responses_within_missing["edu_full"].isna() # 4 Missing
missing_HH_Head = survey_responses_within_missing["HH_Head"].isna() # 275 Missing


# Use posterior means to populate missing values in the original pandas data 
# frame with the features needed for modelling. Note, this is an in place update.
# Also, below, False == only update values that are NA in the original DataFrame.
survey_responses.update(
    other = survey_responses_imputation,
    join = "left",
    overwrite = False, 
    errors = "ignore"
)


# TODO: Don't Round! Graham, J. W. (2009). Missing Data Analysis: Making It
# Work in the Real World. Annual Review of Psychology, 60(1), 549–576.
# https://doi.org/10.1146/annurev.psych.58.110405.085530

# # Discretise the posterior means for the binary variables and reassign.
# # https://realpython.com/python-boolean/#python-booleans-as-numbers
# survey_responses.loc[missing_female, ["female"]] = (survey_responses.loc[missing_female, ["female"]] > 0.50).astype("float64")
# survey_responses.loc[missing_hasPhone, ["hasPhone"]] = (survey_responses.loc[missing_hasPhone, ["hasPhone"]] > 0.50).astype("float64")
# survey_responses.loc[missing_HH_Head, ["HH_Head"]] = (survey_responses.loc[missing_HH_Head, ["HH_Head"]] > 0.50).astype("float64")


# Cast the columns with imputed data as the appropriate type as reassign.
survey_responses = survey_responses.astype(
    dtype = {
        "female": "Int64",
        "income": "float64",
        "hasPhone": "Int64",
        "edu_full": "float64",
        "HH_Head": "Int64"
    }
)


