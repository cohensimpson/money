#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/




 
""" 
Replication Materials for "The Relational Bases of Informal Financial Cooperation".

Python Version Used for Analysis: CPython 3.9.10
PyMC Version Used for Analysis: 4.4.0

Data reanalysed for here come from:
    
    Ferrali, R., Grossman, G., Platas, M. R., & Rodden, J. (2020). It Takes a
    Village: Peer Effects and Externalities in Technology Adoption.
    American Journal of Political Science, 64(3), 536–553.
    https://doi.org/10.1111/ajps.12471
    
    Ferrali, R., Grossman, G., Platas, M. R., & Rodden, J. (2021). Who Registers?
    Village Networks, Household Dynamics, and Voter Registration in Rural Uganda.
    Comparative Political Studies, 001041402110360.
    https://doi.org/10.1177/00104140211036048


Author: Cohen R. Simpson
Maintainer: Cohen R. Simpson
Email: c.r.simpson@lse.ac.uk
"""

import os
os.chdir("/Users/cohen/Desktop IconFree/GitHub/money")


import arviz as az
import arviz.labels as azl
import aesara.tensor as at
import graphviz as gv
import itertools
import joblib
import matplotlib.pyplot as plt
import mizani as miz
import numpy as np
import numexpr
import pandas as pd 
import plotnine as p9
import pymc as pm # pip install pymc==4.4.0; Using Python v. 3.9.10
import scipy as sp
import scipy.stats
import xarray



# Set Random Seed for Analyses
np.random.seed(20200127) 


# Set Number of CPU Cores for PyMC Parallel Processing
# PyMC Uses No More than 4 CPU Cores
cpu_cores = 4


# TODO: Clarify if this is the best way to run all code components.
load_data = open("load_data.py")
build_features = open("build_features.py")
fit_models = open("fit_models.py")
visualise_results = open("visualise_results.py")

exec(load_data.read())
exec(build_features.read())
# exec(fit_models.read())
# exec(visualise_results.read())
