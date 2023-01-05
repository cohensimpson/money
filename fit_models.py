#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/






# The categories of lender analysed here.
types_of_lender = [
    "friend_lender_ij", 
    "family_lender_ij",
    "friend_family_lender_ij",
    "stranger_lender_ij",
    "friend_lender_ij_lender_ji", 
    "family_lender_ij_lender_ji",
    "friend_family_lender_ij_lender_ji",
    "lender_ij_lender_ji"
 ]




# Drop the 488 villagers who nominate zero lenders from the analysis.
# 2,696 villagers remain
print((all_villager_nominations["lender_ij"] > 0).value_counts(), "\n\n")
all_villager_nominations_zeros = all_villager_nominations
all_villager_nominations = all_villager_nominations[all_villager_nominations["lender_ij"] > 0]


# Drop the 137 villagers who have missing values for the variable "female"
#2,559 villagers remain
print(all_villager_nominations["female"].isna().value_counts(), "\n\n")
all_villager_nominations = all_villager_nominations[~all_villager_nominations["female"].isna()]


# The 2,559 Villagers nominate 6,052 Preferred Money Lenders
all_villager_nominations["lender_ij"].sum()




# Village (Group) and Villager (Nominator) IDs as integers/an index.
villages_IDx, villages_IDchar = pd.factorize(all_villager_nominations["village_ID"])
villagers_IDx, villagers_IDchar = pd.factorize(all_villager_nominations.index)

# Num. Villagers, Num. Categories
N, K = all_villager_nominations[types_of_lender].shape

# Num. of villages
G = len(villages_IDchar)

# Num. villagers in each village.
group_sizes = all_villager_nominations["village_ID"].value_counts().to_dict()




# Externally define coordinates (i.e., data dimensions) for the PyMC models.
# https://cluhmann.github.io/inferencedata/
# https://discourse.pymc.io/t/multi-dimensional-dims-do-not-seem-to-work/9859/2
coordinates = {
    "villages": villages_IDchar.to_list(), 
    "villagers": villagers_IDchar.to_list(),
    "categories": types_of_lender,
    "axis_zero": "0" # Placeholder used to map names to multidimensional model components.
}




def extended_dm_model(*, 
    total_count_colname = "lender_ij",
    observed_counts_colnames = types_of_lender,
    model_coords = coordinates,
    model_data = all_villager_nominations,
    total_categories = K,
    total_villagers = N,
    village_names = villages_IDchar,
    ndraws = 3000,
    ntune = 2000,
    nchains = cpu_cores,
    ncores = cpu_cores,
    seeds = [20200127] * cpu_cores
    ) -> pm.Model:
    
    """ Fit extended marginalised Dirichlet-Multinomial (DM) model to the lender data.
   
    The DM is a compound distribution used to model compositional count data (i.e.
    counts of mutually-exclusive categories constitutive of some larger sum). Below:
    
    - "total_count" (n_i) = total number of items observed across the K categories.
     Here, this is the total number of money lenders nominated by the ith villager.
     
    - "observed_counts" (y; N x K matrix) = number of items in each category k. That 
    is, the number of lenders of each type (see above) nominated by each villager.
    
    - "frac" (pi) = the expected fraction of counts falling into each category,
    or, more formally, a k-dimensional simplex (i.e. a set of numbers that 
    sum-to-one) indicating the proportion of counts in each category.
    
    [TODO: Unsure if this makes sense. Revise]
    - "conc" (phi) = concentration factor capturing overdispersion in the counts. 
    Larger values result in a count distribution that more closely
    approximate the multinomial. 
    
    - "alpha" = the "shape" parameters controlling the contours of the 
    dirichlet distribution (i.e., a multivariate distribution for values 
    that fall in the range [0, 1]. The parameterisation of the DM in PyMC uses
    the following definition:  alpha = conc (phi) × frac (pi), and vectors of 
    compositional counts for each observation/sample/villager are simulated by:
    
        1. Drawing values for a k-simplex as p_i ∼ Dirichlet(alpha = conc × frac)
        2. Simulating Y_counts_i ∼ Multinomial(total_count, p_i)
        3. N.b., each observation gets its own latent parameter p_i,
        drawn independently from a common Dirichlet distribution. However,
        the DM distribution is parameterised in such a way where the p_i's are
        not estimated, only frac/pi. 
    
    For addition detail on the DM model and its PyMC implementation, please see:
    
    Douma, J. C., & Weedon, J. T. (2019). Analysing Continuous Proportions in
        Ecology and Evolution: A practical Introduction to Beta and Dirichlet
        Regression. Methods in Ecology and Evolution, 10(9), 1412–1430.
        https://doi.org/10.1111/2041-210X.13234
    
    Harrison, J. G., Calder, W. J., Shastry, V., & Buerkle, C. A. (2020).
        Dirichlet‐Multinomial Modelling Outperforms Alternatives for Analysis
        of Microbiome and other Ecological Count Data. Molecular Ecology
        Resources, 20(2), 481–497. https://doi.org/10.1111/1755-0998.13128
        
    Kemp, C., Perfors, A., & Tenenbaum, J. B. (2007). Learning Overhypotheses
        with Hierarchical Bayesian Models. Developmental Science, 10(3), 307–321.
        https://doi.org/10.1111/j.1467-7687.2007.00585.x
        
    Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic
        Programming in Python Using PyMC3. PeerJ Computer Science, 2, e55.
        https://doi.org/10.7717/peerj-cs.55
        
    Zhang, Y., Zhou, H., Zhou, J., & Sun, W. (2017). Regression Models for
        Multivariate Count Data. Journal of Computational and Graphical
        Statistics, 26(1), 1–13. https://doi.org/10.1080/10618600.2016.1154063
    
    https://gregorygundersen.com/blog/2020/12/24/dirichlet-multinomial/
    https://blog.byronjsmith.com/dirichlet-multinomial-example.html
    https://mc-stan.org/docs/2_26/stan-users-guide/reparameterizations.html#dirichlet-priors
    https://www.isaacslavitt.com/posts/dirichlet-multinomial-for-skittle-proportions/
    https://stats.stackexchange.com/a/44725
    https://stats.stackexchange.com/a/244946
    """
    
    # # "with" + indentation creates a context manager for defining model variables
    # and the invocation of "as" binds the Container for the model random variables
    # to the Python variable name given immediately after.
    with pm.Model(coords = model_coords) as extended_dm_model:
        
        # Create data objects used to define the PyMC model.
        # Village/Group, Villager/Observations/Samples IDs
        villages_idx = pm.Data(
            "villages_idx", villages_IDx,
            dims = "villagers", mutable = False
        )
        villagers_idx = pm.Data(
            "villagers_idx", villagers_IDx,
            dims = "villagers", mutable = False
        )
        
        
        # Observed Data
        Y_total_lenders = pm.Data(
            "Y_total_lenders", model_data[total_count_colname],
            dims = "villagers", mutable = False
        )
        Y_observed_counts = pm.Data(
            "Y_observed_counts", model_data[observed_counts_colnames],
            dims = ("villagers", "categories"), mutable = False
        )
        
        
        # Common Hyperprior for Group-Specific Fractions/Proportions
        # https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LogNormal.html
        hyper_alpha = pm.LogNormal(
            "hyper_alpha", mu = 1, sigma = 1, dims = ("axis_zero", "categories")
        )
        
        
        # Priors for Group-Specific Fractions/Proportions (1 Simplex Per Village) 
        frac_comb = at.concatenate(
            [
                pm.Dirichlet(
                    f"frac_vill_{group}",
                    a = hyper_alpha,
                    dims = ("axis_zero", "categories")
                ) for group in village_names
            ],
            axis = 0
        )
        
        
        # Common Hyperprior for Group-Specific Concentration Factors  
        # https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Gamma.html
        hyper_lambda = pm.Gamma(
            "hyper_lambda", alpha = 2, beta = 2, dims = "axis_zero"
        )
        
        
        # Priors for Group-Specific Concentration Factors (1 Factor Per Village) 
        conc_comb = at.stack([
            pm.Exponential(
                f"conc_vill_{group}",
                lam = hyper_lambda,
                dims = "axis_zero"
            ) for group in village_names
        ])
        
        
        # Deterministic construction of vector of shape parameters alpha for the
        # Dirichlet-Multinomial distribution. These shape params. are used to
        # simulate the villager-specific compositional count vectors (i.e., the rows
        # of Y_observed_counts). Note the indexing by the *numeric representation*
        # of the village names (i.e., villages_idx). This retrieves the elements 
        # of frac_comb and conc_comb specific to a villager's village.
        # https://docs.pymc.io/en/latest/api/generated/pymc.Deterministic.html
        alpha_dm = frac_comb[villages_idx] * conc_comb[villages_idx]
        # alpha_dm = pm.Deterministic(
        #     "alpha_dm", frac_comb[villages_idx] * conc_comb[villages_idx]
        # )
        
        
        # The final component of the model defines Y_counts — i.e., the sampling dist.
        # of the outcomes in the dataset. PyMC3 calls this an "observed stochastic
        # variable" that represents the likelihood of observations. It is indicated
        # by the "observed" argument, which passes the observed data to the variable.
        # Note, unlike for the (hyper) priors, the parameters for the observed 
        # stochastic variable Y_counts do not have fixed values. 
        
        # Here, "n" is an N-length series of row totals and "observed" is simply
        # the compositional response variable given as an 2-D array/dataframe 
        # with shape (N, K) given by the coordinates for the PyMC model.
        print("PyMC version {0}".format(pm.__version__))
                
        Y_counts = pm.DirichletMultinomial(
            "Y_counts",
            n = Y_total_lenders,
            a = alpha_dm,
            observed = Y_observed_counts,
            dims = ("villagers", "categories")
        )
        
        
        # To conduct MCMC sampling to generate posterior samples in PyMC3, create
        # a "step method" object that corresponds to a particular MCMC algorithm.
        extended_dm_trace = pm.sample(
                draws = ndraws, 
                tune = ntune,
                chains = nchains,
                cores = ncores, 
                step = pm.NUTS(),
                random_seed = seeds,
                discard_tuned_samples = True,
                return_inferencedata = True,
                progressbar = True,
                compute_convergence_checks = True,
                idata_kwargs = {"log_likelihood": True}
            )
        
        
        # Draw posterior samples to use for posterior predictive check.
        pm.sample_posterior_predictive(
            trace = extended_dm_trace,
            extend_inferencedata = True,
            random_seed = seeds[0],
            progressbar = True
        )
         
         
        # Perform approximate Leave-One-Out (LOO) Cross-Validation (CV)
        # https://discourse.mc-stan.org/t/question-about-large-pareto-k-value/28796
        extended_dm_model_loo = az.loo(
            data = extended_dm_trace,
            pointwise = True,
            scale = "log"
        )
        
        
    return (extended_dm_model, extended_dm_trace, extended_dm_model_loo)
    # return extended_dm_model
# pm.model_to_graphviz(extended_dm_model()).view()




def baseline_dm_model(*, 
    total_count_colname = "lender_ij",
    observed_counts_colnames = types_of_lender,
    model_coords = coordinates,
    model_data = all_villager_nominations,
    total_categories = K,
    total_villagers = N,
    village_names = villages_IDchar,
    ndraws = 3000,
    ntune = 2000,
    nchains = cpu_cores,
    ncores = cpu_cores,
    seeds = [20200127] * cpu_cores
    ) -> pm.Model:
    
    """ Fit baseline marginalised Dirichlet-Multinomial (DM) model to the lender data.
   
    This model departs from the extended model above by estimating concentration
    and category fraction parameters that are common (i.e., the same/shared) 
    across all sixteen villagers. For details, RUN: print(extended_dm_model.__doc__)
    """
    
    with pm.Model(coords = model_coords) as baseline_dm_model:
        
        # Create data objects used to define the PyMC model.
        # Village/Group, Villager/Observations/Samples IDs
        villages_idx = pm.Data(
            "villages_idx", villages_IDx,
            dims = "villagers", mutable = False
        )
        villagers_idx = pm.Data(
            "villagers_idx", villagers_IDx,
            dims = "villagers", mutable = False
        )
        
        
        # Observed Data
        Y_total_lenders = pm.Data(
            "Y_total_lenders", model_data[total_count_colname],
            dims = "villagers", mutable = False
        )
        Y_observed_counts = pm.Data(
            "Y_observed_counts", model_data[observed_counts_colnames],
            dims = ("villagers", "categories"), mutable = False
        )
        
        
        # Hyperprior for Common Fractions/Proportions
        hyper_alpha = pm.LogNormal(
            "hyper_alpha", mu = 1, sigma = 1, dims = ("axis_zero", "categories")
        )
        
        
        # Prior for Common Fractions/Proportions (1 Simplex for All Villages)
        frac = pm.Dirichlet(
            "frac", a = hyper_alpha,
            dims = ("axis_zero", "categories")
        ) 
        
        
        # Hyperprior for Common Concentration Factor
        hyper_lambda = pm.Gamma(
            "hyper_lambda", alpha = 2, beta = 2, dims = "axis_zero"
        )
        
        
        # Prior for Common Concentration Factor (1 Factor for All Villages)
        conc = pm.Exponential("conc", lam = hyper_lambda, dims = "axis_zero")
        
        
        # Deterministic construction of vector of shape parameters alpha for the
        # Dirichlet-Multinomial distribution. These shape params. are used to
        # simulate the villager-specific compositional count vectors (i.e., the rows
        # of Y_observed_counts). Note the absence of indexing by village, thus 
        # resulting in a common shape parameters across all villages.
        # https://docs.pymc.io/en/latest/api/generated/pymc.Deterministic.html
        alpha_dm = frac * conc
        
        
        # The final component of the model defines Y_counts — i.e., the sampling dist.
        # of the outcomes in the dataset. PyMC3 calls this an "observed stochastic
        # variable" that represents the likelihood of observations. It is indicated
        # by the "observed" argument, which passes the observed data to the variable.
        # Note, unlike for the (hyper) priors, the parameters for the observed 
        # stochastic variable Y_counts do not have fixed values. 
        
        # Here, "n" is an N-length series of row totals and "observed" is simply
        # the compositional response variable given as an 2-D array/dataframe 
        # with shape (N, K) given by the coordinates for the PyMC model.
        print("PyMC version {0}".format(pm.__version__))
                
        Y_counts = pm.DirichletMultinomial(
            "Y_counts",
            n = Y_total_lenders,
            a = alpha_dm,
            observed = Y_observed_counts,
            dims = ("villagers", "categories")
        )
        
        
        # To conduct MCMC sampling to generate posterior samples in PyMC3, create
        # a "step method" object that corresponds to a particular MCMC algorithm.
        baseline_dm_trace = pm.sample(
                draws = ndraws, 
                tune = ntune,
                chains = nchains,
                cores = ncores, 
                step = pm.NUTS(),
                random_seed = seeds,
                discard_tuned_samples = True,
                return_inferencedata = True,
                progressbar = True,
                compute_convergence_checks = True,
                idata_kwargs = {"log_likelihood": True}
            )
        
        
        # Draw posterior samples to use for posterior predictive check.
        pm.sample_posterior_predictive(
            trace = baseline_dm_trace,
            extend_inferencedata = True,
            random_seed = seeds[0],
            progressbar = True
        )
         
         
        # Perform approximate Leave-One-Out (LOO) Cross-Validation (CV)
        baseline_dm_model_loo = az.loo(
            data = baseline_dm_trace,
            pointwise = True,
            scale = "log"
        )
        
        
    return (baseline_dm_model, baseline_dm_trace, baseline_dm_model_loo)
    # return baseline_dm_model
# pm.model_to_graphviz(baseline_dm_model()).view()




def sex_dm_model(*,  
    total_count_colname = "lender_ij",
    observed_counts_colnames = types_of_lender,
    model_coords = coordinates,
    model_data = all_villager_nominations,
    total_categories = K,
    total_villagers = N,
    village_names = villages_IDchar,
    ndraws = 3000,
    ntune = 2000,
    nchains = cpu_cores,
    ncores = cpu_cores,
    seeds = [20200127] * cpu_cores
    ) -> pm.Model:
    
    """ Fit Sex-Specific Marginalised Dirichlet-Multinomial (DM) model.
   
    Harrison, J. G., Calder, W. J., Shastry, V., & Buerkle, C. A. (2020).
        Dirichlet‐Multinomial Modelling Outperforms Alternatives for Analysis
        of Microbiome and other Ecological Count Data. Molecular Ecology
        Resources, 20(2), 481–497. https://doi.org/10.1111/1755-0998.13128
        
    """
    
    # # "with" + indentation creates a context manager for defining model variables
    # and the invocation of "as" binds the Container for the model random variables
    # to the Python variable name given immediately after.
    with pm.Model(coords = model_coords) as sex_dm_model:
        
        # Create data objects used to define the PyMC model.
        # Village/Group, Villager/Observations/Samples IDs
        villages_idx = pm.Data(
            "villages_idx", villages_IDx,
            dims = "villagers", mutable = False
        )
        villagers_idx = pm.Data(
            "villagers_idx", villagers_IDx,
            dims = "villagers", mutable = False
        )
        
        
        # Observed Data
        Y_total_lenders = pm.Data(
            "Y_total_lenders", model_data[total_count_colname],
            dims = "villagers", mutable = False
        ) 
        Y_observed_counts = pm.Data(
            "Y_observed_counts", model_data[observed_counts_colnames],
            dims = ("villagers", "categories"), mutable = False
        )
         
        # Prepare variable for Sex for PyMC
        female = pm.Data(
            "female", model_data["female"],
            dims = "villagers", mutable = False
        )
        
        
        # Common Hyperprior for Sex-Specific Fractions/Proportions 
        hyper_alpha = pm.LogNormal(
            "hyper_alpha", mu = 1, sigma = 1, dims = ("axis_zero", "categories")
        )
        
        
        # Priors for Female Fractions/Proportions (1 Simplex Per Village) 
        frac_comb_female = at.concatenate(
            [
                pm.Dirichlet(
                    f"frac_vill_{group}_female",
                    a = hyper_alpha,
                    dims = ("axis_zero", "categories")
                ) for group in village_names
            ],
            axis = 0
        )
        
        
        # Priors for Male Fractions/Proportions (1 Simplex Per Village) 
        frac_comb_male = at.concatenate(
            [
                pm.Dirichlet(
                    f"frac_vill_{group}_male",
                    a = hyper_alpha,
                    dims = ("axis_zero", "categories")
                ) for group in village_names
            ],
            axis = 0
        )
        
        
        # Common Hyperprior for Sex-Specific Concentration Factors
        hyper_lambda = pm.Gamma(
            "hyper_lambda", alpha = 2, beta = 2, dims = "axis_zero"
        )
        
        
        # Priors for Female Concentration Factors (1 Factor Per Village)
        conc_comb_female = at.stack([
            pm.Exponential(
                f"conc_vill_{group}_female",
                lam = hyper_lambda,
                dims = "axis_zero"
            ) for group in village_names
        ])
        
        
        # Priors for Male Concentration Factors (1 Factor Per Village)
        conc_comb_male = at.stack([
            pm.Exponential(
                f"conc_vill_{group}_male",
                lam = hyper_lambda,
                dims = "axis_zero"
            ) for group in village_names
        ])
        
        
        # Deterministic construction of vector of shape parameters alpha for the
        # Dirichlet-Multinomial distribution. These shape params. are used to
        # simulate the villager-specific compositional count vectors (i.e., the rows
        # of Y_observed_counts). Note the indexing by the *numeric representation*
        # of the village names (i.e., villages_idx). This retrieves the elements 
        # of frac_comb and conc_comb specific to a villager's village.
        
        # https://discourse.pymc.io/t/pymc3-elementwise-if-condition/8670
        # https://docs.pymc.io/en/v3/api/math.html
        
        alpha_dm = pm.math.where(
            pm.math.eq(female[:, np.newaxis], 1), # Equality test.
            frac_comb_female[villages_idx] * conc_comb_female[villages_idx],
            frac_comb_male[villages_idx] * conc_comb_male[villages_idx]
        )
         
        # alpha_dm = pm.Deterministic(
        #     "alpha_dm",
        #     pm.math.where(
        #         pm.math.eq(female[:, np.newaxis], 1), # Equality test.
        #             frac_comb_female[villages_idx] * conc_comb_female[villages_idx],
        #             frac_comb_male[villages_idx] * conc_comb_male[villages_idx]
        #         )
        # )
        
        
        # The final component of the model defines Y_counts — i.e., the sampling dist.
        # of the outcomes in the dataset. PyMC3 calls this an "observed stochastic
        # variable" that represents the likelihood of observations. It is indicated
        # by the "observed" argument, which passes the observed data to the variable.
        # Note, unlike for the (hyper) priors, the parameters for the observed 
        # stochastic variable Y_counts do not have fixed values. 
        
        # Here, "n" is an N-length series of row totals and "observed" is simply
        # the compositional response variable given as an 2-D array/dataframe 
        # with shape (N, K) given by the coordinates for the PyMC model.
        Y_counts = pm.DirichletMultinomial(
            "Y_counts",
            n = Y_total_lenders,
            a = alpha_dm,
            observed = Y_observed_counts,
            dims = ("villagers", "categories")
        )
        
        
        # To conduct MCMC sampling to generate posterior samples in PyMC3, create
        # a "step method" object that corresponds to a particular MCMC algorithm.
        print("PyMC version {0}".format(pm.__version__))
        
        sex_dm_trace = pm.sample(
                draws = ndraws, 
                tune = ntune,
                chains = nchains,
                cores = ncores, 
                step = pm.NUTS(),
                random_seed = seeds,
                discard_tuned_samples = True,
                return_inferencedata = True,
                progressbar = True,
                compute_convergence_checks = True,
                idata_kwargs = {"log_likelihood": True}
            )
        
        
        # Draw posterior samples to use for posterior predictive check.
        pm.sample_posterior_predictive(
            trace = sex_dm_trace,
            extend_inferencedata = True,
            random_seed = seeds[0],
            progressbar = True
        )
        
        
        # Perform approximate Leave-One-Out (LOO) Cross-Validation (CV)
        sex_dm_model_loo = az.loo(
            data = sex_dm_trace,
            pointwise = True,
            scale = "log"
        )
        
        
    return (sex_dm_model, sex_dm_trace, sex_dm_model_loo)
    # return sex_dm_model
# pm.model_to_graphviz(sex_dm_model()).view()




# Fit the Models, Unpacking Results Into Distinct Objects
baseline_model, baseline_model_trace, baseline_model_loo = baseline_dm_model()
extended_model, extended_model_trace, extended_model_loo = extended_dm_model()
sex_model, sex_model_trace, sex_model_loo = sex_dm_model()




## Save the Fitted Models
joblib.dump([baseline_model_trace, baseline_model_loo], "baseline_dm_model.pkl", compress = 2)
joblib.dump([extended_model_trace, extended_model_loo], "extended_dm_model.pkl", compress = 2)
joblib.dump([sex_model_trace, sex_model_loo], "sex_dm_model.pkl", compress = 2)
# baseline_model_trace, baseline_model_loo = joblib.load("baseline_dm_model.pkl")
# extended_model_trace, extended_model_loo = joblib.load("extended_dm_model.pkl")
# sex_model_trace, sex_model_loo = joblib.load("sex_dm_model.pkl")




# Visualise Structure of Estimated Models
# https://graphviz.readthedocs.io/en/stable/manual.html
# pm.model_to_graphviz(baseline_model).view() # .unflatten(stagger = 8)
# pm.model_to_graphviz(extended_model).view()
# pm.model_to_graphviz(sex_model).view()




# Model Results + Basic Diagnostics
# pd.set_option("display.max_rows", 800)
# pd.set_option("display.min_rows", 800)
print(az.summary(baseline_model_trace), "\n\n")
print(az.summary(extended_model_trace), "\n\n")
print(az.summary(sex_model_trace), "\n\n")




# Summary of Approximate Out-of-Sample Predictive Performance
print(baseline_model_loo, "\n\n")
print(extended_model_loo, "\n\n")
print(sex_model_loo, "\n\n")




# Comparison of Approximate Out-of-Sample Predictive Performance Across Models 
# elpd: expected log pointwise predictive density (ELPD). Higher = "Better".
# SE: Standard error of the ELPD estimate. 
# elpd_diff: Difference in ELPD between models. Difference computed 
# relative to the top-ranked model which always has a elpd_diff of 0.
# dSE: Standard error of the difference in ELPD between each model
# and the top-ranked model. It’s always 0 for the top-ranked model.
all_model_comparison = az.compare(
    compare_dict = {
        "Common Parameters": baseline_model_loo,
        "Village-Specific Parameters": extended_model_loo,
        "Village- and Sex-Specific Parameters": sex_model_loo,
    },
    method = "stacking"
) 
print(all_model_comparison, "\n\n")




# Key Model Diagnostics for Parameters
# Baseline Model
assert all(
    np.array(
        az.rhat(baseline_model_trace).to_dataframe() < 1.01
    ).flatten()
), "Rhat diagnostic failed for one or more parameters in Baseline Model!"

assert all(
    np.array(
        az.ess(baseline_model_trace, method = "bulk").to_dataframe() > 1000
    ).flatten()
), "Insufficient Bulk Effective Sample Size for one or more parameters in Baseline Model!"

assert all(
    np.array(
        az.ess(baseline_model_trace, method = "tail", prob = 0.95).to_dataframe() > 1000
    ).flatten()
), "Insufficient Tail Effective Sample Size for one or more parameters in Baseline Model!"


# Extended Model
assert all(
    np.array(
        az.rhat(extended_model_trace).to_dataframe() < 1.01
    ).flatten()
), "Rhat diagnostic failed for one or more parameters in Extended Model!"

assert all(
    np.array(
        az.ess(extended_model_trace, method = "bulk").to_dataframe() > 1000
    ).flatten()
), "Insufficient Bulk Effective Sample Size for one or more parameters in Extended Model!"

assert all(
    np.array(
        az.ess(extended_model_trace, method = "tail", prob = 0.95).to_dataframe() > 1000
    ).flatten()
), "Insufficient Tail Effective Sample Size for one or more parameters in Extended Model!"


# Sex Model
assert all(
    np.array(
        az.rhat(sex_model_trace).to_dataframe() < 1.01
    ).flatten()
), "Rhat diagnostic failed for one or more parameters in Sex-Specific Model!"

assert all(
    np.array(
        az.ess(sex_model_trace, method = "bulk").to_dataframe() > 1000
    ).flatten()
), "Insufficient Bulk Effective Sample Size for one or more parameters in Sex-Specific Model!"

assert all(
    np.array(
        az.ess(sex_model_trace, method = "tail", prob = 0.95).to_dataframe() > 1000
    ).flatten()
), "Insufficient Tail Effective Sample Size for one or more parameters in Sex-Specific Model!"



