#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/






class PymcModelBuilder:
    # Create Class Variables/Attributes
    model_type = "Dirchlet-Multinomial"
    version = "1.0"
    
    
    # Constructor Method 
    def __init__(
        self, *,
        model_data, model_specification,
        observed_counts_colnames, total_count_colname,
        config
    ):
        """
        Initialise an instance of PymcModelBuilder. This class is used to build
        and fit PyMC models in an objected-oriented fashion. My approach leans
        into the typical calling of pm.model() whereby it is used alongside the
        "with" keyword to create a context manager for building out a model.
        
        It is assumed that PyMC has been imported as "pm".
        
        For other examples, see:
        
        https://5hv5hvnk.github.io/blogs/NewModelBuilder
        https://gist.github.com/twiecki/86b02349c60385eb6d77793d37bd96a9
        https://realpython.com/python3-object-oriented-programming/
        https://docs.pymc.io/en/v5.0.1/learn/core_notebooks/pymc_overview.html
        
        
        Parameters
        ----------
        model_data: pd.DataFrame
            Dataframe containing the cleaned data, including the columns for:
            1. The multivariate response variable Y (i.e., columns defining count vectors). 
            2. The total counts/trials n_i for each observation/sample i.
            3. Sample-/observation-specific features to define the model (here, "sex").
            
        model_specification: str
            One of the following: "Baseline", "Extended", or "Sex". The value
            provided for the argument determines which version of the 
            Dirichlet-Multinomial model is estimated.
            See the instance method "build_model" for details.
        
        observed_counts_colnames: list[str]
            List containing the names of the columns in model_data that
            correspond to the response Y.
        
        total_count_colname: str
            Name of column in model_data that contains total number of trials n_i.
        
        config: dict[str: dict]
            Dictionary containing dictionaries of arguments used to control:
            1. Sampling from the posterior:
            https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html
                1. draws: Number of posterior draws per Markov chain
                2. tune: Number of tuning iterations per chain.
                3. chains: Number of chains to sample.
                4. cores: Number of chains to run in parallel. Max = 4.
                5. random_seed: List of Seed(s) used for sampling. One per chain.
                6. discard_tuned_samples: Keep samples used for warm-up?
                7. return_inferencedata: Return trace as arviz.InferenceData object?
                8. progressbar: Display progress bar for feedback?
                9. compute_convergence_check: Calculate core mode diagnostics?
                10. idata_kwargs: "log_likelihood" == True = Calculate for all Observed Vars.
                
            2. Sampling from the posterior predictive distribution:
            https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html
                1. extend_inferencedata: add the posterior predictive samples to trace?
                2. random_seed: Seed for sampling.
                3. progressbar: Display progress bar for feedback?
            
            3. Approximate leave-one-out cross-validation:
            https://python.arviz.org/en/stable/api/generated/arviz.loo.html
                1. pointwise: Return pointwise predictive accuracy?
                2. scale: Output scale for loo.
        """
        
        # Create Instance Variables/Attributes
        # Data Used to Fit All Models
        self.data = model_data.copy(deep = True)
        
        
        # User's Desired Model Specification
        self.specification = model_specification.strip().lower()
        
        
        # Multivariate Response Matrix Y + Total Number of Trials/Counts
        self.response_vars_names = observed_counts_colnames
        self.trials_var_name = total_count_colname
        
        self.response_vars = self.data[self.response_vars_names]
        self.trials_var = self.data[self.trials_var_name]
        
        
        # PyMC Model Coordinates (i.e., Dimensions) + Observation Indices
        # Quite a few things to unpack. Create atom with parentheses to gather.
        (
            self.groups_IDx, self.groups_IDchar,
            self.obs_IDx, self.obs_IDchar,
            self.N, self.K, self.G,
            self.coordinates
        ) = self.prepare_coordinates()
        
        
        # Configuration Settings For Sampling/Cross-Validation
        self.config_sample = config["sample"]
        self.config_sample_pp = config["sample_pp"]
        self.config_loo = config["loo"]
        
        self.model = None
        self.trace = None
        self.loo = None
        
        
    # TODO: How best to put this together given so many instances attributes?
    # def __repr__(self):
    #     return f"PymcModelBuilder(model_data={self.data})"
    
    
    # Create Instance Methods
    # https://realpython.com/instance-class-and-static-methods-demystified/
    def prepare_coordinates(self) -> tuple:
        """
        Using "data", prepare the coordinates which define the dimensions of the PyMC Model.
        https://cluhmann.github.io/inferencedata/
        https://discourse.pymc.io/t/multi-dimensional-dims-do-not-seem-to-work/9859/2
        """
        
        # Village (Group) and Villager (Samples/Observation) IDs as integers + an index.
        villages_IDx, villages_IDchar = pd.factorize(self.data["village_ID"])
        villagers_IDx, villagers_IDchar = pd.factorize(self.data.index)
        
        # Num. Villagers/Observations, Num. of Response Categories
        N, K = self.response_vars.shape
        
        # Num. of villages/groups
        G = len(villages_IDchar)
        
        coordinates = {
            "villages": villages_IDchar.to_list(), 
            "villagers": villagers_IDchar.to_list(),
            "categories": self.response_vars_names,
            "axis_zero": "0" # Placeholder used for dimensions of length = 1.
        }
        
        return (
            villages_IDx, villages_IDchar,
            villagers_IDx, villagers_IDchar,
            N, K, G,
            coordinates
        )
        
        
    def build_model_container(self) -> pm.Model:
        """
        Build the PyMC model container to hold variables + likelihood. 
        Note, in PyMC, pm.Model is used as a context manager within
        which one defines their model (e.g., setting priors, etc.). 
        Thus, one must first build a contained for context.
        https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html
        """
        
        self.model = pm.Model(coords = self.coordinates)
        
        
    def build_model(self) -> pm.Model:
        """
        Using the model container (see build_model_container), define the
        model with the desired specification (i.e., self.specification).
        Note, to keep things modular, each conditional call of "with self.model"
        defines all parts of the specification. This way, one could easily
        slot in/out new specifications if needed. Next, consider the following 
        summary of the models to better understand the specifications.
        
        The Dirichlet-Multinomial is a compound distribution used to model 
        compositional count data (i.e. counts of mutually-exclusive categories
        constitutive of some larger sum). Below:
        
        - "Y_total_lenders" (n_i): Total number of items observed across the
        K categories. Here, this is the total number of money lenders of each
        type k nominated by the ith villager.
         
        - "Y_observed_countsounts" (Y; N x K matrix): Number of items in each
        category k. That is, the number of lenders of each type k (columns)
        nominated by each villager (rows).
        
        - "frac" (pi): Expected fraction of counts falling into each category,
        or, more formally, a k-dimensional simplex (i.e. a set of numbers that 
        sum to one) indicating the proportion of counts in each category k.
        
        - "conc" (phi): Concentration factor capturing overdispersion of the 
        counts. Phi is used to scale frac. And larger values result in a
        distribution of counts that is more sparse. 
        
        - "alpha": The "shape" parameters controlling the contours of the 
        dirichlet distributions (i.e., a multivariate distribution for  
        values that fall in the range [0, 1] that sum to one.
        
        
        The parameterisation of the Dirichlet-Multinomial in PyMC uses the
        following definition:  alpha = conc (phi) × frac (pi). The compositional 
        count vector for each observation/sample/villager is simulated by:
        
        For addition detail on compositional data, the DM model, and its
        PyMC implementation, please see:
        
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
            
        Smith, B. J. (2021, January 29). The Dirichlet-Multinomial in PyMC3:
            Modeling Overdispersion in Compositional Count Data. Deep Ecology:
            A Blog on the New Microbiology.
            https://blog.byronjsmith.com/dirichlet-multinomial-example.html
            
        Zhang, Y., Zhou, H., Zhou, J., & Sun, W. (2017). Regression Models for
            Multivariate Count Data. Journal of Computational and Graphical
            Statistics, 26(1), 1–13. https://doi.org/10.1080/10618600.2016.1154063
        
        https://gregorygundersen.com/blog/2020/12/24/dirichlet-multinomial/
        https://mc-stan.org/docs/2_26/stan-users-guide/reparameterizations.html#dirichlet-priors
        https://www.isaacslavitt.com/posts/dirichlet-multinomial-for-skittle-proportions/
        https://stats.stackexchange.com/a/44725
        https://stats.stackexchange.com/a/244946
        """
        
        if self.specification not in ["baseline", "extended", "sex"]:
            raise ValueError("A valid model specification name was not provided.")
            
            
        if self.specification == "baseline":            
            with self.model:
                # Create data objects used to define the PyMC model.
                # Note that the names passed to "dims" are the coordinates (above).
                
                # Village/Group, Villager/Observations/Samples IDs
                villages_idx = pm.Data(
                    "villages_idx", self.groups_IDx,
                    dims = "villagers", mutable = False
                )
                villagers_idx = pm.Data(
                    "villagers_idx", self.obs_IDx,
                    dims = "villagers", mutable = False
                )
                
                
                # Observed Data
                Y_total_lenders = pm.Data(
                    "Y_total_lenders", self.trials_var,
                    dims = "villagers", mutable = False
                )
                Y_observed_counts = pm.Data(
                    "Y_observed_counts", self.response_vars,
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
                
                
                # Deterministic construction of vector of shape parameters alpha.
                # https://docs.pymc.io/en/latest/api/generated/pymc.Deterministic.html
                alpha_dm = frac * conc
                # alpha_dm = pm.Deterministic("alpha_dm", frac * conc)
                
                
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
                
                
        elif self.specification == "extended":            
            with self.model:
                # Create data objects used to define the PyMC model.
                # Note that the names passed to "dims" are the coordinates (above).
                
                # Village/Group, Villager/Observations/Samples IDs
                villages_idx = pm.Data(
                    "villages_idx", self.groups_IDx,
                    dims = "villagers", mutable = False
                )
                villagers_idx = pm.Data(
                    "villagers_idx", self.obs_IDx,
                    dims = "villagers", mutable = False
                )
                
                
                # Observed Data
                Y_total_lenders = pm.Data(
                    "Y_total_lenders", self.trials_var,
                    dims = "villagers", mutable = False
                )
                Y_observed_counts = pm.Data(
                    "Y_observed_counts", self.response_vars,
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
                        ) for group in self.groups_IDchar
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
                    ) for group in self.groups_IDchar
                ])
                
                
                # Deterministic construction of vector of shape parameters alpha.
                # Note the indexing by the *numeric representation* of the village
                # names (i.e., villages_idx). This retrieves the elements 
                # of frac_comb and conc_comb specific to a villager's village.
                alpha_dm = frac_comb[villages_idx] * conc_comb[villages_idx]
                # alpha_dm = pm.Deterministic(
                #     "alpha_dm", frac_comb[villages_idx] * conc_comb[villages_idx]
                # )
                
                
                # PyMC3 "observed stochastic variable" (i.e., the likelihood)
                Y_counts = pm.DirichletMultinomial(
                    "Y_counts",
                    n = Y_total_lenders,
                    a = alpha_dm,
                    observed = Y_observed_counts,
                    dims = ("villagers", "categories")
                )
                
                
        elif self.specification == "sex":            
            with self.model:
                # Create data objects used to define the PyMC model.
                # Note that the names passed to "dims" are the coordinates (above).
                
                # Village/Group, Villager/Observations/Samples IDs
                villages_idx = pm.Data(
                    "villages_idx", self.groups_IDx,
                    dims = "villagers", mutable = False
                )
                villagers_idx = pm.Data(
                    "villagers_idx", self.obs_IDx,
                    dims = "villagers", mutable = False
                )
                
                
                # Observed Data
                Y_total_lenders = pm.Data(
                    "Y_total_lenders", self.trials_var,
                    dims = "villagers", mutable = False
                )
                Y_observed_counts = pm.Data(
                    "Y_observed_counts", self.response_vars,
                    dims = ("villagers", "categories"), mutable = False
                )
                
                
                # Prepare variable for "sex" for PyMC
                female = pm.Data(
                    "female", self.data["female"],
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
                        ) for group in self.groups_IDchar
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
                        ) for group in self.groups_IDchar
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
                    ) for group in self.groups_IDchar
                ])
                
                
                # Priors for Male Concentration Factors (1 Factor Per Village)
                conc_comb_male = at.stack([
                    pm.Exponential(
                        f"conc_vill_{group}_male",
                        lam = hyper_lambda,
                        dims = "axis_zero"
                    ) for group in self.groups_IDchar
                ])
                
                
                # Deterministic construction of vector of shape parameters alpha.
                # Note the indexing by the *numeric representation* of the village
                # names (i.e., villages_idx). This retrieves the elements 
                # of frac_comb and conc_comb specific to a villager's village
                # Depending on each villager's sex (female == 1, male == 0).
                
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
                
                
                # PyMC3 "observed stochastic variable" (i.e., the likelihood)
                Y_counts = pm.DirichletMultinomial(
                    "Y_counts",
                    n = Y_total_lenders,
                    a = alpha_dm,
                    observed = Y_observed_counts,
                    dims = ("villagers", "categories")
                )
                
                
    def sample(self) -> az.InferenceData:
        """
        Estimate the model and sample from the posterior predictive distribution.
        Note that dictionary unpacking is used so that the user can easily
        pass values to the arguments used to configure the sampling procedure.
                
        https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html
        https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.sample_posterior_predictive.html
        https://realpython.com/python-kwargs-and-args/
        """
        
        with self.model:
            print("PyMC version {0}".format(pm.__version__))
            self.trace = pm.sample(step = pm.NUTS(), **self.config_sample)
            
            # This will simply append the post. predictive samples. to self.trace
            pm.sample_posterior_predictive(trace =  self.trace, ** self.config_sample_pp)
        
        
    def loo_cv(self) -> az.ELPDData:
        """
        Perform approximate leave-one-out cross-validation.
        Note that dictionary unpacking is used so that the user can easily
        pass values to the arguments used to configure the sampling procedure.
        
        https://python.arviz.org/en/stable/api/generated/arviz.loo.html
        
        Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian Model
        Evaluation Using Leave-One-Out Cross-Validation and WAIC. Statistics and
        Computing, 27(5), 1413–1432. https://doi.org/10.1007/s11222-016-9696-4
        """
        
        self.loo = az.loo(data = self.trace, **self.config_loo)
            




# The categories/types of money lenders (see build_features.py).
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
 
 
 
 
# Create dictionary of dictionaries containing the values for arguments controlling:
# (1) sampling from the posterior; (2) sampling from the posterior predictive
# distribution; and (3) approximate leave-one-out cross-validation
configuration_dictionaries = dict(
    sample = dict(
        draws = draws_per_chain, 
        tune = tuning_iterations_per_chain,
        chains = markov_chains,
        cores = cpu_cores, 
        random_seed = [20200127] * markov_chains,
        discard_tuned_samples = True,
        return_inferencedata = True,
        progressbar = True,
        compute_convergence_checks = True,
        idata_kwargs = {"log_likelihood": True}
    ),
    sample_pp = dict(
        extend_inferencedata = True,
        random_seed = 20200127,
        progressbar = True
    ),
    loo = dict(
        pointwise = True,
        scale = "log"
    )
)




# Fit models by creating objects of class "PymcModelBuilder"
baseline_model = PymcModelBuilder(
    model_data = all_villager_nominations,
    model_specification = "Baseline",
    observed_counts_colnames = types_of_lender,
    total_count_colname = "lender_ij",
    config = configuration_dictionaries
)
baseline_model.build_model_container() # Must be run in this order!
baseline_model.build_model()
baseline_model.sample()
baseline_model.loo_cv()

extended_model = PymcModelBuilder(
    model_data = all_villager_nominations,
    model_specification = "Extended",
    observed_counts_colnames = types_of_lender,
    total_count_colname = "lender_ij",
    config = configuration_dictionaries
)
extended_model.build_model_container()
extended_model.build_model()
extended_model.sample()
extended_model.loo_cv()


sex_model = PymcModelBuilder(
    model_data = all_villager_nominations,
    model_specification = "Sex",
    observed_counts_colnames = types_of_lender,
    total_count_colname = "lender_ij",
    config = configuration_dictionaries
)
sex_model.build_model_container()
sex_model.build_model()
sex_model.sample()
sex_model.loo_cv()




# Save the Fitted Models
# https://github.com/cloudpipe/cloudpickle
# https://github.com/pymc-devs/pymc/issues/5886#issuecomment-1163803524
fitted_models = (baseline_model, extended_model, sex_model)
fitted_models = cloudpickle.dumps(fitted_models)
file = open("fitted_dm_models.pkl", "wb") # "wb" = write in binary mode
file.write(fitted_models)
file.close()
# Code to load the fitted models post save.
# file = open("fitted_dm_models.pkl", "rb") # "rb" = read in binary mode
# baseline_model, extended_model, sex_model = cloudpickle.loads(file.read())




# Visualise Structure of Estimated Models
# https://graphviz.readthedocs.io/en/stable/manual.html
# pm.model_to_graphviz(baseline_model.model).view() # .unflatten(stagger = 8)
# pm.model_to_graphviz(extended_model.model).view()
# pm.model_to_graphviz(sex_model.model).view()




# Model Results + Basic Diagnostics
# pd.set_option("display.max_rows", 800)
# pd.set_option("display.min_rows", 800)
print(az.summary(baseline_model.trace), "\n\n")
print(az.summary(extended_model.trace), "\n\n")
print(az.summary(sex_model.trace), "\n\n")




# Summary of Approximate Out-of-Sample Predictive Performance
print(baseline_model.loo, "\n\n")
print(extended_model.loo, "\n\n")
print(sex_model.loo, "\n\n")




# Comparison of Approximate Out-of-Sample Predictive Performance Across Models 
# elpd: expected log pointwise predictive density (ELPD). Higher = "Better".
# SE: Standard error of the ELPD estimate. 
# elpd_diff: Difference in ELPD between models. Difference computed 
# relative to the top-ranked model which always has a elpd_diff of 0.
# dSE: Standard error of the difference in ELPD between each model
# and the top-ranked model. It’s always 0 for the top-ranked model.
all_model_comparison = az.compare(
    compare_dict = {
        "Common Parameters": baseline_model.loo,
        "Village-Specific Parameters": extended_model.loo,
        "Village- and Sex-Specific Parameters": sex_model.loo,
    },
    method = "stacking"
) 
print(all_model_comparison, "\n\n")




# Key Model Diagnostics for Parameters
# Baseline Model
assert all(
    np.array(
        az.rhat(baseline_model.trace).to_dataframe() < 1.01
    ).flatten()
), "Rhat diagnostic failed for one or more parameters in Baseline Model!"

assert all(
    np.array(
        az.ess(baseline_model.trace, method = "bulk").to_dataframe() > 1000
    ).flatten()
), "Insufficient Bulk Effective Sample Size for one or more parameters in Baseline Model!"

assert all(
    np.array(
        az.ess(baseline_model.trace, method = "tail", prob = 0.95).to_dataframe() > 1000
    ).flatten()
), "Insufficient Tail Effective Sample Size for one or more parameters in Baseline Model!"


# Extended Model
assert all(
    np.array(
        az.rhat(extended_model.trace).to_dataframe() < 1.01
    ).flatten()
), "Rhat diagnostic failed for one or more parameters in Extended Model!"

assert all(
    np.array(
        az.ess(extended_model.trace, method = "bulk").to_dataframe() > 1000
    ).flatten()
), "Insufficient Bulk Effective Sample Size for one or more parameters in Extended Model!"

assert all(
    np.array(
        az.ess(extended_model.trace, method = "tail", prob = 0.95).to_dataframe() > 1000
    ).flatten()
), "Insufficient Tail Effective Sample Size for one or more parameters in Extended Model!"


# Sex Model
assert all(
    np.array(
        az.rhat(sex_model.trace).to_dataframe() < 1.01
    ).flatten()
), "Rhat diagnostic failed for one or more parameters in Sex-Specific Model!"

assert all(
    np.array(
        az.ess(sex_model.trace, method = "bulk").to_dataframe() > 1000
    ).flatten()
), "Insufficient Bulk Effective Sample Size for one or more parameters in Sex-Specific Model!"

assert all(
    np.array(
        az.ess(sex_model.trace, method = "tail", prob = 0.95).to_dataframe() > 1000
    ).flatten()
), "Insufficient Tail Effective Sample Size for one or more parameters in Sex-Specific Model!"
