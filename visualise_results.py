#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/






# Create list of parameter names to aid with transforming results from PyMC/arviz
fraction_parameter_names = (
    ["frac"]
    + [f"frac_vill_{group}" for group in villages_IDchar]
    + [f"frac_vill_{group}_{sex}" for group in villages_IDchar for sex in ["female", "male"]]
)
concentration_parameter_names = (
    ["conc"] 
    + [f"conc_vill_{group}" for group in villages_IDchar]
    + [f"conc_vill_{group}_{sex}" for group in villages_IDchar for sex in ["female", "male"]]
)
all_parameter_names = (
    ["hyper_alpha"]
    + ["hyper_lambda"]
    + ["frac"]
    + [f"frac_vill_{group}" for group in villages_IDchar]
    + ["conc"]
    + [f"conc_vill_{group}" for group in villages_IDchar]
)
parameter_categories = types_of_lender + ["common"]
model_names = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)", "Sex (Common)"]





# Define various helper functions to aid with transforming results from PyMC/arviz
def pymc_trace_xarray_to_pandas(
    inference_data, *,
    samples: bool = True,
    villages_names: list[str] = villages_IDchar, 
    villagers_names: list[str] = villagers_IDchar, 
    conc_parm_names: list[str] = concentration_parameter_names) -> pd.DataFrame: 
    """
    Extracts MCMC samples from a PyMC posterior (xarray dataset) and converts
    to a Pandas data frame + a bit of processing to prepare for plotting.
    
    Function assume that arviz has been imported as "az" and xarray as "xarray".
    Furthermore, an InferenceData object (returned from pymc.sample) or posterior
    predictions (returned from pymc.sample_posterior_predictive) are required.
    """
    
    
    # https://switowski.com/blog/checking-for-true-or-false/
    if samples:
        posterior_samples = getattr(
            inference_data, "posterior",
            "inference_data is not an arviz or xarray object with posterior samples!"
        )
        
        
        if isinstance(posterior_samples, az.data.inference_data.InferenceData):
            posterior_samples = posterior_samples["posterior"]
            posterior_samples = posterior_samples.to_dataframe()
            posterior_samples = posterior_samples.reset_index(drop = False)
            posterior_samples = posterior_samples.melt(
                id_vars = ["chain", "draw", "axis_zero", "categories"],
                var_name = "parameter",
                value_name = "sampled_estimate"
            )
            
            # Estimated fraction parameters are multidimensional (i.e., one
            # per modelled category) and village-specific. However, the concentration
            # parameters are only village-specific and common across the categories.
            # Accordingly, we need to filter out the duplicate entries for the
            # concentration parameters. To do this, the values in the  "parameter"
            # column — i.e., the name of the category that each concentration factor 
            # belongs to — with "common" as the estimates are the same across categories.
            # Below, "duplicated(keep = "first")" means that the first-encountered
            # row of the duplicate rows is marked "False", where the duplicates 
            # that follow are marked as "True". 
            posterior_samples.loc[posterior_samples["parameter"].isin(conc_parm_names), ["categories"]] = "common"        
            posterior_samples = posterior_samples[~posterior_samples.duplicated(keep = "first")]
            
            del posterior_samples["axis_zero"]
            
            return posterior_samples
            
            
        elif isinstance(posterior_samples, xarray.core.dataset.Dataset):
            posterior_samples = posterior_samples.to_dataframe()
            posterior_samples = posterior_samples.reset_index(drop = False)
            posterior_samples = posterior_samples.melt(
                id_vars = ["chain", "draw", "axis_zero", "categories"],
                var_name = "parameter",
                value_name = "sampled_estimate"
            )
            
            posterior_samples.loc[posterior_samples["parameter"].isin(conc_parm_names), ["categories"]] = "common"        
            posterior_samples = posterior_samples[~posterior_samples.duplicated(keep = "first")]
            
            del posterior_samples["axis_zero"]
            
            return posterior_samples
            
            
        else:
            print(posterior_samples)
        
        
    else:
        posterior_predictions = getattr(
            inference_data, "posterior_predictive",
            "inference_data is not an arviz or xarray object with posterior predictions!"
        )
        
        
        if isinstance(posterior_predictions, az.data.inference_data.InferenceData):
            # No need to melt as "posterior_predictive" only concerns the response Y
            posterior_predictions = posterior_predictions["posterior_predictive"]
            posterior_predictions = posterior_predictions.to_dataframe()
            posterior_predictions = posterior_predictions.reset_index(drop = False)
            
            
            # Convert "categories" to pd.Categorical for faster pivoting below.
            # https://stackoverflow.com/a/55405384
            posterior_predictions["categories"] = pd.Categorical(
                posterior_predictions["categories"], categories = [
                 "friend_lender_ij", 
                 "family_lender_ij",
                 "friend_family_lender_ij",
                 "stranger_lender_ij",
                 
                 "friend_lender_ij_lender_ji",
                 "family_lender_ij_lender_ji",
                 "friend_family_lender_ij_lender_ji",
                 "lender_ij_lender_ji"
              ], ordered = False
            )
            
            posterior_predictions["villagers"] = pd.Categorical(
                posterior_predictions["villagers"],
                categories = villagers_names,
                ordered = False
            )
            
            # Widen the dataframe by creating columns for
            # the posterior predictions for each category
            posterior_predictions = posterior_predictions.pivot(
                index = ["chain", "draw", "villagers"],
                columns = ["categories"],
                values = ["Y_counts"]
            )
            
            posterior_predictions = posterior_predictions.reset_index(drop = False)
            
            # Pivot + reset_index results in a multi-index that should be flattened.
            posterior_predictions.columns = [
                " ".join(column).strip().replace(" ", "_")
                for column in posterior_predictions.columns.to_flat_index()
            ]
            
            return posterior_predictions
            
            
        elif isinstance(posterior_predictions, xarray.core.dataset.Dataset):
            posterior_predictions = posterior_predictions.to_dataframe()
            posterior_predictions = posterior_predictions.reset_index(drop = False)
            
            posterior_predictions["categories"] = pd.Categorical(
                posterior_predictions["categories"], categories = [
                 "friend_lender_ij", 
                 "family_lender_ij",
                 "friend_family_lender_ij",
                 "stranger_lender_ij",
                 
                 "friend_lender_ij_lender_ji",
                 "family_lender_ij_lender_ji",
                 "friend_family_lender_ij_lender_ji",
                 "lender_ij_lender_ji"
              ], ordered = False
            )
            
            posterior_predictions["villagers"] = pd.Categorical(
                posterior_predictions["villagers"],
                categories = villagers_names,
                ordered = False
            )
            
            posterior_predictions = posterior_predictions.pivot(
                index = ["chain", "draw", "villagers"],
                columns = ["categories"],
                values = ["Y_counts"]
            )
            
            posterior_predictions = posterior_predictions.reset_index(drop = False)
                        
            posterior_predictions.columns = [
                " ".join(column).strip().replace(" ", "_")
                for column in posterior_predictions.columns.to_flat_index()
            ]
            
            return posterior_predictions
            
            
        else:
            print(posterior_predictions)





def pymc_hdi_xarray_to_pandas(
    inference_data, *,
    preferred_prob: float = 0.95,
    villages_names: list[str] = village_IDs,
    response_categories: list[str] = types_of_lender,
    conc_parm_names: list[str] = concentration_parameter_names) -> pd.DataFrame:
    """
    Takes an arviz.data.inference_data.InferenceData object returned from 
    pymc.sample, calculates highest density credible intervals (HDIs) using
    one's preferred probability and then converts to a Pandas data frame with
    a bit of processing to prepare for plotting.
    
    Function assumes that arviz has been imported as "az". For details on az.hdi,
    see: https://python.arviz.org/en/stable/api/generated/arviz.hdi.html
    """
    
    
    if isinstance(inference_data, az.data.inference_data.InferenceData):
        hd_intervals = az.hdi(
            inference_data,
            hdi_prob = preferred_prob,
            group = "posterior"
        )
        hd_intervals = hd_intervals.to_dataframe()
        hd_intervals = hd_intervals.reset_index(drop = False)
        hd_intervals = hd_intervals.melt(
            id_vars = ["axis_zero", "categories", "hdi"],
            var_name = "parameter",
            value_name = "hdi_boundary"
        )
        
        # Widen dataframe by creating columns for credible
        # intervals for each parameter in each category.
        hd_intervals = hd_intervals.pivot(
            index = ["axis_zero", "categories", "parameter"],
            columns = ["hdi"],
            values = ["hdi_boundary"]
        )
        hd_intervals = hd_intervals.reset_index(drop = False)
        
        
        # Pivot + reset_index results in a multi-index that should be flattened.
        hd_intervals.columns = [
            " ".join(column).strip().replace(" ", "_")
            for column in hd_intervals.columns.to_flat_index()
        ]
        
        # Estimated fraction parameters are multidimensional (i.e., one
        # per modelled category) and village-specific. However, the concentration
        # parameters are only village-specific and common across the categories.
        # Accordingly, we need to filter out the duplicate entries for the
        # concentration parameters. To do this, the values in the  "parameter"
        # column — i.e., the name of the category that each concentration factor 
        # belongs to — with "common" as the estimates are the same across categories.
        hd_intervals.loc[hd_intervals["parameter"].isin(conc_parm_names), ["categories"]] = "common"        
        hd_intervals = hd_intervals[~hd_intervals.duplicated(keep = "first")]
        
        # https://stackoverflow.com/a/52825733
        hd_intervals.loc[:, ["categories"]] = pd.Categorical(
            hd_intervals["categories"],
            categories = (response_categories + ["common"]),
            ordered = True
        )
        
        hd_intervals = hd_intervals.sort_values("categories")
        hd_intervals = hd_intervals.reset_index(drop = True)
        
        del hd_intervals["axis_zero"]
        
        return hd_intervals
        
        
    else:
        print("inference_data is not an arviz object with posterior samples!")





def make_posterior_means(
    inference_data, *,
    preferred_prob: float = 0.95,
    demographic_params: bool = False,
    villages_names: list[str] = village_IDs,
    conc_parm_names: list[str] = concentration_parameter_names) -> pd.DataFrame:
    
    """
    Takes an arviz.data.inference_data.InferenceData object returned from 
    pymc.sample and calculates the posterior mean for each parameter using
    the posterior samples.
    """
    
    if isinstance(inference_data, az.data.inference_data.InferenceData):
        # Transform posterior samples and calculate posterior mean across chains. 
        posterior_means = pymc_trace_xarray_to_pandas(
            inference_data,
            samples = True,
            villages_names = villages_names,
            conc_parm_names = conc_parm_names
        )
        
        posterior_means = posterior_means.groupby(
            by = ["categories", "parameter"]
        ).mean(numeric_only = True)
        
        posterior_means = posterior_means.reset_index(drop = False)
        posterior_means.drop(columns = ["chain", "draw"], inplace = True)
        
        
        # Join posterior means to dataframe containing highest density intervals.
        posterior_means = pd.merge(
            how = "left",
            left = pymc_hdi_xarray_to_pandas(
                inference_data,
                preferred_prob = preferred_prob,
                villages_names = villages_names
            ), 
            right = posterior_means, 
            left_on = ["categories", "parameter"], 
            right_on = ["categories", "parameter"],
            suffixes = ("_hdi", "_means"),
            indicator = False
        )
        
        posterior_means = posterior_means.rename(
            columns = {
                "hdi_boundary_higher": "higher_bound",
                "hdi_boundary_lower": "lower_bound",
                "sampled_estimate": "pmean"
            }
        )
        
        
        if demographic_params:
            # Create a flag for whether or not a parameter is
            # specific to males or females.
            posterior_means["sex"] = np.nan
            posterior_means.loc[posterior_means["parameter"].str.contains("male"), ["sex"]] = "Male"
            posterior_means.loc[posterior_means["parameter"].str.contains("female"), ["sex"]] = "Female"
            
            posterior_means.loc[:, ["parameter"]] = (
                posterior_means["parameter"].str.replace("_female", "").str.replace("_male", "")
            )
            
            
            return posterior_means
            
            
        else:
            return posterior_means
        
        
    else:
        print("inference_data is not an arviz object with posterior samples!")





def prepare_pymc_trace_for_ppc(
    inference_data, *,
    model_data = all_villager_nominations,
    villages_names: list[str] = villages_IDchar,
    response_categories: list[str] = types_of_lender,
    conc_parm_names: list[str] = concentration_parameter_names) -> pd.DataFrame: 
    
    
    if isinstance(inference_data, az.data.inference_data.InferenceData):
        
        model_ppc = pymc_trace_xarray_to_pandas(
            inference_data,
            samples = False,
            villages_names = villages_names,
            conc_parm_names = conc_parm_names
        )
        
        # Unique id for each draw from the posterior predictive distribution.
        # This is slow, and it may perhaps be speed up using pd.eval()?
        # https://jakevdp.github.io/PythonDataScienceHandbook/03.12-performance-eval-and-query.html
        model_ppc.loc[:, ["chain_draw"]] = (
            model_ppc["chain"].astype("string") + "_" + model_ppc["draw"].astype("string")
        )
        
        
        # Recall that the response variable for this analysis is an N x K matrix
        # wherein each row is a compositional vector of counts indicating the
        # number of each of the K types of lenders that each of the N villagers
        # name as a source of money. Keeping this in mind, I compare the
        # frequency that various counts appear in the response variable.
        # Specifically, for each draw from the posterior (i.e., an N x K Matrix),
        # calculate the number of times that counts for a given type of lender
        # appear. Note, villagers could only name up to five lenders.
        # Thus, counts for each type of lender will never be greater than five.
        # Finally, as the response variable is a matrix/multivariate, tallies of
        # the counts must be performed for each of the K types of lender.
        model_ppc = pd.concat(
            objs = [
                # To try to clean up this ugly method chaining, parentheses "()" 
                # are used to create an "atom" wherein various methods are used  
                # to transform the result from pd.crosstab before concatenation.
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_friend_lender_ij"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T # Transpose for concatenation along the row axis
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_friend_lender_ij")
                    .rename(columns = {
                        "Y_counts_friend_lender_ij": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_family_lender_ij"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_family_lender_ij")
                    .rename(columns = {
                        "Y_counts_family_lender_ij": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_friend_family_lender_ij"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_friend_family_lender_ij")
                    .rename(columns = {
                        "Y_counts_friend_family_lender_ij": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_stranger_lender_ij"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_stranger_lender_ij")
                    .rename(columns = {
                        "Y_counts_stranger_lender_ij": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_friend_lender_ij_lender_ji"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_friend_lender_ij_lender_ji")
                    .rename(columns = {
                        "Y_counts_friend_lender_ij_lender_ji": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_family_lender_ij_lender_ji"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_family_lender_ij_lender_ji")
                    .rename(columns = {
                        "Y_counts_family_lender_ij_lender_ji": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_friend_family_lender_ij_lender_ji"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_friend_family_lender_ij_lender_ji")
                    .rename(columns = {
                        "Y_counts_friend_family_lender_ij_lender_ji": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
                
                
                (pd.crosstab(
                    index = model_ppc["chain_draw"],
                    columns = model_ppc["Y_counts_lender_ij_lender_ji"]
                    )
                    .describe().loc[["mean","std"], :]
                    .T
                    .reset_index(drop = False)
                    .assign(categories = "Y_counts_lender_ij_lender_ji")
                    .rename(columns = {
                        "Y_counts_lender_ij_lender_ji": "number_of_lenders",
                        "mean": "Y_count_ppc_mean", "std": "Y_count_ppc_std"
                        }
                    )
                ),
            ],
            join = "outer",
            axis = 0,
            verify_integrity = False,
            sort = False
        )
        
        
        # Calculate the number of times that counts for a given 
        # type of lender appear in the *observed* data
        Y_observed_counts = model_data[types_of_lender]
        Y_observed_counts.columns = [
            f"Y_counts_{column}" for column in Y_observed_counts.columns
        ]
        Y_observed_counts = Y_observed_counts.apply(
            axis = 0, func = lambda col: col.value_counts()).fillna(0)
        
        Y_observed_counts = Y_observed_counts.unstack()
        Y_observed_counts = Y_observed_counts.reset_index(drop = False)
        Y_observed_counts = Y_observed_counts.rename(
            columns = {
                "level_0": "categories",
                "level_1": "number_of_lenders",
                0: "Y_observed_count"
            }
        )
        
        
        # Join mean number of times that counts for a given type of leader 
        # across posterior predictions to data frame containing the number of
        # times that counts of each type of lender appear in the observed data.
        model_ppc = pd.merge(
            how = "left",
            left = Y_observed_counts, 
            right = model_ppc, 
            left_on = ["number_of_lenders", "categories"], 
            right_on = ["number_of_lenders", "categories"],
            suffixes = ("_observed", "_ppc"),
            indicator = False
        )
        
        
        return model_ppc
        
        
    else:
        print("inference_data is not an arviz object with posterior samples!")





# Figure 1 — Posterior Mean Proportions
# First, derive the Pandas dataframe to pass to ggplot for plotting.
baseline_model_pmean = make_posterior_means(
    inference_data = baseline_model_trace,
    demographic_params = False
)
baseline_model_pmean["model"] = "Baseline"
baseline_model_pmean["sex"] = np.nan

extended_model_pmean = make_posterior_means(
    inference_data = extended_model_trace,
    demographic_params = False
)
extended_model_pmean["model"] = "Extended"
extended_model_pmean["sex"] = np.nan

sex_model_pmean = make_posterior_means(
    inference_data = sex_model_trace,
    demographic_params = True
)
sex_model_pmean["model"] = "Sex (Common)" # For params. that are not sex-specific.
sex_model_pmean.loc[sex_model_pmean["sex"] == "Female", ["model"]] = "Sex (Female)"
sex_model_pmean.loc[sex_model_pmean["sex"] == "Male", ["model"]] = "Sex (Male)"


# Second, join the model-specific dataframes of posterior means.
model_pmeans = pd.concat(
    objs = [baseline_model_pmean, extended_model_pmean, sex_model_pmean],
    join = "outer",
    axis = 0,
    verify_integrity = False,
    sort = False
)


# Third, create ordered string variables for pretty plotting of results.
# https://plotnine.readthedocs.io/en/stable/tutorials/miscellaneous-order-plot-series.html
model_pmeans["categories"] = pd.Categorical(
    model_pmeans["categories"],
    categories = parameter_categories,
    ordered = True
)
model_pmeans["model"] = pd.Categorical(
    model_pmeans["model"],
    categories = model_names, 
    ordered = True
)
model_pmeans["parameter"] = pd.Categorical(
    model_pmeans["parameter"],
    categories = all_parameter_names,
    ordered = True
)


# Fourth, create unique string IDs for each parameter to make parallel coordinate
# plot wherein posterior mean proportions are tied across the lender categories.
model_pmeans = model_pmeans.reset_index(drop = True)
model_pmeans["estimate_ID"] = (
    model_pmeans["parameter"].astype("string") + "_" + model_pmeans["model"].astype("string")
)

 
# Fourth, create Figure 1
figure_1_para_coord = (
    p9.ggplot(
        data = model_pmeans[model_pmeans["parameter"].isin(fraction_parameter_names)]
    ) 
    + p9.aes(
        x = "categories", y = "pmean", group = "estimate_ID",
        ymin = "lower_bound", ymax = "higher_bound",
        colour = "model", alpha = "model", size = "model"
    )
    + p9.geom_linerange(
        size = 0.25,
        linetype = "solid",
        colour = "#1E1E1E",
        show_legend = False
    )
    + p9.geom_line(
        linetype = "solid",
        show_legend = False
    )
    + p9.geom_point(
        size = 1.75,
        show_legend = True
    )
    + p9.labs(
        x = "\nVillage\n",
        y = "Posterior Mean Proportion + 95% Highest Density Interval (Square-Root Scale)\n"
    )
    + p9.scale_x_discrete(
        breaks = [
            "friend_lender_ij", 
            "family_lender_ij",
            "friend_family_lender_ij",
            "stranger_lender_ij",
            "friend_lender_ij_lender_ji", 
            "family_lender_ij_lender_ji",
            "friend_family_lender_ij_lender_ji",
            "lender_ij_lender_ji"
         ],
        labels = [
            "Best Friend\n(Non-indebted)",
            "Salient Kin\n(Non-indebted)",
            "B. Friend + S. Kin\n(Non-indebted)",
            "Not B. Friend or S. Kin\n(Non-indebted)",
             
            "Best Friend\n(Indebted)",
            "Salient Kin\n(Indebted)",
            "B. Friend + S. Kin\n(Indebted)",
            "Not B. Friend or S. Kin\n(Indebted)"
         ]
    )
    # + p9.facet_wrap(
    #     "model",
    #      ncol = 1,
    #      dir = "v"
    #  )
    + p9.scale_y_sqrt(
        limits = [0, 0.6],
        breaks = [0, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )    
    # https://jtools.jacob-long.com/reference/jtools_colors.html
    # https://personal.sron.nl/~pault/
    + p9.scale_colour_manual(
        name = "Model",
        drop = True,
        values = ["#CC3311", "#EE7733", "#0077BB", "#EECC66"],
        limits = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        breaks = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        labels = [
            "Baseline Model", "Extended Model", 
            "Sex-Specic Model (Female)", "Sex-Specic Model (Male)"
        ]
    )
    + p9.scale_alpha_manual(
        name = "Model",
        drop = True,
        values = [1, 0.55, 0.55, 0.55],
        limits = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        breaks = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        labels = [
            "Baseline Model", "Extended Model", 
            "Sex-Specic Model (Female)", "Sex-Specic Model (Male)"
        ]
    )
    + p9.scale_size_manual(
        name = "Model",
        drop = True,
        values = [1, 0.2, 0.2, 0.2],
        limits = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        breaks = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        labels = [
            "Baseline Model", "Extended Model", 
            "Sex-Specic Model (Female)", "Sex-Specic Model (Male)"
        ]
    )
    + p9.theme(
        axis_line_x = p9.element_blank(),
        axis_line_y = p9.element_blank(), 
        plot_background = p9.element_blank(),
        strip_background = p9.element_blank(),
        legend_position = "bottom",
        legend_box_spacing = 0.5, # inches
        legend_background = p9.element_blank(),
        legend_box_background = p9.element_blank(),
        legend_key = p9.element_blank(),
        legend_key_size = 34,
        legend_entry_spacing = 17,
        panel_background = p9.element_blank(),
        panel_border = p9.element_blank(),
        panel_grid_major_x = p9.element_blank(),
        panel_grid_minor_x = p9.element_blank(),
        panel_grid_major_y = p9.element_line(size = 0.25, linetype = "solid", colour = "#767676"),
        panel_grid_minor_y = p9.element_blank(),
        panel_spacing_x = 0.25,
        panel_spacing_y = 0.35,
        axis_ticks_major_x = p9.element_line(size = 0.5, linetype = "solid", colour = "#1E1E1E"),
        axis_ticks_minor_x = p9.element_blank(),
        axis_ticks_major_y = p9.element_blank(), 
        axis_ticks_minor_y = p9.element_blank(),
        axis_ticks_direction_x = "out",
        axis_ticks_direction_y = "out", 
        axis_text_x = p9.element_text(family = "sans-serif", style = "normal", size = 10),
        axis_text_y = p9.element_text(family = "sans-serif", style = "normal", size = 10),
        axis_title = p9.element_text(family = "sans-serif", style = "normal", size = 10, linespacing = 1.5),
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(family = "sans-serif", style = "normal", size = 10),
        plot_title = p9.element_text(family = "sans-serif", style = "normal", size = 10),
        strip_text = p9.element_text(family = "sans-serif", style = "normal", size = 10),
        figure_size = (16, 10) # Inches
    )
    # This (esoteric) line is used to control how the legend looks, where legend
    # aesthetics are typically handled by the various calls to scale_FEATURE_manual.
    + p9.guides(colour = p9.guide_legend(override_aes = dict(alpha = 1, size = 4)))
)


figure_1_para_coord.save(
    filename = "F1_Proportions_Lender_Types_Parallel_Coordinates.svg",
    format = "svg",
    dpi = 900, width = 16, height = 10, units = "in",
    limitsize = False
)





# Figure 2 — Posterior Predictive Checks
# First, derive the Pandas dataframe to pass to ggplot for plotting.
# TODO: The calls to prepare_pymc_trace_for_ppc() are *VERY* slow (approx 10 min runtime). How to speed up?
# TODO: Perhaps this is one way to speed things up with groupby: https://stackoverflow.com/a/53148084
baseline_model_ppc = prepare_pymc_trace_for_ppc(inference_data = baseline_model_trace)
baseline_model_ppc["model"] = "Baseline" 

extended_model_ppc = prepare_pymc_trace_for_ppc(inference_data = extended_model_trace)
extended_model_ppc["model"] = "Extended" 

sex_model_ppc = prepare_pymc_trace_for_ppc(inference_data = sex_model_trace)
sex_model_ppc["model"] = "Sex" # For params. that are not sex-specific.


# Second, join model-specific dataframes of summaries of posterior predictive samples.
model_ppc = pd.concat(
    objs = [baseline_model_ppc, extended_model_ppc, sex_model_ppc],
    join = "outer",
    axis = 0,
    verify_integrity = False,
    sort = False
)

model_ppc = model_ppc.astype(
    dtype = {
        "categories": "string",
        "number_of_lenders": "int64",
        "Y_observed_count": "int64",
        "Y_count_ppc_mean": "float64",
        "Y_count_ppc_std": "float64",
        "model": "string"

    }
)


# Third, create ordered string variable for pretty plotting of results.
model_ppc["categories"] = pd.Categorical(model_ppc["categories"], categories = [
     "Y_counts_friend_lender_ij", 
     "Y_counts_family_lender_ij",
     "Y_counts_friend_family_lender_ij",
     "Y_counts_stranger_lender_ij",
     
     "Y_counts_friend_lender_ij_lender_ji",
     "Y_counts_family_lender_ij_lender_ji",
     "Y_counts_friend_family_lender_ij_lender_ji",
     "Y_counts_lender_ij_lender_ji" 
  ], ordered = True
)

model_ppc["Y_count_ppc_upper"] = model_ppc["Y_count_ppc_mean"] + (1.96 * model_ppc["Y_count_ppc_std"])
model_ppc["Y_count_ppc_lower"] = model_ppc["Y_count_ppc_mean"] - (1.96 * model_ppc["Y_count_ppc_std"])


# Fourth, create Figure 3
figure_2_ppc = (
    p9.ggplot(
        data = model_ppc
    ) 
    + p9.aes(
        x = "number_of_lenders", y = "Y_count_ppc_mean", colour = "model",
        ymin = "Y_count_ppc_lower", ymax = "Y_count_ppc_upper"
    )
    + p9.geom_ribbon(
        colour = "#767676",
        alpha = 0.10,
        linetype = "None",
        show_legend = False
    )
    + p9.geom_point(
        mapping = p9.aes(
            x = "number_of_lenders",
            y = "Y_observed_count"
        ),
        colour = "#767676",
        size = 1.50,
        alpha = 0.75,
        inherit_aes = False,
        show_legend = False
    ) 
    + p9.geom_hline(
        yintercept = [0, 10, 100, 1000],
        linetype = "solid",
        alpha = 0.5, size = 0.15,
        colour = "black"
    )
    + p9.geom_line(
            mapping = p9.aes(
            x = "number_of_lenders",
            y = "Y_observed_count"
        ),
        colour = "#767676",
        size = 0.25,
        linetype = "dashed",
        inherit_aes = False,
        show_legend = False
    )
    + p9.geom_point(
        size = 1.50,
        alpha = 0.75,
        show_legend = False
    )
    + p9.geom_line(
        size = 0.25,
        alpha = 0.75,
        linetype = "solid",
        show_legend = False
    )
    + p9.scale_x_continuous(breaks = [0, 1, 2, 3, 4, 5])
    + p9.scale_y_continuous(
        breaks = [0, 10, 100, 1000, 3000],
        limits = [-2, 3000], # Lower Bound for Ribbon/Confidence Bands Can Be Negative
        minor_breaks = 2,
        # https://mizani.readthedocs.io/en/stable/transforms.html
        trans = miz.transforms.pseudo_log_trans(base = 10, sigma = 1)
    )
    + p9.facet_grid(
        "model ~ categories",
        labeller = lambda lab: {
             "Y_counts_friend_lender_ij": "Best Friend\n(Non-indebted)",
             "Y_counts_family_lender_ij": "Salient Kin\n(Non-indebted)",
             "Y_counts_friend_family_lender_ij": "B. Friend + S. Kin\n(Non-indebted)",
             "Y_counts_stranger_lender_ij": "Not B. Friend or S. Kin\n(Non-indebted)",
             
             "Y_counts_friend_lender_ij_lender_ji": "Best Friend\n(Indebted)",
             "Y_counts_family_lender_ij_lender_ji": "Salient Kin\n(Indebted)",
             "Y_counts_friend_family_lender_ij_lender_ji": "B. Friend + S. Kin\n(Indebted)",
             "Y_counts_lender_ij_lender_ji": "Not B. Friend or S. Kin\n(Indebted)",
             
             "Baseline": "Baseline\nModel\n",
             "Extended": "Extended\nModel\n",
             "Sex": "Sex-Specific\nModel\n"
          }[lab]
     )
     # https://jtools.jacob-long.com/reference/jtools_colors.html
     # https://personal.sron.nl/~pault/
    + p9.scale_colour_manual(
        values = ["#CC3311", "#EE7733", "#0077BB"],
        breaks = ["Baseline", "Extended", "Sex"],
        labels = ["Baseline", "Extended", "Sex"]
    )
    # https://matplotlib.org/stable/tutorials/text/mathtext.html
    + p9.labs(
        x = ( # Parentheses creates an atom so we can break string over lines.
            "\nMean Frequency Across 12,000 Samples from the Posterior Predictive"
             " Distribuiton (Solid) versus Observed Frequency (Dashed) of Counts"
             " (0 - 5) of Lenders of Each Type"
        ),
        y = "Frequency in Modelled Compositional Count Matrix Y ($\mathregular{Log_{10}}$ Scale)\n",
        title = ""
    )
    + p9.theme(
        axis_line_x = p9.element_blank(),
        axis_line_y = p9.element_blank(), 
        plot_background = p9.element_blank(),
        strip_background = p9.element_blank(),
        legend_position = "bottom",
        legend_box_spacing = 0.5, # inches
        legend_background = p9.element_blank(),
        legend_box_background = p9.element_blank(),
        legend_key = p9.element_blank(),
        legend_key_size = 8,
        legend_entry_spacing = 16,
        panel_background = p9.element_blank(),
        panel_border = p9.element_blank(),
        panel_grid_major_x = p9.element_blank(),
        panel_grid_minor_x = p9.element_blank(),
        panel_grid_major_y = p9.element_blank(), # p9.element_line(size = 0.25, linetype = "solid", colour = "#767676"),
        panel_grid_minor_y = p9.element_blank(),
        panel_spacing_x = 0.25,
        panel_spacing_y = 0.35,
        axis_ticks_major_x = p9.element_line(size = 0.5, linetype = "solid", colour = "#1E1E1E", alpha = 0.05),
        axis_ticks_minor_x = p9.element_blank(),
        axis_ticks_major_y = p9.element_blank(),
        axis_ticks_minor_y = p9.element_line(size = 0.5, linetype = "solid", colour = "#1E1E1E", alpha = 0.05),
        axis_ticks_direction_x = "out",
        axis_ticks_direction_y = "out", 
        axis_text_x = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        axis_text_y = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        axis_title = p9.element_text(family = "sans-serif", style = "normal", size = 8, linespacing = 1.5),
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        plot_title = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        strip_text = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        figure_size = (12.5, 13) # Inches
    )
    # This (esoteric) line is used to control how the legend looks, where legend
    # aesthetics are typically handled by the various calls to scale_FEATURE_manual.
    + p9.guides(colour = p9.guide_legend(override_aes = dict(alpha = 1, size = 4)))
) 
 
figure_2_ppc.save(
    filename = "F2_Posterior_Predictive_Checks_Count_Frequencies.svg",
    format = "svg",
    dpi = 900, width = 13, height = 5, units = "in",
    limitsize = False
)





# Supplementary Figure 2 — Posterior Mean Proportions (Wide/Not Overplotted)
supplementary_figure_1_category_fractions = (
    p9.ggplot(
        data = model_pmeans[model_pmeans["parameter"].isin(fraction_parameter_names)]
    ) 
    + p9.aes(
        x = "parameter", y = "pmean",
        ymin = "lower_bound", ymax = "higher_bound",
        colour = "model"
    )
    + p9.geom_pointrange(
        size = 0.7, alpha = 1, shape = "o", linetype = "solid", fatten = 3,
        position = p9.position_dodge(width = 0.40),
        show_legend = True
    )
    + p9.facet_wrap(
        "categories",
         ncol = 2,
         dir = "v",
         # scales = "free_y",
         labeller = lambda lab: {
             "friend_lender_ij": "Best Friend\n(Non-indebted)",
             "family_lender_ij": "Salient Kin\n(Non-indebted)",
             "friend_family_lender_ij": "Best Friend + Salient Kin\n(Non-indebted)",
             "stranger_lender_ij": "Not Best Friend or Salient Kin\n(Non-indebted)",
             
             "friend_lender_ij_lender_ji": "Best Friend\n(Indebted)",
             "family_lender_ij_lender_ji": "Salient Kin\n(Indebted)",
             "friend_family_lender_ij_lender_ji": "Best Friend + Salient Kin\n(Indebted)",
             "lender_ij_lender_ji": "Not Best Friend or Salient Kin\n(Indebted)"
          }[lab]
     )
    + p9.labs(
        x = "\nVillage",
        y = "Posterior Mean Proportion + 95% Highest Density Interval (Square-Root Scale)\n"
    )
    + p9.scale_x_discrete(
        breaks = ["frac"] + [f"frac_vill_{group}" for group in villages_IDchar],
        labels = ["All"] + [f"V{group}" for group in villages_IDchar]
    )
    + p9.scale_y_sqrt(
        limits = [0, 0.6],
        breaks = [0, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )    
    # https://jtools.jacob-long.com/reference/jtools_colors.html
    # https://personal.sron.nl/~pault/
    + p9.scale_colour_manual(
        name = "Model",
        drop = True,
        values = ["#CC3311", "#EE7733", "#0077BB", "#EECC66"],
        limits = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        breaks = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        labels = [
            "Baseline Model", "Extended Model", 
            "Sex-Specic Model (Female)", "Sex-Specic Model (Male)"
        ]
    )
    + p9.theme(
        axis_line_x = p9.element_blank(),
        axis_line_y = p9.element_blank(), 
        plot_background = p9.element_blank(),
        strip_background = p9.element_blank(),
        legend_position = "bottom",
        legend_box_spacing = 0.5, # inches
        legend_background = p9.element_blank(),
        legend_box_background = p9.element_blank(),
        legend_key = p9.element_blank(),
        legend_key_size = 8,
        legend_entry_spacing = 16,
        panel_background = p9.element_blank(),
        panel_border = p9.element_blank(),
        panel_grid_major_x = p9.element_blank(),
        panel_grid_minor_x = p9.element_blank(),
        panel_grid_major_y = p9.element_line(size = 0.25, linetype = "solid", colour = "#767676"),
        panel_grid_minor_y = p9.element_blank(),
        panel_spacing_x = 0.25,
        panel_spacing_y = 0.35,
        axis_ticks_major_x = p9.element_line(size = 0.5, linetype = "solid", colour = "#1E1E1E"),
        axis_ticks_minor_x = p9.element_blank(),
        axis_ticks_major_y = p9.element_blank(), 
        axis_ticks_minor_y = p9.element_blank(),
        axis_ticks_direction_x = "out",
        axis_ticks_direction_y = "out", 
        axis_text_x = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        axis_text_y = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        axis_title = p9.element_text(family = "sans-serif", style = "normal", size = 8, linespacing = 1.5),
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        plot_title = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        strip_text = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        figure_size = (12.5, 13) # Inches
    )
    # This (esoteric) line is used to control how the legend looks, where legend
    # aesthetics are typically handled by the various calls to scale_FEATURE_manual.
    + p9.guides(colour = p9.guide_legend(override_aes = dict(alpha = 1, size = 0.4)))
)


supplementary_figure_1_category_fractions.save(
    filename = "SF1_Proportions_Lender_Types_Wide.svg",
    format = "svg",
    dpi = 900, width = 12.5, height = 13, units = "in",
    limitsize = False
)





# Supplementary Figure 2 — Posterior Mean Concentration Factors
supplementary_figure_2_concentration_factors = (
    p9.ggplot(
        data = model_pmeans[model_pmeans["parameter"].isin(concentration_parameter_names)]
    ) 
    + p9.aes(
        x = "parameter", y = "pmean",
        ymin = "lower_bound", ymax = "higher_bound",
        colour = "model"
    )
    + p9.geom_pointrange(
        size = 0.7, alpha = 1, shape = "o", linetype = "solid", fatten = 3,
        position = p9.position_dodge(width = 0.35),
        show_legend = True
    )
    + p9.labs(
        x = "\nVillage",
        y = "Posterior Mean Concentration + 95 Highest Density Interval\n"
    )
    + p9.scale_x_discrete(
        breaks = ["conc"] + [f"conc_vill_{group}" for group in villages_IDchar],
        labels = ["All"] + [f"V{group}" for group in villages_IDchar]
    )
    + p9.scale_y_continuous(limits = [0, 15], breaks = [0, 5, 10, 15])
    + p9.scale_colour_manual(
        name = "Model",
        drop = True,
        values = ["#CC3311", "#EE7733", "#0077BB", "#EECC66"],
        limits = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        breaks = ["Baseline", "Extended", "Sex (Female)", "Sex (Male)"],
        labels = [
            "Baseline Model", "Extended Model", 
            "Sex-Specic Model (Female)", "Sex-Specic Model (Male)"
        ]
    )
    + p9.theme(
        axis_line_x = p9.element_blank(),
        axis_line_y = p9.element_blank(), 
        plot_background = p9.element_blank(),
        strip_background = p9.element_blank(),
        legend_position = "bottom",
        legend_box_spacing = 0.5, # inches
        legend_background = p9.element_blank(),
        legend_box_background = p9.element_blank(),
        legend_key = p9.element_blank(),
        legend_key_size = 8,
        legend_entry_spacing = 16,
        panel_background = p9.element_blank(),
        panel_border = p9.element_blank(),
        panel_grid_major_x = p9.element_blank(),
        panel_grid_minor_x = p9.element_blank(),
        panel_grid_major_y = p9.element_line(size = 0.25, linetype = "solid", colour = "#767676"),
        panel_grid_minor_y = p9.element_blank(),
        panel_spacing_x = 0.25,
        panel_spacing_y = 0.35,
        axis_ticks_major_x = p9.element_line(size = 0.5, linetype = "solid", colour = "#1E1E1E"),
        axis_ticks_minor_x = p9.element_blank(),
        axis_ticks_major_y = p9.element_blank(), 
        axis_ticks_minor_y = p9.element_blank(),
        axis_ticks_direction_x = "out",
        axis_ticks_direction_y = "out", 
        axis_text_x = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        axis_text_y = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        axis_title = p9.element_text(family = "sans-serif", style = "normal", size = 8, linespacing = 1.5),
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        plot_title = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        strip_text = p9.element_text(family = "sans-serif", style = "normal", size = 8),
        figure_size = (12.5, 13) # Inches
    )
    # This (esoteric) line is used to control how the legend looks, where legend
    # aesthetics are typically handled by the various calls to scale_FEATURE_manual.
    + p9.guides(colour = p9.guide_legend(override_aes = dict(alpha = 1, size = 0.4)))
)

supplementary_figure_2_concentration_factors.save(
    filename = "SF2_Concentration_Factors_Wide.svg",
    format = "svg",
    dpi = 900, width = 13, height = 4.5, units = "in",
    limitsize = False
)




# ## Village-Specific Colours
# village_ID_colours = {
#     "7" : "#2166AC",
#    "9"  : "#4393C3",
#    "11" : "#FFEE99",
#    "14" : "#D1E5F0", 
#    "29" : "#B2182B",
#    "40" : "#D6604D",
#    "46" : "#F4A582",
#    "47" : "#FDDBC7",
#    "50" : "#762A83",
#    "51" : "#9970AB",
#    "55" : "#C2A5CF",
#    "66" : "#E7D4E8", 
#    "75" : "#1B7837",
#    "77" : "#5AAE61",
#    "80" : "#ACD39E",
#    "81" : "#D9F0D3"
# }


