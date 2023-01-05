#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/





# Multiple quantities are reported in the Results section of my paper. However,
# as there is no "big table of estimates", only figures visualising estimates,
# the following code is used to obtain the reported quantities and is included
# on GitHub for transparency/reproducibility. Note that this script will only
# run without error after running "visualiation.py" in full as the Pandas 
# dataframe of results — i.e., "model_pmeans" — is created therin. 

# Note, for pd.query(), the strings tested for equality must be wrapped in quotes.
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html




# Reported quantities (i.e., the fraction of lenders of various types) only
# only relate to the baeline model, not the "Extended" or "Sex"-specific models.
model, parm = "Baseline", "frac"




# Posterior Mean Proportion of Lenders Who are Non-indebted Non-Friend Kin
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'family_lender_ij'",
    inplace = False
)




# Posterior Mean Proportion of Lenders Who are Non-indebted Non-Kin-Who-Are-Not-Friends (i.e., Strangers)
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'stranger_lender_ij'",
    inplace = False
)




# Posterior Mean Proportion of Lenders Who are Non-indebted Non-Kin Friends
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'friend_lender_ij'",
    inplace = False
)




# Posterior Mean Proportion of Lenders Who are Non-indebted Friends-Who-Are-Kin
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'friend_family_lender_ij'",
    inplace = False
)




# Village-Specific Posterior Mean Proportion of Lenders Who are Non-indebted Friends-Who-Are-Kin
# Note, model_pmeans includes posterior means and HDIs for parameters from all three models
# On the calling of local variables in pd.query with "@", see:
# https://stackoverflow.com/a/57696055
np.sort(
    model_pmeans.query(
        f"parameter.isin(@fraction_parameter_names) and categories == 'friend_family_lender_ij'",
        inplace = False
    )["pmean"].values
)




# Posterior Mean Proportion of Lenders Who are Indebted Non-Kin-Who-Are-Not-Friends (i.e., Pure Reciprocity)
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'lender_ij_lender_ji'",
    inplace = False
)




# Posterior Mean Proportion of Lenders Who are Indebted Non-Friend Kin
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'family_lender_ij_lender_ji'",
    inplace = False
)




# Posterior Mean Proportion of Lenders Who are Indebted Non-Kin Friends
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'friend_lender_ij_lender_ji'",
    inplace = False
)




# Posterior Mean Proportion of Lenders Who are Indebted Friends-Who-Are-Kin
model_pmeans.query(
    f"model == '{model}' and parameter == '{parm}' and categories == 'friend_family_lender_ij_lender_ji'",
    inplace = False
)




# Sum of Posterior Mean Proportions of Lenders of Any Type Who are Indebted 
types_of_lender_indebted = [
    "friend_lender_ij_lender_ji", 
    "family_lender_ij_lender_ji",
    "friend_family_lender_ij_lender_ji",
    "lender_ij_lender_ji"
]

sum([
    model_pmeans.query(
        f"model == '{model}' and parameter == '{parm}' and categories == '{lender}'",
        inplace = False
    )["pmean"].item()
    for lender in types_of_lender_indebted
])




# Sum of Posterior Mean Proportions of Lenders Who are Intimates, Ignoring Indebtedness
types_of_lender_intimates = [
    "friend_lender_ij", 
    "family_lender_ij",
    "friend_family_lender_ij",
    "friend_lender_ij_lender_ji", 
    "family_lender_ij_lender_ji",
    "friend_family_lender_ij_lender_ji"
]

sum([
    model_pmeans.query(
        f"model == '{model}' and parameter == '{parm}' and categories == '{lender}'",
        inplace = False
    )["pmean"].item()
    for lender in types_of_lender_intimates
])



