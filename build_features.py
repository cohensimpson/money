#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/


import os
os.chdir("/Users/cohen/Desktop IconFree/GitHub/money")


import numpy as np
import pandas as pd 
import itertools

pd.set_option("display.max_rows", 20)
pd.set_option("display.min_rows", 20)




# TODO: Determine if the categorical data need qualitative labels for cmdstanpy.
# TODO: Is this useful: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html

# Create dictionary wherein the keys are each of the 16 village IDs and
# the values are the IDs of each villages' residents as these individuals are
# ordered in the column "survey_responses["village_ID"]".
villagers = {}
for village in village_IDs:
    survey_responses_village = survey_responses[survey_responses["village_ID"] == village]
    villagers[village] = survey_responses_village.index 

del village, survey_responses_village


# This analysis concerns asymmetric relationships between the residents of each village.
# Accordingly, for each village, construct a data frames wherein 
# each row is for an *ordered* dyad, where the villager who sends an
# an asymmetric relationship/makes a nomination (e.g., for friendship) is
# labeled "i_ID", and the receiving/nominated actor is labeled "j_ID".
# Note, nominating oneself is disallowed (i.e., no "self loops" in network science jargon).
all_village_dyads = {}
for village in village_IDs:
    
    # https://docs.python.org/3/library/itertools.html#itertools.permutations
    village_dyads = itertools.permutations(iterable = villagers[village], r = 2)
    village_dyads = list(village_dyads)
    village_dyads = pd.DataFrame(
        data = (
            (
                villager_pair[0], 
                villager_pair[1], 
                villager_pair[0] + "_" + villager_pair[1]
            ) 
            for villager_pair in village_dyads
        ),
        
        columns =['i_ID', 'j_ID', "ij_ID"]
    )
    
    all_village_dyads[village] = village_dyads
    
del village, village_dyads


# Concatenate/stack/bind each village-specific data frame of ordered dyads to 
# create the backbone of the backbone of the design matrix to use for the
# statistical analysis/to feed to cmdstanpy.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
all_village_dyads = pd.concat(
    objs = all_village_dyads,
    axis = 0, # Concatenate/combine along rows (i.e., a vertical stack)
    join = "outer",
    ignore_index = True,
    sort = False
)
all_village_dyads = all_village_dyads.set_index("ij_ID", drop = True)



# A left join of two pandas data frames will will result in additional
# rows being added to the result when the right-position data frame has
# multiple matches for a key/index in the left-position data frame.
# Here, 
# To get around this, first use groupby to sum dummy variables for relationship
# type as rows for the different relationships are stacked such that
# an ordered dyad can appear multiple times if i/i_ID nominates j/j_ID
# in response to multiple name generators and the data geographic distance. 
# To see this, RUN: nominations.loc[nominations.index[nominations.index.duplicated()]]

# https://stackoverflow.com/questions/22720739/pandas-left-outer-join-results-in-table-larger-than-left-table
nominations_wide = pd.get_dummies(nominations["type"])
del nominations_wide["speak"]
nominations_wide = nominations_wide.groupby(by = nominations_wide.index, axis = 0).sum()
nominations_wide.loc[:, ["geo"]] = nominations_wide["geo"].index
nominations_wide.loc[:, ["geo"]] = nominations_wide["geo"].map(nominations[nominations["dist"] >= 0]["dist"])


test = pd.merge(
    how = "left",
    left = all_village_dyads, 
    right = nominations_wide, 
    # left_on = "ID", 
    # right_on = "ID"
    left_index = True,
    right_index = True,
    suffixes = ("_dyads", "_noms"),
    indicator = False
)



# test.loc[test["geo"].isna(), ["geo"]]

# Geographic distance is only recorded for each unordered dyad and is
# otherwise missing. Accordingly, take the distance for each ordered
# dyad (i_ID, j_ID) and assign it to the distance for (j_ID, i_ID).
# test = test.loc[
#     [
#         "11100101_11100201", "11100201_11100101",
#         "11100101_11100401", "11100401_11100101",
#         "11100101_11203703", "11203703_11100101",
#         "19203602_19101502", "19101502_19203602",
#         "24414001_24414102", "24414102_24414001",
#         "991837012_991837006", "991837006_991837012",
#         
#     ]
# ]

test.update(
    other = test.loc[test["j_ID"] + "_" + test["i_ID"], ["geo"]].set_index(test["i_ID"] + "_" + test["j_ID"]),
    join = "left",
    overwrite = False, 
    errors = "ignore"
) 
test = test.rename(columns = {"geo": "geo_ij"})



# Here, 0 == i_ID did not nominate j_ID
test = test.fillna(
    value = {
        "lender": 0,
        "friend": 0,
        "family": 0,
        "Contgame": 0,
        "solver": 0,
        
    }
)

# "lender" documents the answer to the following sociometric survey question:
# “Think about up to five people in this village that you would ask to 
# borrow a significant amount of money if you had a personal emergency.”
# "lender" features in the analysis in its raw form (i.e., "lender_ij")
# and in its transposed form (i.e., "lender_ji"), the latter of which is used
# to capture reciprocity. Note that Pandas will add a new column of values for
# "lender_ji" (i.e., the transposed values for "lender_ij") by matching 
# on row indices, negating the transpose. Accordingly, create "lender_ji"
# using just the values of the rearranged series created from "lender_ij".
# https://stackoverflow.com/questions/56715112/how-to-add-a-pandas-series-to-a-dataframe-ignoring-indices
# https://stackoverflow.com/questions/15979339/how-to-assign-columns-while-ignoring-index-alignment
test = test.rename(columns = {"lender": "lender_ij"})
test["lender_ji"] = test.loc[test["j_ID"] + "_" + test["i_ID"], :]["lender_ij"].values


# "friend" documents the answer to the following sociometric survey question:
# “Think about up to five of your best friends in this village. By friends I mean
# someone who will help you when you have a problem or who spends much of
# his or her free time with you. If there are less than five, that is okay too.”
# "lender" features in the analysis in its raw form (i.e., "friend_ij")
test = test.rename(columns = {"friend": "friend_ij"})


# "family" is a combination of information on coresidence and a villagers
# answer to the following sociometric survey question:
# “Think about up to five family members in this village not living in your
# household with whom you most frequently spend time. For instance, you might
# visit one another, eat meals together, or attend events together.”
# However, not all possible asymmetric ties between people who live together 
# based on the household membership data — i.e. coresidence — appear
# in the “family” nominations in “ties.csv”. 

# As detailed in personal communication (i.e, emails) with
# Romain Ferrali (25 February 2022), the gentleman who collected the Ugandan
# data, there are three sources of information for kinship. There is the baseline
# household membership data that Romain and his colleagues gathered prior to
# their full survey and cleaned ex-post. There is the household membership data
# that Romain and his colleagues collected during the survey. And, finally,
# there is the data on non-coresident kin elicited using the sociometric
# question for family/kinship (above).

# According to Romain, the discrepancy between coresidence and the “family”
# nominations in “ties.csv” is due to his team failing to properly clean
# the original/baseline household membership data by updating these data 
# through the addition of all new household members discovered during their
# survey but who did not appear in the baseline household membership data.

# Note that the “family” nominations in “ties.csv” already includes the 
# connections with the household members uncovered by Romain and his colleagues 
# during the survey. Accordingly, for my analysis for each village, I simply add
# the symmetric networks constructed with the “family” nominations in “ties.csv”
# to a symmetric network  constructed using the original/baseline household
# membership data  — i.e., coresidence networks that only include all possible
# asymmetric ties between people who live together
# according to "nodes_CPS_Version_1.2.csv". 
test = test.rename(columns = {"family": "family_ij"})

# Set (j_ID, i_ID) equal to one if i_ID names j_ID as family, and vice versa.
test["kinship_ij"] = test["family_ij"]
test["kinship_ji"] = test.loc[test["j_ID"] + "_" + test["i_ID"], :]["kinship_ij"].values

# Binary indicator for whether or not i_ID and j_ID live in the same household
# https://stackoverflow.com/a/27475514
test["coresidents"] = np.where(
    survey_responses.loc[test["i_ID"], :]["HH_ID"].values == survey_responses.loc[test["j_ID"], :]["HH_ID"].values, 
    1, 0
)
# test["coresidents"] = (
#     survey_responses.loc[test["i_ID"], :]["HH_ID"].values == survey_responses.loc[test["j_ID"], :]["HH_ID"].values
# ).astype("int64")

# Combine kinship nominations and coresidence and binarise
test["family_ij"] = test["kinship_ij"] + test["kinship_ji"] + test["coresidents"] 
test["family_ij"] = (test["family_ij"] > 0)
del test["kinship_ij"], test["kinship_ji"], test["coresidents"]

# "solver" documents the answer to the following sociometric survey question:
# “Imagine there is a problem with public services in this village. For example, 
# you might imagine that a teacher has not come to school for several days or
# that a borehole in your village needs to be repaired. Think about up to
# five people in this village whom you would be most likely to approach to help
# solve these kinds of problems.”
test = test.rename(columns = {"solver": "problemsolver_ij"})


# "contgame" documents results from a modified public goods game in all 16 villages. 
# Specifically, villagers were given an opportunity to contribute to the village
# any share of their survey participation remuneration (n.b., contributions 
# matched by survey team). For the public goods game, villagers were asked to
# name which individual they would like to handle funds on behalf of the village, 
# regardless of whether that individual holds formal leadership position.
# "contgame" indicates whom a villager chose as a money handler.
test = test.rename(columns = {"Contgame": "goodsgame_ij"})



# Incorporate the data on villagers' attributes into the data frame of ordered dyads.
test["female_i"] = survey_responses.loc[test["i_ID"], :]["female"].values
test["female_j"] = survey_responses.loc[test["j_ID"], :]["female"].values
test["same_gender_ij"] = test["female_i"] == test["female_j"]

test["age_i"] = survey_responses.loc[test["i_ID"], :]["age"].values  # np.square in models
test["age_j"] = survey_responses.loc[test["j_ID"], :]["age"].values  # np.square in models
test["age_absdiff_ij"] = np.absolute(test["age_i"] - test["age_j"])  # np.sqrt in models

test["edu_full_i"] = survey_responses.loc[test["i_ID"], :]["edu_full"].values
test["edu_full_j"] = survey_responses.loc[test["j_ID"], :]["edu_full"].values
test["same_edu_full_ij"] = test["edu_full_i"] == test["edu_full_j"]

test["catholic_i"] = survey_responses.loc[test["i_ID"], :]["rlg"].values
test["catholic_j"] = survey_responses.loc[test["j_ID"], :]["rlg"].values
test["same_catholic_ij"] = np.where(
    (test["catholic_i"] == 1) & (test["catholic_j"] == 1),
    1, 0
)

test["income_i"] = survey_responses.loc[test["i_ID"], :]["income"].values
# test["income_j"] = survey_responses.loc[test["j_ID"], :]["income"].values

test["hasPhone_i"] = survey_responses.loc[test["i_ID"], :]["hasPhone"].values
# test["hasPhone_j"] = survey_responses.loc[test["j_ID"], :]["hasPhone"].values

test["leader_i"] = survey_responses.loc[test["i_ID"], :]["leader"].values
# test["leader_j"] = survey_responses.loc[test["j_ID"], :]["leader"].values

test["HH_Head_i"] = survey_responses.loc[test["i_ID"], :]["HH_Head"].values
# test["HH_Head_j"] = survey_responses.loc[test["j_ID"], :]["HH_Head"].values


# Incorporate the data on villagers' village into the data frame of ordered dyads.
test["village_ID"] = survey_responses.loc[test["i_ID"], :]["village_ID"].values
test["village_pop_size"] = villages.loc[test["village_ID"], :]["census_pop"].values
test["village_emp_rate"] = villages.loc[test["village_ID"], :]["census_employ"].values
test["village_nonag_emp_rate"] = villages.loc[test["village_ID"], :]["census_noAgri"].values
test["village_savingsgroup"] = villages.loc[test["village_ID"], :]["pg_savingsgroup"].values
test["village_market"] = villages.loc[test["village_ID"], :]["pg_market_any"].values
test["village_distArua"] = villages.loc[test["village_ID"], :]["distArua"].values


test = test.loc[:, ["i_ID", "j_ID", "village_ID",
                    "lender_ij", "lender_ji", "friend_ij", "family_ij",
                    "problemsolver_ij", "goodsgame_ij", "geo_ij",
                    "female_i", "female_j", "same_gender_ij", 
                    "age_i", "age_j", "age_absdiff_ij",
                    "edu_full_i", "edu_full_j", "same_edu_full_ij", 
                    "catholic_i", "catholic_j", "same_catholic_ij", 
                    "income_i", "hasPhone_i", "leader_i", "HH_Head_i",
                    "village_pop_size",
                    "village_emp_rate",
                    "village_nonag_emp_rate",
                    "village_savingsgroup",
                    "village_market",
                    "village_distArua"
                ]
]

test = test.astype(
    dtype = {
        "i_ID": "string",
        "j_ID": "string",
        "village_ID": "string",
        "lender_ij": "Int64", 
        "lender_ji": "Int64", 
        "friend_ij": "Int64", 
        "family_ij": "Int64",
        "problemsolver_ij": "Int64", 
        "goodsgame_ij": "Int64", 
        "geo_ij": "float64",
        "female_i": "float64", # Values imputed using posterior means
        "female_j": "float64", # Values imputed using posterior means
        "same_gender_ij": "Int64", 
        "age_i": "Int64", 
        "age_j": "Int64", 
        "age_absdiff_ij": "Int64",
        "edu_full_i": "float64", # Values imputed using posterior means
        "edu_full_j": "float64", # Values imputed using posterior means
        "same_edu_full_ij": "Int64", 
        "catholic_i": "Int64", 
        "catholic_j": "Int64", 
        "same_catholic_ij": "Int64", 
        "income_i": "float64", # Values imputed using posterior means
        "hasPhone_i": "float64", # Values imputed using posterior means
        "leader_i": "Int64", 
        "HH_Head_i": "float64", # Values imputed using posterior means
        "village_pop_size": "Int64",
        "village_emp_rate": "float64",
        "village_nonag_emp_rate": "float64",
        "village_savingsgroup": "Int64",
        "village_market": "Int64",
        "village_distArua": "float64"
    }
)

# Basic information about the data frame of ordered dyads
test.info()








