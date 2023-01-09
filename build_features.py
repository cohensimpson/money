#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/






# Create dictionary wherein the keys are each of the 16 village IDs and
# the values are the IDs of each villages' residents. IDs of the residents 
# are ordered as they appear in the column "survey_responses["village_ID"]".
villagers = {}
for village in village_IDs:
    survey_responses_village = survey_responses[survey_responses["village_ID"] == village]
    villagers[village] = survey_responses_village.index 

del village, survey_responses_village




# This analysis concerns asymmetric lending relationships between the residents 
# of each village. Accordingly, for each village, construct a data frames wherein 
# each row is for an *ordered* dyad. The villager who sends an an asymmetric 
# relationship (i.e., the villager making the nomination, e.g., for friendship) 
# is labeled "i_ID", and the receiving/nominated actor is labeled "j_ID".
# Note, nominating oneself is disallowed (i.e., "self loops" in network-science jargon).
all_village_dyads = {}
for village in village_IDs:
    
    # Derive all possible pairs of two villagers in a village and use tuple
    # comprehension to build a Pandas dataframe.
    # https://docs.python.org/3/library/itertools.html#itertools.permutations
    village_dyads = itertools.permutations(iterable = villagers[village], r = 2)
    village_dyads = list(village_dyads)
    village_dyads = pd.DataFrame(
        data = (
            (
                villager_pair[0], 
                villager_pair[1], 
                villager_pair[0] + "_" + villager_pair[1],
                "_".join(sorted(villager_pair))
            ) 
            for villager_pair in village_dyads
        ),
        
        columns =['i_ID', 'j_ID', "ij_ID", "unordered_ij_ID"] 
    )
    
    all_village_dyads[village] = village_dyads
    




# Concatenate/stack/bind each village-specific dataframe of ordered dyads.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
all_village_dyads = pd.concat(
    objs = all_village_dyads,
    axis = 0, # Concatenate/combine along rows (i.e., a vertical stack)
    join = "outer",
    ignore_index = True,
    sort = False
)
all_village_dyads = all_village_dyads.set_index("ij_ID", drop = True)




# A left join of two pandas data frames will will result in additional rows being
# added to the result when the right-position data frame has multiple matches for
# a key/index in the left-position data frame. In the raw nominations data, different
# relationships are stacked such that an a specific ordered dyad can appear
# multiple times if i/i_ID nominates j/j_ID in response to multiple name generators.
# To get around this, first use groupby to sum dummy variables for relationship
# type and then merge the result with the dataframe of all, unique ordered dyads.
# To see this, RUN: nominations.loc[nominations.index[nominations.index.duplicated()]]

# https://stackoverflow.com/questions/22720739/pandas-left-outer-join-results-in-table-larger-than-left-table
nominations_wide = pd.get_dummies(nominations["type"])
nominations_wide = nominations_wide.groupby(by = nominations_wide.index, axis = 0).sum()

all_village_dyads = pd.merge(
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




# If an ordered dyad does not appear in nominations it is not "missing". Instead,
# it means that i_ID did not nominate j_ID. Accordingly, replace NaN with "0"
all_village_dyads = all_village_dyads.fillna(
    value = {
        "lender": 0,
        "friend": 0,
        "family": 0
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
all_village_dyads = all_village_dyads.rename(columns = {"lender": "lender_ij"})

# Note the reversal of i_ID and j_ID!
all_village_dyads["lender_ji"] = (
    all_village_dyads.loc[all_village_dyads["j_ID"] + "_" + all_village_dyads["i_ID"], :]["lender_ij"].values
)




# "friend" documents the answer to the following sociometric survey question:
# “Think about up to five of your best friends in this village. By friends I mean
# someone who will help you when you have a problem or who spends much of
# his or her free time with you. If there are less than five, that is okay too.”
# "lender" features in the analysis in its raw form (i.e., "friend_ij")
all_village_dyads = all_village_dyads.rename(columns = {"friend": "friend_ij"})




# "family" is a combination of information on coresidence and a villager's
# answer to the following survey question:
# “Think about up to five family members in this village not living in your
# household with whom you most frequently spend time. For instance, you might
# visit one another, eat meals together, or attend events together.”

# As discussed in my paper, not all possible asymmetric ties between people who 
# live together based on the household membership data — i.e. coresidence — 
# appear in the “family” nominations documented in “ties.csv”. 

# As detailed in personal communication (i.e, emails) with
# Romain Ferrali (25 February 2022), the gentleman who collected the Ugandan
# data, there are three sources of information for kinship. There is the baseline
# household membership data that Romain and his colleagues gathered prior to
# their full survey and cleaned ex-post. There is the household membership data
# that Romain and his colleagues collected during the survey. And, finally,
# there is the data on non-coresident kin elicited using the sociometric
# question for "family" (above).

# Note that the “family” nominations in “ties.csv” already includes some of the
# connections between household members uncovered by Romain and his colleagues 
# during the survey. Accordingly, for each village, I simply add
# symmetric connections based on the “family” nominations in “ties.csv”
# to symmetric connections constructed using the original/baseline household
# membership data in "nodes_CPS_Version_1.2.csv". 
all_village_dyads = all_village_dyads.rename(columns = {"family": "family_ij"})

# Set (j_ID, i_ID) equal to one if i_ID names j_ID as family, and vice versa.
all_village_dyads["kinship_ij"] = all_village_dyads["family_ij"]
all_village_dyads["kinship_ji"] = (
    all_village_dyads.loc[all_village_dyads["j_ID"] + "_" + all_village_dyads["i_ID"], :]["kinship_ij"].values
)

# Binary indicator for whether or not i_ID and j_ID live in the same household
# https://stackoverflow.com/a/27475514
all_village_dyads["coresidents"] = np.where(
    survey_responses.loc[all_village_dyads["i_ID"], :]["HH_ID"].values == survey_responses.loc[all_village_dyads["j_ID"], :]["HH_ID"].values, 
    1, 0
)

# Combine kinship nominations and coresidence and binarise
all_village_dyads["family_ij"] = (
    all_village_dyads["kinship_ij"]
    + all_village_dyads["kinship_ji"]
    + all_village_dyads["coresidents"]
)
all_village_dyads["family_ij"] = np.where(all_village_dyads["family_ij"] > 0, 1, 0)

del all_village_dyads["kinship_ij"]
del all_village_dyads["kinship_ji"]
del all_village_dyads["coresidents"]




all_village_dyads = all_village_dyads.loc[:, ["i_ID", "j_ID", "unordered_ij_ID",
                                "lender_ij", "lender_ji", "friend_ij", "family_ij"
                                ]
]

all_village_dyads = all_village_dyads.astype(
    dtype = {
        "i_ID": "string",
        "j_ID": "string",
        "unordered_ij_ID": "string",
        "lender_ij": "int64", 
        "lender_ji": "int64", 
        "friend_ij": "int64", 
        "family_ij": "int64"
    }
)




# Use booleans to construct new binary indicators for whether a given villager
# nominates someone as a money lender relative to alter patronage (lender_ji),
# friendship, and kinship.
lend_comp = all_village_dyads
lend_comp = lend_comp.loc[:, ["i_ID", "j_ID", "lender_ij",
                     "friend_ij", "family_ij", "lender_ji"]
                 ]

# Lenders who are also non-indebted friends.
lend_comp["friend_lender_ij"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 1) & (lend_comp["family_ij"] == 0) & (lend_comp["lender_ji"] == 0),
    1, 0
)

# Lenders who are also non-indebted kin.
lend_comp["family_lender_ij"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 0) & (lend_comp["family_ij"] == 1) & (lend_comp["lender_ji"] == 0),
     1, 0
)

# Lenders who are non-indebted but also friends and kin.
lend_comp["friend_family_lender_ij"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 1) & (lend_comp["family_ij"] == 1) & (lend_comp["lender_ji"] == 0),
     1, 0
 )
 
# Lenders who are neither friends, kin, or indebted.
lend_comp["stranger_lender_ij"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 0) & (lend_comp["family_ij"] == 0) & (lend_comp["lender_ji"] == 0),
     1, 0
)

# Lenders who are also indebted friends.
lend_comp["friend_lender_ij_lender_ji"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 1) & (lend_comp["family_ij"] == 0) & (lend_comp["lender_ji"] == 1),
    1, 0
)

# Lenders who are also indebted kin.
lend_comp["family_lender_ij_lender_ji"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 0) & (lend_comp["family_ij"] == 1) & (lend_comp["lender_ji"] == 1),
     1, 0
)

# Lenders who are also friends, kin, and indebted.
lend_comp["friend_family_lender_ij_lender_ji"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 1) & (lend_comp["family_ij"] == 1) & (lend_comp["lender_ji"] == 1),
     1, 0
 )
 
# Lenders who are indebted but neither friends or kin.
lend_comp["lender_ij_lender_ji"] = np.where(
    (lend_comp["lender_ij"] == 1) & (lend_comp["friend_ij"] == 0) & (lend_comp["family_ij"] == 0) & (lend_comp["lender_ji"] == 1),
     1, 0
)




# Collapse the data frame of ordered dyads (i_ID, j_ID) into a dataframe wherein
# each row is for a villager and the columns are the number of types of lenders
# that a villager reports (i.e., lenders who are either: friends, kin, friend
# and kin,  and neither friend or kin (strangers)). Note that categories for
# types of lender are constructed to be exclusive given the available data. In 
# this respect, the counts are *compositional* as that they describe the breakdown
# of a villager's particular basket of lenders. For those from network science,
# these counts are (out-)degrees.
lend_comp = lend_comp.groupby(by = ["i_ID"]).sum(numeric_only = True)
del lend_comp["friend_ij"], lend_comp["family_ij"]




# Join the compositional data with the dataframe for the individual villagers
all_villager_nominations = pd.merge(
    how = "left",
    left = survey_responses, 
    right = lend_comp, 
    # left_on = "ID", 
    # right_on = "ID"
    left_index = True,
    right_index = True,
    suffixes = ("_villagers", "_comp"),
    indicator = False
)
all_villager_nominations = all_villager_nominations.astype(
    dtype = {
        "lender_ij": "int64",
        "friend_lender_ij": "int64",
        "family_lender_ij": "int64",
        "friend_family_lender_ij": "int64",
        "stranger_lender_ij": "int64",
        "friend_lender_ij_lender_ji": "int64",
        "family_lender_ij_lender_ji": "int64",
        "friend_family_lender_ij_lender_ji": "int64",
        "lender_ij_lender_ji": "int64"
    }
)



# This analysis focuses on the composition of each villager's "basket" of lenders.
# Thus, the number of lenders of each type should sum to a villager's total number of lenders.
assert all(
    all_villager_nominations["lender_ij"]  == all_villager_nominations[
        [
            "friend_lender_ij", 
            "family_lender_ij",
            "friend_family_lender_ij",
            "stranger_lender_ij",
            "friend_lender_ij_lender_ji", 
            "family_lender_ij_lender_ji",
            "friend_family_lender_ij_lender_ji",
            "lender_ij_lender_ji"
            
         ]
    ].sum(axis = 1)
), "Mismatch on sum of lenders!"




# Basic information about the dataframe of ordered dyads
print(all_village_dyads.info(), "\n\n")

# Basic information about the dataframe of villager nominations
print(all_villager_nominations.info(), "\n\n")

# Frequency of lenders of various types across the villagers
print(
    all_villager_nominations[
        [
            "friend_lender_ij", 
            "family_lender_ij",
            "friend_family_lender_ij",
            "stranger_lender_ij",
            "friend_lender_ij_lender_ji", 
            "family_lender_ij_lender_ji",
            "friend_family_lender_ij_lender_ji",
            "lender_ij_lender_ji"
            
         ]
    ].apply(axis = 0, func = lambda col: col.value_counts()).fillna(0)
)




# Drop the 488 villagers who nominate zero lenders from the analysis.
# 2,696 villagers remain
print((all_villager_nominations["lender_ij"] > 0).value_counts(), "\n\n")
all_villager_nominations_zeros = all_villager_nominations
all_villager_nominations = all_villager_nominations[all_villager_nominations["lender_ij"] > 0]


# Drop the 137 villagers who have missing values for the variable "female"
# 2,559 villagers remain
print(all_villager_nominations["female"].isna().value_counts(), "\n\n")
all_villager_nominations = all_villager_nominations[~all_villager_nominations["female"].isna()]


# The 2,559 Villagers nominate 6,052 Preferred Money Lenders
all_villager_nominations["lender_ij"].sum()

# Num. villagers in each village.
group_sizes = all_villager_nominations["village_ID"].value_counts().to_dict()
