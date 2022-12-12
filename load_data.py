#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/


import os
os.chdir("/Users/cohen/Desktop IconFree/GitHub/money")


import numpy as np
import pandas as pd 




# Load Primary and Secondary Attribute Data
# These data contain each monadic (individual-level) attribute data for each
# villager in each of the 16 villages wherein network data were collected.
# The variables/features used for the analysis are as follows:
# female: 1 == Female; 0 == Male
# age: Respondent's age in years 
# edu_full: Level of schooling (No Schooling, Some Primary, Some Secondary, Some Tertiary)
# income: "In comparison to other typical households in this village, how would
# you describe your household’s economic situation?" Survey answers: 1 = "Much 
# worse", 2 == "Somewhat worse", 3 == "About the same", 4 == "Somewhat better", 5 == "Much better"
# hasPhone: 1 == Owns Phone; 0 == Does Not Own Phone
# leader: 1 == Respondent occupies a formal leadership position within the village
# HH_Head: 1 == the head of the household;  0 == all other household members
# religion (rlg): 1 == Catholic (dominant local religion); 0 == Other religion
# ethnicity (eth): 1 == Lugbara (dominant local ethnicity); 0 == Other ethnic group


# Primary Attribute Data
survey_responses_AJPS = pd.read_csv(
    filepath_or_buffer = "nodes.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
) 

# Retain only the columns needed
survey_responses_AJPS = survey_responses_AJPS.loc[:, [ "villageId", "i",
                                                        "female", "age",
                                                        "income", "hasPhone",
                                                        "leader"
                                                        ]
] 

survey_responses_AJPS = survey_responses_AJPS.rename(
    columns = {
        "i": "ID", 
        "villageId": "village_ID"
    }
)

# Cast data in the columns as the appropriate type and reassign.
# Note, in Pandas type == object == mixed types. For details, see:
# https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
# https://stackoverflow.com/a/28648923
# Also, there are missing data for some of the categorial features/variables. 
# Thus, dtype == "float64" will accomodate "NaN", Numpy style.
# However, dtype == "Int64" (note capital "I") will support integer NaN in Pandas.
# https://wesmckinney.com/book/data-cleaning.html#pandas-ext-types
survey_responses_AJPS = survey_responses_AJPS.astype(
    dtype = {
        "ID": "string",
        "village_ID": "string",
        "female": "Int64",
        "age": "Int64",
        "income": "Int64",
        "hasPhone": "Int64",
        "leader": "Int64"
    }
)

survey_responses_AJPS = survey_responses_AJPS.set_index("ID", drop = True)


# Secondary Attribute Data
survey_responses_CPS = pd.read_csv(
    filepath_or_buffer = "nodes_CPS_Version_1.2.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
)

survey_responses_CPS = survey_responses_CPS.loc[:, [ "villageId", "i",
                                                    "villageTxt", "hh",
                                                    "edu_full", "head",
                                                    "rlg", "eth"
                                                    ]
] 

survey_responses_CPS = survey_responses_CPS.rename(
    columns = {
        "i": "ID", 
        "villageId": "village_ID",
        "villageTxt": "village_ID_txt",
        "hh": "HH_ID",
        "head": "HH_Head"
     }
)

survey_responses_CPS = survey_responses_CPS.astype(
    dtype = {
        "ID": "string",
        "village_ID": "string",
        "village_ID_txt": "string",
        "HH_ID": "string",
        "edu_full": "string",
        "HH_Head": "Int64",
        "rlg": "Int64",
        "eth": "Int64"
    }
)

survey_responses_CPS = survey_responses_CPS.set_index("ID", drop = True)


# Merge/join (one-to-one) Primary Attribute Data and Secondary Attribute Data
# Merge/join operations combine datasets by linking rows using one or more keys.
# Inner joins produce the intersection of tables based on common keys (here, ID).
# https://wesmckinney.com/book/data-wrangling.html
survey_responses = pd.merge(
    how = "inner",
    left = survey_responses_AJPS, 
    right = survey_responses_CPS,
    # left_on = "ID", 
    # right_on = "ID"
    left_index = True,
    right_index = True,
    suffixes = ("_AJPS", "_CPS"),
    indicator = False
)
del survey_responses_AJPS, survey_responses_CPS


# Duplicate columns appear in the merged dataframe when columns in the left/right
# dataframe have identical names. Pandas does allow multiple/hierarchical row indices. 
# indices. However, this is avoided here in order to facilitate the use of calls 
# to R functions. Accordingly, just keep one of the village_ID columns.
del survey_responses["village_ID_CPS"]

survey_responses = survey_responses.rename(
    columns = {
        "village_ID_AJPS": "village_ID"
    }
)


# Recode edu_full and income:
# Now, for edu_full: 0 == "No schooling", 1 == "Some Primary",
# 2 == "Some Secondary", 3 == "Some Tertiary"
# Now, for income: -2 == 1 == "Much Worse", -1 == 2 == "Somewhat Worse",
# 0 == 3 == "About the Same", 1 == 4 == "Somewhat Better", 2 == 5 == "Much Better".
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
survey_responses = survey_responses.replace(
    to_replace = {"edu_full": {
                                "No schooling": "0",
                                "Some Primary": "1",
                                "Some Secondary": "2",
                                "Some Tertiary": "3"
                                },
                    "income": {
                                1: -2,
                                2: -1,
                                3: 0,
                                4: 1,
                                5: 2
                                }
    }
)

survey_responses = survey_responses.astype(
    dtype = {
        "edu_full": "Int64"
    }
)


# Basic information on the monadic data
print(survey_responses, "\n")
survey_responses.info()
# survey_responses.describe(),




# Load social network data
# These network data are arranged in a "long" format (i.e., each row of the data frame 
# is an ordered dyad) and they detail information on one of six relationships 
# between residents living in the same village plus within-village geographic distance.
# Thus, villager i/i_ID is the sending/nominating actor and villager j/j_ID is 
# the receiving/nominated actor. 
# The relationship types are as follows: 
# family: i named j as a member of their family.
# lender: i named j as someone who would lend them money.
# friend: i named j as a friend.
# solver: i named j as someone she would go to to solve a problem.
# Contgame: i voted for j to receive the village's money in a public goods game.
# speak: i spoke to j about UBridge (not analysed for this study)
# geo: geographic distance between i and j in meters (NA for all i, j not in same village).
nominations = pd.read_csv(
    filepath_or_buffer = "ties.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
)

nominations = nominations.rename(
    columns = {
        "villageId": "village_ID",
         "i": "i_ID", 
         "j": "j_ID"
     }
)

nominations = nominations.astype(
    dtype = {
        "village_ID": "string",
        "i_ID": "string",
        "j_ID": "string",
        "type": "string",
        "dist": "float64"
    }
)

# Create new column for the unique id of each directed relationship
nominations["ij_ID"] = nominations['i_ID'] + "_" + nominations['j_ID']
nominations = nominations.set_index("ij_ID", drop = True)


# There are 3,184 sending/nominating villagers in nominations, matching the attribute 
# data in survey_responses. However, there are 4,417 receiving/nominated villagers in
# nominations. And 1,233 of these individuals do not appear in survey_responses.
nominations["i_ID"].unique()
nominations["j_ID"].unique()

nominations["i_ID"][~nominations["i_ID"].isin(survey_responses.index)].unique()
nominations["j_ID"][~nominations["j_ID"].isin(survey_responses.index)].unique()


# Save the types of relationships measured for each network survey.
relationship_type = pd.unique(nominations["type"])





# Load Village-Level Data
# These data contain each monadic (individual-level) attribute data for each
# the Ugandan villages, 16 of which featured network data collection.
# Variables with "census_" come from the 2014 Uganda National Population
# and Housing Census. Variables  with "pg_" come from a baseline survey of 
# public goods conducted in 2014. 
# The variables/features used for the analysis are as follows:
# census_pop: village population size
# census_employ: village employment rate
# census_noAgri: percent of villagers employed in non-agricultural work
# pg_savingsgroup: number of village savings groups
# pg_market:  general market in the village (1 = Yes, 0 = No)
# pg_market_crops: marketplace for crops in the village (1 = Yes, 0 = No)
# census_employ: village distance from Arua (nearest economic hub)


villages = pd.read_csv(
    filepath_or_buffer = "villages.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
)

villages = villages.loc[:, [ "villageId", "census_pop", "census_employ",
                            "census_noAgri", "pg_savingsgroup", "pg_market",
                            "pg_market_crops", "distArua"
                            ]
] 

villages = villages.rename(
    columns = {
        "villageId": "village_ID"
    }
)


# Construct indicator for how many markets for goods and vegetables a village has
villages["pg_market_any"] = villages["pg_market"] + villages["pg_market_crops"]


# Construct indicator for whether or not a village has a market of any kind
# https://stackoverflow.com/a/69440643
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html
# Note, "~villages["pg_market_any"].isna()" is read as "element is NOT missing"
villages.loc[:, ["pg_market_any"]] = villages.loc[:, ["pg_market_any"]].mask(
    cond = ~villages.loc[:, ["pg_market_any"]].isna(),
    other = villages.loc[:, ["pg_market_any"]] > 0
)
del villages["pg_market"], villages["pg_market_crops"]


# Retain data only for the 16 villages in which network data were collected.
# https://stackoverflow.com/questions/42268549/membership-test-in-pandas-data-frame-column
village_IDs = pd.unique(survey_responses["village_ID"])
villages = villages[villages["village_ID"].isin(village_IDs)]
# villages.loc[villages["village_ID"].apply(func = lambda element: element in village_IDs), :]


villages = villages.astype(
    dtype = {
        "village_ID": "string",
        "pg_savingsgroup": "Int64",
        "pg_market_any": "Int64"
    }
)
