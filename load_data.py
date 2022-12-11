#!/usr/bin/env python
# https://stackoverflow.com/a/2429517\


import os
os.chdir("/Users/cohen/Desktop IconFree/GitHub/money")


import numpy as np
import pandas as pd ## https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html

pd.set_option("display.expand_frame_repr", True)



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
# religion: 1 == Catholic (dominant local religion); 0 == Other religion
# ethnicity: 1 == Lugbara (dominant local ethnicity); 0 == Other ethnic group


# Primary Attribute Data
survey_responses_AJPS = pd.read_csv(
    filepath_or_buffer = "nodes.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
) 

# Retain only the columns we need
survey_responses_AJPS = survey_responses_AJPS.loc[:, [ "villageId", "i",
                                                        "female", "age",
                                                        "income", "hasPhone",
                                                        "leader"
                                                        ]
] 

survey_responses_AJPS = survey_responses_AJPS.rename(
    columns = {"i": "ID", "villageId": "village_ID"}
)

# Note, in Pandas type == object == mixed types. For details, see:
# https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
# https://stackoverflow.com/a/28648923
# Also, there are missing data for some of the categorial features/variables. 
# Thus, dtype == "float64" will accomodate "NaN", Numpy style.
# However, dtype == "Int64" (note capital "I") will support integer NaN in Pandas.
# https://wesmckinney.com/book/data-cleaning.html#pandas-ext-types
survey_responses_AJPS["ID"] = survey_responses_AJPS["ID"].astype("string")
survey_responses_AJPS["village_ID"] = survey_responses_AJPS["village_ID"].astype("string")
survey_responses_AJPS["female"] = survey_responses_AJPS["female"].astype("Int64")
survey_responses_AJPS["age"] = survey_responses_AJPS["age"].astype("Int64")
survey_responses_AJPS["income"] = survey_responses_AJPS["income"].astype("Int64") # Ordinal
survey_responses_AJPS["hasPhone"] = survey_responses_AJPS["hasPhone"].astype("Int64")
survey_responses_AJPS["leader"] = survey_responses_AJPS["leader"].astype("Int64")

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
    columns = {"i": "ID", "villageId": "village_ID", "villageTxt": "village_ID_txt",
                "hh": "HH_ID", "head": "HH_Head"}
)

survey_responses_CPS["ID"] = survey_responses_CPS["ID"].astype("string")
survey_responses_CPS["village_ID"] = survey_responses_CPS["village_ID"].astype("string")
survey_responses_CPS["village_ID_txt"] = survey_responses_CPS["village_ID_txt"].astype("string")
survey_responses_CPS["HH_ID"] = survey_responses_CPS["HH_ID"].astype("string")
survey_responses_CPS["edu_full"] = survey_responses_CPS["edu_full"].astype("string")
survey_responses_CPS["HH_Head"] = survey_responses_CPS["HH_Head"].astype("Int64")
survey_responses_CPS["rlg"] = survey_responses_CPS["rlg"].astype("Int64")
survey_responses_CPS["eth"] = survey_responses_CPS["eth"].astype("Int64")

survey_responses_CPS = survey_responses_CPS.set_index("ID", drop = True)


# Merge/Join (One-to-One) Primary and Secondary Attribute Data
# Merge or join operations combine datasets by linking rows using one or more keys.
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
survey_responses = survey_responses.rename(columns = {"village_ID_AJPS": "village_ID"})


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

survey_responses["edu_full"] = survey_responses["edu_full"].astype("Int64")


# Basic information on the monadic data
survey_responses
survey_responses.info()
survey_responses.describe()




# Load social network data
nominations = pd.read_csv(
    filepath_or_buffer = "ties.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
)

nominations = nominations.rename(
    columns = {"villageId": "village_ID", "i": "i_ID", "j": "j_ID"}
)

nominations["village_ID"] = nominations["village_ID"].astype("string")
nominations["i_ID"] = nominations["i_ID"].astype("string")
nominations["j_ID"] = nominations["j_ID"].astype("string")
nominations["type"] = nominations["type"].astype("string")
nominations["dist"] = nominations["dist"].astype("float64")

nominations["ij_ID"] = nominations['i_ID'] + "_" + nominations['j_ID']

nominations = nominations.set_index("ij_ID", drop = True)


# The types of relationships measured for each network survey.
relationship_type = pd.unique(nominations["type"])





# Load Village-Level Data
# TODO: Variable Descriptions Variables with the "census_" prefix come from the 2014 Uganda National Population and Housing Census
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

villages = villages.rename(columns = {"villageId": "village_ID"})

# Construct indicate for whether or not a village has a market of any kind
# https://stackoverflow.com/a/69440643
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html
# Note, "~villages["pg_market_any"].isna()" is read as "element is NOT missing"
villages["pg_market_any"] = villages["pg_market"] + villages["pg_market_crops"]
villages["pg_market_any"] = villages["pg_market_any"].mask(
    cond = ~villages["pg_market_any"].isna(),
    other = villages["pg_market_any"] > 0
)
del villages["pg_market"], villages["pg_market_crops"]


villages["village_ID"] = villages["village_ID"].astype("string")
villages["pg_market_any"] = villages["pg_market_any"].astype("float64")


# Retain data only for the 16 villages in which network data were collected.
# https://stackoverflow.com/questions/42268549/membership-test-in-pandas-data-frame-column
village_IDs = pd.unique(survey_responses["village_ID"])
villages = villages[villages["village_ID"].isin(village_IDs)]
# villages.loc[villages["village_ID"].apply(func = lambda element: element in village_IDs), :]




# TODO: Impute Monadic Attribute Data
# TODO: Determine if the categorical data need qualitative labels for cmdstanpy.



