#!/usr/bin/env python
# https://stackoverflow.com/a/2429517/






# Load Primary and Secondary Attribute Data
# These data contain monadic (individual-level) attribute data for each
# villager in each of the 16 villages wherein network data were collected.
# The variables/features used for the analysis are as follows:
# female: 1 == Female; 0 == Male
# hh: The ID of each villager's household




# Primary Attribute Data
survey_responses_AJPS = pd.read_csv(
    filepath_or_buffer = "nodes.csv",
    header = 0,
    keep_default_na = True,
    # nrows = 10
) 

survey_responses_AJPS = survey_responses_AJPS.loc[:, [ "villageId", "i", "female" ]] 

survey_responses_AJPS = survey_responses_AJPS.rename(
    columns = {"i": "ID", "villageId": "village_ID"}
)

# Cast data in the columns as the appropriate type and reassign.
# Note, there are missing data for some of the features/variables. 
# Thus, dtype == "float64" will accommodate "NaN", Numpy style.
# However, dtype == "Int64" (note capital "I") will support NaN integer in Pandas.
# https://wesmckinney.com/book/data-cleaning.html#pandas-ext-types
survey_responses_AJPS = survey_responses_AJPS.astype(
    dtype = {
        "ID": "string",
        "village_ID": "string",
        "female": "Int64"
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

survey_responses_CPS = survey_responses_CPS.loc[:, [ "villageId", "i","villageTxt", "hh"]] 

survey_responses_CPS = survey_responses_CPS.rename(
    columns = {
        "i": "ID", 
        "villageId": "village_ID",
        "villageTxt": "village_ID_txt",
        "hh": "HH_ID"
     }
)

survey_responses_CPS = survey_responses_CPS.astype(
    dtype = {
        "ID": "string",
        "village_ID": "string",
        "village_ID_txt": "string",
        "HH_ID": "string"
    }
)

survey_responses_CPS = survey_responses_CPS.set_index("ID", drop = True)




# Merge/join (One-to-One) Primary Attribute Data and Secondary Attribute Data
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


# Duplicate columns appear in the merged dataframe when columns in
# the left/right dataframe have identical names. 
del survey_responses["village_ID_CPS"]
survey_responses = survey_responses.rename(columns = {"village_ID_AJPS": "village_ID"})




# Load social network data
# These network data are arranged in a "long" or "edge list" format 
# (i.e., each row of the data frame is an ordered dyad) and they
# detail information on one of six relationships between residents
# living in the same village plus within-village, pairwise geographic distance.
# Thus, villager i/i_ID is the sending/nominating actor and villager j/j_ID is 
# the receiving/nominated actor. The relationship types are as follows: 
# family: i named j as a member of their family.
# lender: i named j as someone who would lend them money.
# friend: i named j as a friend.
# solver: i named j as someone she would go to to solve a problem.
# Contgame: i voted for j to receive the village's money in a public goods game.
# speak: i spoke to j about UBridge (i.e., a government reporting platform)
# geo: geographic distance between i and j in meters (NA for all i, j not in same village).

# The only relationships analysed for my study are: lender, family, and friend.

# Note, rows for the different relationships are stacked such that
# an ordered dyad can appear multiple times if i/i_ID nominates j/j_ID
# in response to more than one name generator + the dyad for geographic distance. 
# To see this, RUN: nominations.loc[nominations.index[nominations.index.duplicated()]]
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

nominations = nominations[nominations["type"].isin(["lender", "family", "friend"])]
del nominations["dist"]

nominations = nominations.astype(
    dtype = {
        "village_ID": "string",
        "i_ID": "string",
        "j_ID": "string",
        "type": "string"
    }
)

# Create new column for the unique id of each directed relationship
nominations["ij_ID"] = nominations['i_ID'] + "_" + nominations['j_ID']
nominations = nominations.set_index("ij_ID", drop = True)




# Save the types of relationships use for this analysis.
relationship_type = pd.unique(nominations["type"])


# Set aside list of the IDs of each village in the study for use later.
village_IDs = pd.unique(survey_responses["village_ID"])




# Basic information on the monadic data
print(survey_responses, "\n")
print(survey_responses.info())
# survey_responses.describe(),
