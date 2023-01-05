# The Relational Bases of Informal Financial Cooperation <br> (Simpson, In Prep.)


## Abstract
Access to money is vital to day-to-day human survival. But who do we turn to for the cash we need when formal avenues are unavailable, undesirable, or malicious? The evolutionary theory of kin selection and the network theory of multiplexity (i.e., link superimposition) suggest that financial aid will freely flow from friends and family. Yet, economic sociologists theorise that finance is highly context-specific such that friendship and kinship need not systematically mix with money. Using Bayesian Dirichlet-Multinomial models and data from 2,559 adults in 16 Ugandan villages, here I show that roughly 60% of preferred money lenders are friends and/or kin — where non-friend kin outrank non-kin friends and friends-who-are-kin, with the least-abundant patrons being intimates for whom one is also a source of cash (i.e., reciprocal aid). Around 40% of lenders are estimated to be non-kin-who-are-not-friends. Still, models contravene the sociological notion that money is not fungible across relational scenarios as they corroborate diverse combinations of friendship and kinship with lending — possibly due to a lack of other viable strategies for securing informal finance amongst the poor. 

<br>

![](https://github.com/cohensimpson/money/blob/main/F1_Proportions_Lender_Types_Parallel_Coordinates.svg) 

<br> 

**Figure 1.** Parallel-coordinate style plot (common y-axis) of posterior means for $\pi_{g,k}, \ldots, \pi_{g,K}$ — i.e., the proportion of money lenders of various types amongst 2,559 Ugandan villagers in 16 villages. Posterior means proportions (bullets) are overplotted and connected across the $K = 8$ types of lenders based on model specification. The red line joins the simplex $\vec{\pi} = \pi_{g,k}, \ldots, \pi_{g,K}$ from the baseline model, each orange lines joins a village-specific simplex $\vec{\pi}\_{g}$ from the extended model, and each of the 32 blue and yellow lines join a village- and sex-specific simplex $\vec{\pi}\_{g, \text{Male}}$ or $\vec{\pi}\_{g, \text{Female}}$ from the sex-specific model, where there are 16 lines for males and 16 for females. Posterior means are derived using three Dirichlet-Multinomial models (12,000 posterior draws). Vertical black lines indicate the 95% highest density interval for each proportion (i.e., each bullet) and are also overplotted. Note, values are plotted using a square-root scale for the y-axis. Thus, vertical changes are not constant. Overplotting is used to emphasise agreement across models.

<br>
<br> 

![](https://github.com/cohensimpson/money/blob/main/F2_Posterior_Predictive_Checks_Count_Frequencies.svg) 

<br> 

**Figure 2.** Small multiple of posterior predictive checks for the three Dirichlet-Multinomial models in Fig. 1. Model fit is assessed by comparing the observed frequency of counts (0 to 5) of various types of money lenders across the entire $N \times K$ observed matrix of compositional count vectors $y^{\text{Observed}}$ to the posterior mean frequency of counts across 12,000 fake versions of theses matrices simulated under each fitted model (i.e., one matrix $y^{\text{Synthetic}}$ for each sample from the posterior predictive distribution). For each posterior mean frequency, shaded regions indicate the range of values that are $\pm 1.96$ times the standard deviation of a given frequency across the 12,000 synthetic count matrices. Frequencies of counts, which range from zero to roughly 3,000, are plotted using a $\text{log}\_{10}$ scale for the y-axis. Thus, modest differences are exaggerated at lower frequencies and vertical changes are not constant, where each major axis tick represents a value ten-times larger than the major tick immediately prior. In general, models fit the data well, although each model is overoptimistic about the frequency that villagers have more than one indebted lender of various types.

<br>
<br> 


## Python Code
Here, you will find five Python scripts in addition to three ".csv" data files. Throughout the scripts, you will find code to carry out the analyses reported in my paper alongside commands used to produce useful print out (e.g., descriptive statistics) and comments that (hopefully) give you insight into the thinking behind the decisions I take.

**_After_** you have placed the data files and all Python scripts in the same working directory/folder, installed the necessary Python modules (see "main.py" for a list of all necessary modules), and set the number of available computing cores for your machine (see "main.py"), you should be able to simply to run "main.py", which executes the other scripts, to redo my analyses. This will also carry out all of the goodness-of-fit tests and generate the figures in the paper using [Plotnine](https://plotnine.readthedocs.io/en/stable/) — a wonderful Python implementation of [R's "ggplot" library](https://ggplot2.tidyverse.org).

Finally, when re-running my analyses, some numerical results may differ slightly from those reported in the paper due to stochastic perturbations. I have used the same random seed (20200127) to ensure exact reproducibility wherever possible. However, this is not always an option depending on the function. Also, note that models are fitting in a Bayesian framework using [PyMC — i.e., Python-based probabilistic-programming language](https://www.pymc.io/welcome.html). Accordingly, changing the number of CPU cores, which currently also controls the number of Markov chains for each model, could lead to somewhat different results.  


## Summary of Key Files in Repository
 1) main.py (Script for Loading Modules and Executing Other Parts of the Analysis)
 
 2) load_data.py (Script for Loading and Filtering Data)
 
 3) build_features.py (Script for the Transformation of Data for Model Fitting)
 
 4) fit_models.py (Script for Writing + Estimating Dirichlet-Multinomial Models with PyMC)
 
 5) visualise_results.py (Script for Visualisation of Results with Plotnine)
 
 6) nodes.csv (Monadic covariates for the individual villagers collected by Ferrali et al. (2020)[2] for their paper in the _American Journal of Political Science_ [2]) 

 7) nodes_CPS_Version_1.2.csv (Dataset of monadic covariates that includes household membership which was used by Ferrali et al. (2021)[3] for their paper in _Comparative Political Studies_) 

 8) ties.csv (Sociometric data on lending, kinship, and friendship for the individual villagers collected by Ferrali et al. [2]) 

 9) Ferrali_et_al_2020.pdf [2] 
 
 10) Ferrali_et_al_2021.pdf [3]
 
 11) dataverse_files_Ferrali_et_al_AJPS_Version_1 (2019-08-07).zip (Ferrali et al.'s [1] [original replication materials](https://doi.org/10.7910/DVN/NOYBCQ))
 
 12) dataverse_files_Ferrali_et_al_CPS_Version_1.2 (2021-06-01).zip (Ferrali et al.'s [2] [original replication materials](https://doi.org/10.7910/DVN/YEFRPC))
 
 
## Key Citations for Replicators
[1] Simpson, C.R. In Prep. "The Relational Bases of Informal Financial Cooperation". Working Paper.

[2] Ferrali, R., Grossman, G., Platas, M. R., & Rodden, J. (2020). It Takes a Village: Peer Effects and Externalities in Technology Adoption. _American Journal of Political Science_, 64(3), 536–553. https://doi.org/10.1111/ajps.12471

[3] Ferrali, R., Grossman, G., Platas, M. R., & Rodden, J. (2021). Who Registers? Village Networks, Household Dynamics, and Voter Registration in Rural Uganda. _Comparative Political Studies_, 001041402110360. https://doi.org/10.1177/00104140211036048


## Notes
1) Thank you for your interest in my work! Please do let me know if something goes wrong. I am more than happy to help and you can always email me.

2) Some steps in "visualise_results.py" are slow, namely the calls to the custom/user-defined function "prepare_pymc_trace_for_ppc()" which can take a few minutes per call on my machine (Apple M1 Max; 32GB Ram). I creating this function to help transform the posterior predictive samples into Pandas dataframes that are suitable for visualisation with Plotnine. And I am still exploring how to speed up the function given the information needed and the large dataframe of posterior samples (n.b., the response variable is 8-dimensional, leading to quite large Pandas dataframes).

