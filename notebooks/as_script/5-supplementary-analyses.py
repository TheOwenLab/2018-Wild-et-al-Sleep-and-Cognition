
# coding: utf-8

# # Part 5 - Supplementary Analyses

# In[1]:


# Import all required Python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as smsq
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats, linalg

import sys
sys.path.insert(0, '../lib')
import sleep_study_utils as ss

get_ipython().run_line_magic('matplotlib', 'inline')
idx = pd.IndexSlice


# In[2]:


# List of all columns in the data frame that have scores
score_columns = ss.score_columns() + [score+"_score" for score in ss.COMPOSITE_SCORE_NAMES]


# In[3]:


data = pd.read_pickle('../data/final_sample.pickle.bz2')


# In[4]:


# Shift (offset) continuous predictor variables so that regression 
# intercepts are interpretable. (mean centre variables)
age_offset   = data.loc[:,'age_at_test'].mean()
sleep_offset = data.loc[:,'typical_sleep_duration'].mean()
data.loc[:,'age_at_test'] -= age_offset
data.loc[:,'typical_sleep_duration'] -= sleep_offset
data.loc[:,'prev_night_sleep_duration'] -= sleep_offset


# In[5]:


# Re-order the score columns (mainly for visualization purposes)
score_columns = np.array(score_columns)
score_columns = score_columns[[0,4,9,11,3,5,6,8,10,1,2,7,12,13,14,15]]


# In[6]:


# Calculate the effective number of scores being tested. It's not the
# exact number of scores (16) because there is some correlation among
# these measurements.
#  - Nyholt 2004
num_scores = data[score_columns].shape[1]
score_corr = np.corrcoef(data[score_columns], rowvar=False)
eigen_vals = np.linalg.eigvals(score_corr)
eff_num_scores = 1+(num_scores-1)*(1-np.var(eigen_vals)/num_scores)
print("Effective Number of Comparisons: %.03f"%eff_num_scores)

alpha = 0.05
eff_alpha = alpha/eff_num_scores
print("Effective alpha: %.05f"%eff_alpha)


# In[7]:


# Specify the variables that will make up terms in the regression equation
age_regressors   = set(['age_at_test'])
sleep_regressors = set(['np.power(typical_sleep_duration, 2)', 'typical_sleep_duration'])
other_covariates = set(['gender', 'education', 'anxiety', 'depression'])
age_by_sleep     = ss.build_interaction_terms(age_regressors, sleep_regressors)
full_model       = age_regressors | sleep_regressors | age_by_sleep | other_covariates


# ## 1) Perform the primary analysis, but without the "tails"

# Is the U-shaped relationship driven by poor performers with very little, or way too much, sleep?

# In[8]:


# Construct a mask that removes subjects who reported sleeping more
# or less than 1.5 standard deviations from the average amount of sleep.
mean_sleep_duration = data['typical_sleep_duration'].mean()
std_sleep_duration  = data['typical_sleep_duration'].std()
not_in_the_tails    = np.abs(data['typical_sleep_duration']-mean_sleep_duration) <= 1.5 * std_sleep_duration
print('%d subjects remain after filtering sleep durations more than 1.5 SDs from the average.'%
      data[not_in_the_tails].shape[0])


# In[9]:


# Likelihood Ratio (LR) Tests for different effects
model_comparisons = [
    {'name':'Age',
     'h0':ss.build_model_expression(full_model - age_regressors),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Typical Sleep Duration',
     'h0':ss.build_model_expression(full_model - sleep_regressors),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Quadratic?',
     'h0':ss.build_model_expression(full_model - set(['np.power(typical_sleep_duration, 2)'])),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Age X Sleep Duration',
     'h0':ss.build_model_expression(full_model - age_by_sleep),
     'h1':ss.build_model_expression(full_model)},
    ]  


results = ss.compare_models(model_comparisons, data[not_in_the_tails], score_columns, n_comparisons=eff_num_scores)
ss.create_stats_figure(results, 'LR', 'p_adj')
ss.create_bayes_factors_figure(results)
pd.options.display.float_format = '{:.3f}'.format
results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:].to_excel('../CSVs/TableS6.xlsx')
results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:]


# ## 2) Use Previous Night's Sleep Instead of Typical Sleep Duration

# In[10]:


prev_night_sleep  = set(['np.power(prev_night_sleep_duration,2)', 'prev_night_sleep_duration'])
age_by_prev_sleep = ss.build_interaction_terms(age_regressors, prev_night_sleep)
prev_night_full_model = age_regressors | prev_night_sleep | age_by_prev_sleep | other_covariates


# In[11]:


# Likelihood Ratio (LR) Tests for different effects
prev_night_model_comparisons = [
    {'name':'Age',
     'h0':ss.build_model_expression(prev_night_full_model - age_regressors),
     'h1':ss.build_model_expression(prev_night_full_model)},
    
    {'name':'Prev Night''s Sleep',
     'h0':ss.build_model_expression(prev_night_full_model - prev_night_sleep),
     'h1':ss.build_model_expression(prev_night_full_model)},
    
    {'name':'Quadratic?',
     'h0':ss.build_model_expression(prev_night_full_model - set(['np.power(prev_night_sleep_duration,2)'])),
     'h1':ss.build_model_expression(prev_night_full_model)},
    
    {'name':'Age X Sleep Duration',
     'h0':ss.build_model_expression(prev_night_full_model - age_by_prev_sleep),
     'h1':ss.build_model_expression(prev_night_full_model)},
    ]  

prev_night_results = ss.compare_models(prev_night_model_comparisons, data, score_columns)
figS2a = ss.create_stats_figure(prev_night_results, 'LR', 'p_adj')
figS2b = ss.create_bayes_factors_figure(prev_night_results)
figS2a.savefig('../images/FigureS2a.pdf', format='pdf')
figS2b.savefig('../images/FigureS2b.pdf', format='pdf')
prev_night_results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:].to_excel('../CSVs/TableS7.xlsx')
prev_night_results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:]


# ## 3) Effects of Other Covariates

# - Gender (and interactions with gender)
# - levels of education, anxiety, depression

# In[12]:


gender_by_sleep   = ss.build_interaction_terms(set(['gender']), sleep_regressors)
full_model        = age_regressors | sleep_regressors | age_by_sleep | other_covariates
full_gender_model = age_regressors | sleep_regressors | age_by_sleep | other_covariates | gender_by_sleep


# In[13]:


# Likelihood Ratio (LR) Tests for different effects
model_comparisons = [
    {'name':'Gender',
     'h0':ss.build_model_expression(full_model - set(['gender'])),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Gender X Sleep Duration',
     'h0':ss.build_model_expression(full_gender_model - gender_by_sleep),
     'h1':ss.build_model_expression(full_gender_model)},
    
    {'name':'Anxiety',
     'h0':ss.build_model_expression(full_model - set(['anxiety'])),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Depression',
     'h0':ss.build_model_expression(full_model - set(['depression'])),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Education',
     'h0':ss.build_model_expression(full_model - set(['education'])),
     'h1':ss.build_model_expression(full_model)},
    ]  

results = ss.compare_models(model_comparisons, data, score_columns)
figS3a = ss.create_stats_figure(results, 'LR', 'p_adj')
figS3b = ss.create_bayes_factors_figure(results)
figS3a.savefig('../images/FigureS3a.pdf', format='pdf')
figS3b.savefig('../images/FigureS3b.pdf', format='pdf')

results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:]
results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:].to_excel('../CSVs/TableS8.xlsx')


# - Real effects of gender, frequency of anxiety-related episodes, and level of education.
# - No interaction between gender and sleep duration

# In[14]:


# Re-build and estimate regression models for the four composite scores
# These will be used to table of parameters, generate plots, etc.
estimated_models = [smf.ols(ss.build_model_expression(full_model)%score, data=data).fit() for score in score_columns[-4:]]


# In[15]:


# Plot the marginal effects of each of these effects
# i.e., the mean of each category, corrected for all other variables.
fig, axs = plt.subplots(figsize=(10,8), nrows=2, ncols=2, sharex=False, sharey=True)
for var_index, variable in enumerate(['gender', 'education', 'anxiety', 'depression']):
    categories = list(data[variable].cat.categories)
    plt_index  = np.unravel_index(var_index, axs.shape) 
    if variable == 'gender':
        categories.remove('Other')
    category_means = pd.DataFrame(index=ss.COMPOSITE_SCORE_NAMES, columns=categories)
    category_SEMs  = pd.DataFrame(index=ss.COMPOSITE_SCORE_NAMES, columns=categories)
    axs[plt_index].axhline(y=0, c='k', linewidth=1)
    for score_index, score_model in enumerate(estimated_models):
        score_name  = ss.COMPOSITE_SCORE_NAMES[score_index]
        prediction  = ss.get_prediction_and_confidence(score_model, variable, x=categories, in_place=False).summary_frame()
        category_means.loc[score_name, :] = prediction['mean']
        category_SEMs.loc[score_name, :]  = prediction['mean_se']
    category_means.plot(kind='bar', yerr=category_SEMs, ax=axs[plt_index], width=0.85)
    axs[plt_index].set_title(variable.capitalize())
    plt.setp(axs[plt_index].xaxis.get_majorticklabels(), rotation=45 )
    
plt.tight_layout()
fig.savefig('../images/FigureS4.pdf', format='pdf')


# ## 4) Does removing covariates (gender, etc.) affect the results?

# In[16]:


other_covariates


# In[17]:


reduced_model = full_model - other_covariates
print(ss.build_model_expression(reduced_model))


# In[18]:


# Likelihood Ratio (LR) Tests for different effects
model_comparisons = [
    {'name':'Age',
     'h0':ss.build_model_expression(reduced_model - age_regressors),
     'h1':ss.build_model_expression(reduced_model)},
    
    {'name':'Typical Sleep Duration',
     'h0':ss.build_model_expression(reduced_model - sleep_regressors),
     'h1':ss.build_model_expression(reduced_model)},
    
    {'name':'Quadratic?',
     'h0':ss.build_model_expression(reduced_model - set(['np.power(typical_sleep_duration, 2)'])),
     'h1':ss.build_model_expression(reduced_model)},
    
    {'name':'Age X Sleep Duration',
     'h0':ss.build_model_expression(reduced_model - age_by_sleep),
     'h1':ss.build_model_expression(reduced_model)},
    ]  

results = ss.compare_models(model_comparisons, data, score_columns)
ss.create_stats_figure(results, 'LR', 'p_adj')
ss.create_bayes_factors_figure(results)
results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:]


# ## 5) Check Whether Dependent Variables are Normally Distributed

# In[19]:


from scipy import stats
from statsmodels.compat import lzip
import statsmodels.api as sm
import statsmodels.stats.api as sms
fig, axs = plt.subplots(figsize=(19.2,10), nrows=2, ncols=len(estimated_models))
for i, model in enumerate(estimated_models):
    residuals = model.resid
    sns.distplot(residuals, ax=axs[0,i])
    stats.probplot(residuals, dist="norm", plot=axs[1,i])
    axs[0,i].set_title(model.model.endog_names)

