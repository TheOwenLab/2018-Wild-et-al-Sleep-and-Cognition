
# coding: utf-8

# # Part 3 - Repeated Measures Analysis

# - Does the relationship between sleep duration and cognitive performance differ between domains?

# In[1]:


# Import all required Python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
domain_names  = [domain+"_score" for domain in ss.FACTOR_NAMES]


# In[3]:


# Load the final data sample (saved in Part 1)
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


# Create a stacked dataframe, where every subject has three rows (one per domain)
stacked_data = pd.DataFrame(columns = ss.QUESTIONNAIRE_ITEMS+['score'])
sample_data  = data.sample(frac=1.0, replace=False)
for score in domain_names:
    sample_data['score_type'] = score[:-6]
    score_data = sample_data[ss.QUESTIONNAIRE_ITEMS+[score,'score_type']].set_index('score_type', append=True).copy()
    score_data.rename(columns={score:'score'}, inplace=True)
    score_data['score_type'] = score_data.index.get_level_values('score_type') 
    score_data['subject'] = sample_data.index
    stacked_data = pd.concat([stacked_data, score_data])
stacked_data['score_type'] = stacked_data['score_type'].astype('category')
stacked_data['subject'] = stacked_data['subject'].astype('category')
stacked_data.head()


# In[6]:


stacked_data.shape


# In[7]:


# Build all the  terms that will go into the fixed-effects regression formula
# Include all interactions with cognitive domain, given that Hampshire et al. (2012)
# show that these factors affect the domains differently.
age_regressors   = set(['age_at_test'])
sleep_regressors = set(['np.power(typical_sleep_duration, 2)', 'typical_sleep_duration'])
other_covariates = set(['gender', 'education', 'anxiety', 'depression'])
domain_regressor = set(['score_type'])
age_by_sleep     = ss.build_interaction_terms(age_regressors, sleep_regressors)
sleep_by_domain  = ss.build_interaction_terms(sleep_regressors, domain_regressor)
age_by_domain    = ss.build_interaction_terms(age_regressors, domain_regressor)
other_by_domain  = ss.build_interaction_terms(other_covariates, domain_regressor)
mixed_fx_factors = age_regressors | sleep_regressors | domain_regressor | sleep_by_domain | age_by_domain | age_by_sleep | other_covariates | other_by_domain


# In[8]:


# Fit the model, using a random-intercepts model grouping by subject
# That is, each subject gets their own intercept.
mixed_fx_model   = smf.mixedlm(ss.build_model_expression(mixed_fx_factors)%'score', stacked_data, groups=stacked_data["subject"] )
mixed_fx_result  = mixed_fx_model.fit(reml=False)


# In[9]:


mixed_fx_result.summary()


# In[10]:


# Test the overall interaction
mixed_fx_result.f_test("""
    np.power(typical_sleep_duration, 2):score_type[T.STM] = 
    np.power(typical_sleep_duration, 2):score_type[T.Verbal] = 
    typical_sleep_duration:score_type[T.STM] = 
    typical_sleep_duration:score_type[T.Verbal] = 0
""")


# In[11]:


# Contrast STM to Reasoning (baseline)
mixed_fx_result.f_test("""
    np.power(typical_sleep_duration, 2):score_type[T.STM] = 
    typical_sleep_duration:score_type[T.STM] = 0
""")


# In[12]:


# Contrast Verbal to Reasoning (baseline)
mixed_fx_result.f_test("""
    np.power(typical_sleep_duration, 2):score_type[T.Verbal] = 
    typical_sleep_duration:score_type[T.Verbal] = 0
""")


# In[13]:


# Contrast Verbal and STM
mixed_fx_result.f_test("""
    np.power(typical_sleep_duration, 2):score_type[T.STM] - np.power(typical_sleep_duration, 2):score_type[T.Verbal] = 
    typical_sleep_duration:score_type[T.STM] - typical_sleep_duration:score_type[T.Verbal] = 0
""")

