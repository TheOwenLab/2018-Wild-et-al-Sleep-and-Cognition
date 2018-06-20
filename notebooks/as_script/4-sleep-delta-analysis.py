
# coding: utf-8

# # Part 4 - "Sleep Delta" Analysis

# - "Sleep Delta" is the difference between the previous night's sleep and the reported average amount of sleep. Therefore, a positive delta indicates getting more sleep than usual, whereas a negative delta indicates get less sleep than usual.
# - Is a change from the regular amount of sleep associated with cognitive performance?

# In[1]:


# Import all required Python modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys
sys.path.insert(0, '../lib')
import sleep_study_utils as ss

get_ipython().run_line_magic('matplotlib', 'inline')
idx = pd.IndexSlice


# In[2]:


# List of all columns in the data frame that have scores
score_columns = ss.score_columns() + [score+"_score" for score in ss.COMPOSITE_SCORE_NAMES]
domain_names  = [domain+"_score" for domain in ss.COMPOSITE_SCORE_NAMES[0:-1]]


# In[3]:


# Load the final data sample (saved in Part 1)
data = pd.read_pickle('../data/final_sample.pickle.bz2')


# In[4]:


# Calculate the effective number of scores being tested. It's not the
# exact number of scores (16) because there is some correlation among
# these measurements (Nyholt 2004)
eff_num_scores = ss.effective_number_of_comparisons(data[score_columns].values)
print("Effective Number of Comparisons: %.03f"%eff_num_scores)

alpha = 0.05
eff_alpha = alpha/eff_num_scores
print("Effective alpha: %.05f"%eff_alpha)


# In[5]:


# Calculate the sleep delta
data.loc[:,'sleep_delta'] = data['prev_night_sleep_duration'] - data['typical_sleep_duration']


# In[6]:


sns.distplot(data['sleep_delta'], bins=20)


# In[7]:


data['sleep_delta'].describe()


# In[8]:


# Shift (offset) continuous predictor variables so that regression 
# intercepts are interpretable. (mean centre variables)
age_offset   = data.loc[:,'age_at_test'].mean()
sleep_offset = data.loc[:,'typical_sleep_duration'].mean()

data.loc[:,'age_at_test'] -= age_offset
data.loc[:,'typical_sleep_duration'] -= sleep_offset
data.loc[:,'prev_night_sleep_duration'] -= sleep_offset


# In[9]:


age_regressors   = set(['age_at_test'])
sleep_regressors = set(['np.power(typical_sleep_duration, 2)', 'typical_sleep_duration'])
other_covariates = set(['gender', 'education', 'anxiety', 'depression'])
age_by_sleep     = ss.build_interaction_terms(age_regressors, sleep_regressors)

# Sleep delta will be modelled with a quadratic term, too.
delta_regressors  = set(['np.power(sleep_delta, 2)', 'sleep_delta'])
delta_by_duration = ss.build_interaction_terms(sleep_regressors, delta_regressors)
delta_by_age      = ss.build_interaction_terms(age_regressors, delta_regressors)
full_delta_model  = age_regressors | sleep_regressors | delta_regressors | delta_by_duration | other_covariates
full_delta_model


# In[10]:


import importlib
importlib.reload(ss)


# In[11]:


# Likelihood Ratio (LR) Tests for different effects
model_comparisons = [
    {'name':'Overall Sleep Delta',
     'h0':ss.build_model_expression(full_delta_model - delta_regressors),
     'h1':ss.build_model_expression(full_delta_model)},
    
    {'name':'Sleep Delta Quadratic?',
     'h0':ss.build_model_expression(full_delta_model - set(['np.power(sleep_delta, 2)'])),
     'h1':ss.build_model_expression(full_delta_model)},
    
    {'name':'Sleep Delta X Duration',
     'h0':ss.build_model_expression(full_delta_model - delta_by_duration),
     'h1':ss.build_model_expression(full_delta_model)},
]

delta_results = ss.compare_models(model_comparisons, data, score_columns[-4:], n_comparisons=eff_num_scores)

figS5a = ss.create_stats_figure(delta_results, 'LR', 'p_adj')
figS5b = ss.create_bayes_factors_figure(delta_results)

figS5a.savefig('../images/FigureS5a.pdf', format='pdf')
figS5b.savefig('../images/FigureS5b.pdf', format='pdf')

pd.set_option('precision', 3)
delta_results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:].to_excel('../CSVs/Table3.xlsx')
delta_results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:]


# In[12]:


# Build a table of estimated parameters (for effects of interest)
# Includes the coefficient value, standard error of the estimate,
# t value, p value, and 95% confidence intervals.
delta_models       = [smf.ols(ss.build_model_expression(full_delta_model)%score, data=data).fit() for score in score_columns[-4:]]
parameters_to_show = list(sleep_regressors)+ list(delta_regressors) + list(delta_by_duration)
parameter_index    = pd.MultiIndex.from_product([parameters_to_show, ss.COMPOSITE_SCORE_NAMES],
                                                names=['parameter', 'score'])
parameter_table    = pd.DataFrame(index=parameter_index, columns=delta_models[-1].summary().tables[1].data[0][1:])
for index, score in enumerate(ss.COMPOSITE_SCORE_NAMES):
    all_parameter_info = np.array(delta_models[index].summary().tables[1].data)
    all_parameter_info = pd.DataFrame(data=all_parameter_info[1:,1:],
                                      index=all_parameter_info[1:,0],
                                      columns=all_parameter_info[0,1:])
    parameter_table.loc[idx[:,score],:] = all_parameter_info.loc[parameters_to_show,:].values
parameter_table.to_excel('../CSVs/TableS10.xlsx')
parameter_table


# In[13]:


# Re-estimate the models, not including the interaction between sleep delta and typical duration
# Use these to plot the marginal effect of sleep delta, and calculate the peak locations.
estimated_delta_models = [smf.ols(ss.build_model_expression(full_delta_model-delta_by_duration)%score, data=data).fit() for score in score_columns[-4:]]

fig5, axs = plt.subplots(figsize=(8.7,3.2), nrows=1, ncols=2, sharey=True)
plt_score = {'STM':False, 'Reasoning':True, 'Verbal':False, 'Overall':True}
plt_index = 0
for score_index, score_model in enumerate(estimated_delta_models):
    score_color = ss.COMPOSITE_SCORE_COLORS[score_index]
    score_name  = ss.COMPOSITE_SCORE_NAMES[score_index]
    if plt_score[score_name]:
        prediction = ss.get_prediction_and_confidence(score_model, 'sleep_delta').summary_frame()
        vertex     = ss.calculate_parabola_vertex(score_model, 'sleep_delta')
        vertex_CI  = ss.fieller_ci(score_model, 'sleep_delta')
        x_points   = prediction.index
        axs[plt_index].axvline(x=vertex.x, c=score_color)
        axs[plt_index].axvspan(vertex_CI[0], vertex_CI[1], color=score_color, alpha=0.2)
        axs[plt_index].fill_between(x_points, prediction['mean_ci_lower'], prediction['mean_ci_upper'], alpha=0.3, color=score_color)
        axs[plt_index].plot(x_points, prediction['mean'], linewidth=3, color=score_color, label=score_name)
        axs[plt_index].plot(vertex.x, vertex.y, marker='o', c=score_color)
        axs[plt_index].set_xlim([-6,6])
        axs[plt_index].set_ylim([-1,0.2])
        axs[plt_index].set_xlabel('Sleep Delta (hours)')
        axs[plt_index].legend(loc='lower left')
        print('%20s peak: %.02f (%.03f-%.03f)'%(score_name, vertex.x, vertex_CI[0], vertex_CI[1]))
        plt_index += 1
axs[0].set_ylabel('Score (SDs)')
plt.tight_layout()
fig5.savefig('../images/Figure5.pdf', format='pdf')


# In[14]:


# Sleep_offset is the sample mean for typical_sleep_duration
print("Average typical sleep duration = %.03f hours"%sleep_offset)
print("%.03f hours + %.03f hours = %.03f hours"%(sleep_offset, vertex.x, sleep_offset+vertex.x))


# ### For interest, let's break apart the Sleep Delta X Typical Sleep Duration Interaction:

# In[15]:


estimated_delta_models = [smf.ols(ss.build_model_expression(full_delta_model)%score, data=data).fit() for score in score_columns[-4:]]
fig, axs = plt.subplots(figsize=(10,4), nrows=1, ncols=2, sharey=True)
sleep_durations = [ None,                  # STM
                    [7.16,4,3]-sleep_offset,   # Reasoning
                    None,                  # Verbal
                    [7.33,4,3]-sleep_offset ] # Overall
shading_steps   = np.array([1.1,0.75,0.7])
linestyle_steps = ['-','--','-.']
plt_index       = 0
for score_index, score_model in enumerate(estimated_delta_models):
    durations = sleep_durations[score_index]
    if durations is not None:
        score_color = ss.COMPOSITE_SCORE_COLORS[score_index]
        score_name  = ss.COMPOSITE_SCORE_NAMES[score_index].capitalize()

        for duration_index, duration in enumerate(durations):
            score_color = list(np.array(score_color)*shading_steps[duration_index])
            prediction = ss.get_prediction_and_confidence(score_model, 'sleep_delta', constants={'typical_sleep_duration':duration}).summary_frame()
            x_points   = np.array(prediction.index)
            show_x     = (duration+sleep_offset+x_points) > 0
            vertex     = ss.calculate_parabola_vertex(score_model, 'sleep_delta', constants={'typical_sleep_duration':duration})
            
            axs[plt_index].fill_between(x_points[show_x], 
                                        prediction['mean_ci_lower'][show_x], prediction['mean_ci_upper'][show_x],
                                        alpha=0.3, color=score_color)
            axs[plt_index].plot(x_points[show_x], prediction['mean'][show_x], 
                                linewidth=3, color=score_color, 
                                linestyle=linestyle_steps[duration_index],
                                label='%.01f hours'%(duration+sleep_offset))
            axs[plt_index].axvline(x=vertex.x, c=score_color, linestyle=linestyle_steps[duration_index])
            axs[plt_index].plot(vertex.x, vertex.y, marker='o', c=score_color)
        
        axs[plt_index].set_xlim([-6,6])
        axs[plt_index].set_ylim([-1,0.2])
        axs[plt_index].set_xlabel('Sleep Delta (hours)')
        axs[plt_index].set_title(score_name)
        axs[plt_index].legend(title='Typical Sleep Duration', loc='lower left')
        plt_index += 1
axs[0].set_ylabel('Score (SDs)')
plt.tight_layout()

