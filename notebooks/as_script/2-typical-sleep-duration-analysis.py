
# coding: utf-8

# # Part 2 - Statistical Analysis of Typical Sleep Duration

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


# In[3]:


# Load the final data sample (saved in Part 1)
data = pd.read_pickle('../data/final_sample.pickle.bz2')


# In[4]:


# Re-order the score columns (mainly for visualization purposes)
score_columns = np.array(score_columns)
score_columns = score_columns[[0,4,9,11,3,5,6,8,10,1,2,7,12,13,14,15]]


# In[5]:


# Calculate the effective number of scores being tested. It's not the
# exact number of scores (16) because there is some correlation among
# these measurements (Nyholt 2004)
eff_num_scores = ss.effective_number_of_comparisons(data[score_columns].values)
print("Effective Number of Comparisons: %.03f"%eff_num_scores)

alpha = 0.05
eff_alpha = alpha/eff_num_scores
print("Effective alpha: %.05f"%eff_alpha)


# In[6]:


# Shift (offset) continuous predictor variables so that regression 
# intercepts are interpretable. (mean centre variables)
age_offset   = data.loc[:,'age_at_test'].mean()
sleep_offset = data.loc[:,'typical_sleep_duration'].mean()
data.loc[:,'age_at_test'] -= age_offset
data.loc[:,'typical_sleep_duration'] -= sleep_offset
data.loc[:,'prev_night_sleep_duration'] -= sleep_offset


# In[7]:


age_regressors   = set(['age_at_test'])
sleep_regressors = set(['np.power(typical_sleep_duration, 2)', 'typical_sleep_duration'])
other_covariates = set(['gender', 'education', 'anxiety', 'depression'])
age_by_sleep     = ss.build_interaction_terms(age_regressors, sleep_regressors)
full_model       = age_regressors | sleep_regressors | age_by_sleep | other_covariates


# In[8]:


# Print a model expression, so we can see what it looks like:
print(ss.build_model_expression(full_model))


# In[9]:


# Likelihood Ratio (LR) Tests for different effects
model_comparisons = [
    {'name':'Age',
     'h0':ss.build_model_expression(full_model - age_regressors),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Typical Sleep Duration',
     'h0':ss.build_model_expression(full_model - sleep_regressors),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Quadratic',
     'h0':ss.build_model_expression(full_model - set(['np.power(typical_sleep_duration, 2)'])),
     'h1':ss.build_model_expression(full_model)},
    
    {'name':'Age X Sleep Duration',
     'h0':ss.build_model_expression(full_model - age_by_sleep),
     'h1':ss.build_model_expression(full_model)},
    ]  


table2_results = ss.compare_models(model_comparisons, data, score_columns, n_comparisons=eff_num_scores)

figS1a = ss.create_stats_figure(table2_results, 'LR', 'p', alpha=eff_alpha)
figS1a.savefig('../images/FigureS1a.pdf', format='pdf')
figS1b = ss.create_bayes_factors_figure(table2_results)
figS1b.savefig('../images/FigureS1b.pdf', format='pdf')

table2_results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:].to_excel('../CSVs/Table2.xlsx')
table2_results.loc[idx[:,ss.TEST_NAMES],:].to_excel('../CSVs/Table2_tests.xlsx')

pd.options.display.float_format = '{:.3f}'.format
table2_results.loc[idx[:,ss.COMPOSITE_SCORE_NAMES],:]


# In[10]:


# Re-build and estimate regression models for the four composite scores
# These will be used to table of parameters, generate plots, etc.
estimated_models = [smf.ols(ss.build_model_expression(full_model)%score, data=data).fit() for score in score_columns[-4:]]


# In[11]:


print("Design matrices are %d x %d" % estimated_models[0].model.exog.shape)


# In[12]:


# Build a table of estimated parameters (for effects of interest)
# Includes the coefficient value, standard error of the estimate,
# t value, p value, and 95% confidence intervals.
parameters_to_show = list(sleep_regressors)+ list(age_regressors) + list(age_by_sleep)
parameter_index    = pd.MultiIndex.from_product([parameters_to_show, ss.COMPOSITE_SCORE_NAMES],
                                                names=['parameter', 'score'])
parameter_table    = pd.DataFrame(index=parameter_index, columns=estimated_models[-1].summary().tables[1].data[0][1:])
for index, score in enumerate(ss.COMPOSITE_SCORE_NAMES):
    all_parameter_info = np.array(estimated_models[index].summary().tables[1].data)
    all_parameter_info = pd.DataFrame(data=all_parameter_info[1:,1:],
                                      index=all_parameter_info[1:,0],
                                      columns=all_parameter_info[0,1:])
    parameter_table.loc[idx[:,score],:] = all_parameter_info.loc[parameters_to_show,:].values
parameter_table.to_excel('../CSVs/Table1.xlsx')
parameter_table


# In[13]:


x = estimated_models[-1]
x.df_resid


# In[14]:


# Generate predicted scores over a range of typical sleep durations
# For this, just use the overall score model
overall_score_model = estimated_models[-1]
overall_prediction  = ss.get_prediction_and_confidence(overall_score_model, 'typical_sleep_duration').summary_frame()

# Plot the predicted scores over-top of a scatter plot of the data
from scipy.stats import gaussian_kde
x = data['typical_sleep_duration']
y = data['Overall_score']
xy = np.vstack([x,y])
z = gaussian_kde(xy,bw_method=0.7)(xy)
fig, axs = plt.subplots(figsize=(4.35,3))
axs.scatter(x+sleep_offset,y,c=z,s=120,edgecolor='')
axs.fill_between(overall_prediction.index+sleep_offset, overall_prediction['mean_ci_lower'], overall_prediction['mean_ci_upper'], color=ss.COMPOSITE_SCORE_COLORS[-1], alpha=0.5)
axs.plot(overall_prediction.index+sleep_offset, overall_prediction['mean'], color=ss.COMPOSITE_SCORE_COLORS[-1], linewidth=3)
axs.set_xlabel("Typical Sleep Duration (hours)")
axs.set_ylabel("Overall Score (SDs)")
plt.show()
fig.savefig('../images/Figure2.pdf', format='pdf')


# In[19]:


# Plot the predicted score as a function of reported sleep duration
# with 95% confidence intervals. Also, calculate the location of the
# quadratic vertex (in hours of sleep) and it's 95% CIs.
fig, axs = plt.subplots(figsize=(8.7,6), nrows=2, ncols=2, sharey=True, sharex=True)
labels   = ['A) ', 'B) ', 'C) ', 'D) ']
for score_index, score_model in enumerate(estimated_models):
    plot_index  = np.unravel_index(score_index, [2,2])
    score_color = ss.COMPOSITE_SCORE_COLORS[score_index]
    score_name  = ss.COMPOSITE_SCORE_NAMES[score_index]
    prediction  = ss.get_prediction_and_confidence(score_model, 'typical_sleep_duration').summary_frame()
    
    axs[plot_index].fill_between(prediction.index+sleep_offset, prediction['mean_ci_lower'], prediction['mean_ci_upper'], alpha=0.3, color=score_color)
    axs[plot_index].plot(prediction.index+sleep_offset, prediction['mean'], c=score_color, label=labels[score_index]+score_name, linewidth=3)

    vertex    = ss.calculate_parabola_vertex(score_model, 'typical_sleep_duration')
    vertex_CI = ss.fieller_ci(score_model, 'typical_sleep_duration')+sleep_offset
    if score_index > 0:
        axs[plot_index].axvspan(vertex_CI[0], vertex_CI[1], color=score_color, alpha=0.2)
        print('%20s peak: %.02f (%.03f-%.03f)'%(score_name, vertex.x+sleep_offset, vertex_CI[0], vertex_CI[1]))
    axs[plot_index].axvline(x=vertex.x+sleep_offset, c=score_color, dashes=[3,2])    
    axs[plot_index].set_ylim([-0.4,0.2])
    axs[plot_index].set_xlim([3,12])

    axs[plot_index].legend(loc='upper left')

axs[0,0].set_ylabel('Score (SDs)')
axs[1,0].set_ylabel('Score (SDs)')
axs[1,0].set_xlabel('Typical Sleep Duration (hours)')
axs[1,1].set_xlabel('Typical Sleep Duration (hours)')
plt.tight_layout()
fig.savefig('../images/Figure3.pdf', format='pdf')


# In[17]:


# Plot the difference betwen the predicted score curve and the optimum of that curve.
# That is how different are predcited scores for every sleep duration from the predicted
# score that occurs at the optimal amount?
# Also, let's test the size of this difference for 4 hours of sleep.
test_duration = 4 # We will test difference between 4 hours of sleep, and the optimum
fig, axs = plt.subplots(figsize=(8.7,6), nrows=2, ncols=2, sharey=True, sharex=True)
for score_index, score_model in enumerate(estimated_models):
    plot_index  = np.unravel_index(score_index, [2,2])
    score_color = ss.COMPOSITE_SCORE_COLORS[score_index]
    score_name  = ss.COMPOSITE_SCORE_NAMES[score_index]
    prediction  = ss.get_prediction_and_confidence(score_model, 'typical_sleep_duration')
    predict_xy  = prediction.summary_frame()
    x_pts       = np.array(predict_xy.index)
    vertex      = ss.calculate_parabola_vertex(score_model, 'typical_sleep_duration')
    vertex_pr   = ss.get_prediction_and_confidence(score_model, 'typical_sleep_duration', x=[vertex.x]).summary_frame()

    # Calculate the difference between the predicted score cuve, and the peak of that curve
    diff_from_mn = vertex.y - predict_xy['mean']
    score_var    = np.power(predict_xy['mean_se'], 2)       # Variance around each point in the curve
    max_pt_var   = np.power(vertex_pr['mean_se'], 2).values # Variance around the peak
    diff_se      = np.sqrt(score_var+max_pt_var)            # Variance adds when calculating the difference
    df           = prediction.dist_args[0] -1
    q            = prediction.dist.ppf(1-0.05/2., df)
    diff_ci_lo   = diff_from_mn - q * diff_se
    diff_ci_hi   = diff_from_mn + q * diff_se
    
    # Plot the curve
    axs[plot_index].plot([0,16],[0,0], c='0.4', linewidth=1.0)
    axs[plot_index].fill_between(x_pts+sleep_offset, diff_ci_lo, diff_ci_hi, alpha=0.25, color=score_color)
    axs[plot_index].plot(x_pts+sleep_offset, diff_from_mn, c=score_color, linewidth=3, label=score_name)
    
    # Only do this analysis for models with a significant quadratic fit (i.e., not memory) 
    if score_model.model.endog_names != 'STM_score':

        print(score_model.model.endog_names.capitalize())
        intersection_pt = np.argmax(diff_ci_lo < 0)
        print('Lower confidence bound intersects y=0 at at %.02f hours'%(intersection_pt+sleep_offset))

        # Plot the vertical line where lower confidence bound intersects y=0 (on the left)
        axs[plot_index].axvline(x=intersection_pt+sleep_offset, c=score_color, linestyle=':')

        # Plot the vertical line where the vertex is located
        axs[plot_index].axvline(x=vertex.x+sleep_offset, c=score_color, dashes=[3,2])

        # Calculate the difference from peak for a given sleep duration, and stats
        test_x_point  = x_pts[np.argmax(x_pts+sleep_offset > 4)-1] # Index of first point to the left of our test point
        diff_at_test  = diff_from_mn[test_x_point]
        t_of_diff     = diff_at_test / diff_se[test_x_point]
        p_of_diff     = 2 * (1 - stats.t.cdf(t_of_diff, df))
        print('Difference at 4 hours = %.02f, t(%d) = %.02f, p = %.04f' % (diff_at_test, df, t_of_diff, p_of_diff))
        print('Equivalent to aging %.02f years\n' % (-diff_at_test/score_model.params['age_at_test']))
        
    axs[plot_index].set_xlim([3,11])
    axs[plot_index].set_ylim([-0.1, 0.65])
    axs[plot_index].legend(loc='upper right')
    
axs[0,0].set_ylabel('Difference From Max (SDs)')
axs[1,0].set_ylabel('Difference From Max (SDs)')
axs[1,0].set_xlabel('Typical Sleep Duration (hours)')
axs[1,1].set_xlabel('Typical Sleep Duration (hours)')

plt.tight_layout()
fig.savefig('../images/Figure4.pdf', format='pdf')


# In[18]:


# How many people slept an amount below this detectable threshold?
cutoff = 6.30
percentage = data[data['typical_sleep_duration']<(cutoff-sleep_offset)].shape[0] / data.shape[0] * 100
print('%% of sample who reported slept less than %.02f hours: %.02f%%'%(cutoff, percentage))


# In[19]:


# Generate a plot of predicted domain score vs age.
fig, axs   = plt.subplots()
for score_index, score_model in enumerate(estimated_models):
    score_color = ss.COMPOSITE_SCORE_COLORS[score_index]
    score_name  = ss.COMPOSITE_SCORE_NAMES[score_index]    
    prediction  = ss.get_prediction_and_confidence(score_model, 'age_at_test').summary_frame()
    axs.fill_between(prediction.index+age_offset, prediction['mean_ci_lower'], prediction['mean_ci_upper'], alpha=0.3, color=score_color)
    axs.plot(prediction.index+age_offset, prediction['mean'], label=score_name, linewidth=3, color=score_color)

axs.set_xlabel('Age at Test (Years)')
axs.set_ylabel('Score (stdevs)')
plt.legend(loc='lower left')
plt.show()

