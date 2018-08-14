
# coding: utf-8

# # Part 1 - Data Preprocessing, and Descriptives

# In[1]:


# Import all required Python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg

# Import custom sleep study code
import sys
sys.path.insert(0, '../lib')
import sleep_study_utils as ss

idx = pd.IndexSlice


# In[2]:


# Change settings to embed TrueType fonts
# http://phyletica.org/matplotlib-fonts/
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Default font size to 8pt
matplotlib.rcParams.update({'font.size': 8})


# In[3]:


# Display only 3 decimal points in tables
pd.options.display.float_format = '{:.3f}'.format


# In[4]:


# List of all columns in the data frame that have scores
score_columns = ss.score_columns()


# In[5]:


# Load the data
data = pd.read_pickle('../data/sleep_study_data.pickle.bz2')
data.info() # List all the columns in the data frame


# In[6]:


data = data[~data.index.duplicated(keep='first')] # Remove duplicate rows
data = data.drop(columns='odd_one_out_max')       # Remove unused score column
print("%d people completed the study"%data.shape[0])


# In[7]:


# Remove subjects reported older than 100 or younger than 10 years of age
data = data[(data['age_at_test'] <= 100) & (data['age_at_test'] >= 18)]
print("%d people remain after filtering for age"%data.shape[0])

# Remove subjects who reported sleep zero or more than 16 hours per night.
data = data[(data['typical_sleep_duration'] > 0) & (data['typical_sleep_duration'] <= 16)]
# data = data[(data['prev_night_sleep_duration'] > 0) & (data['prev_night_sleep_duration'] <= 16)]
print("%d people after filtering for sleep duration"%data.shape[0])

# Remove subject who are missing data in any column (i.e., any questionnaire response or test score)
data = data.dropna(axis=0, how='any')
print("%d people have data in all columns"%data.shape[0])

# In two passes, remove scores that are more than 6 then more than 4 standard deviations from the mean
data = ss.filter_df_with_stdevs(data, scores=score_columns, stdevs=[6])
print("%d people after 6 stdev filter"%data.shape[0])

data = ss.filter_df_with_stdevs(data, scores=score_columns, stdevs=[4])
print("%d people after 4 stdev filter"%data.shape[0])

data.describe()


# In[8]:


# Convert gender (stored as string) to M/F/Other categories
data.loc[:,'gender'] = data['gender'].cat.set_categories(['Male','Female','Other'])
data.loc[:,'gender'] = data['gender'].fillna('Other')


# In[9]:


# Convert test scores to z-scores, calculate an "overall" score, and domain scores
all_test_scores   = data[score_columns].copy()
all_test_z_scores = all_test_scores.apply(stats.zscore)

# Calculate the three orthogonal domain scores using the factor loadings
domain_names  = [score+"_score" for score in ss.COMPOSITE_SCORE_NAMES[:-1]]
domain_scores = np.dot(all_test_z_scores, linalg.pinv(ss.FACTOR_LOADINGS).T)
for i, name in enumerate(domain_names):
    data.loc[:, name] = domain_scores[:, i]

# Calculate the Overall Score as the mean of all 12 test z-scores
# Then scale the Overall Score so that the population mean = 0 and stdev = 1.0
data.loc[:, 'Overall_score'] = all_test_z_scores.mean(axis=1)
data.loc[:, 'Overall_score'] = data[['Overall_score']].apply(stats.zscore)
domain_names  += ['Overall_score']
score_columns += domain_names


# In[10]:


# Display the factor loadings (in a table for supplementary materials)
factor_loading_df = pd.DataFrame(ss.FACTOR_LOADINGS, index=ss.TEST_NAMES, columns=ss.COMPOSITE_SCORE_NAMES[:-1])
factor_loading_df.to_excel('../CSVs/TableS3.xlsx')
factor_loading_df


# In[11]:


score_summary = data[score_columns].describe().T
score_summary.index =  [score[:-6] for score in score_columns]
score_summary.to_excel('../CSVs/TableS4.xlsx')
score_summary


# # Participant Descriptives

# In[12]:


print("Number of people older than 70 years: %d"%data[data['age_at_test']>70].shape[0])


# In[13]:


# Histogram and counts for gender categories
ss.create_histogram(data, 'gender')


# In[14]:


# Re-order levels education (from least to most)
education_level_order = [ "None", "High School Diploma", "Bachelor's Degree",  "Master's Degree", "Doctoral or Professional Degree"]
data['education'] = data['education'].cat.reorder_categories(education_level_order, ordered=True)


# In[15]:


# Get counts for each level of education 
ss.create_histogram(data, 'education')


# In[16]:


# Re-order the categories representing frequency of anxiety-related episodes (least to most)
anxiety_level_order = ['Not during the past month', 'Less than once a week', 'Once or twice a week', 'Three or more times a week', 'Every day']
data['anxiety'] = data['anxiety'].cat.reorder_categories(anxiety_level_order, ordered=True)


# In[17]:


# Get the counts for the frequency of anxiety-related episodes
ss.create_histogram(data, 'anxiety')


# In[18]:


# Re-order the frequency of depressive episodes (least to most)
depression_level_order = ['Not during the past month', 'Less than once a week', 'Once or twice a week', 'Three or more times a week', 'Every day']
data['depression'] = data['depression'].cat.reorder_categories(depression_level_order, ordered=True)


# In[19]:


ss.create_histogram(data, 'depression')


# In[20]:


# Split continuous variables into some bins, just for table purposes
age_bin_names   = ['18-30','30-40','40-50','50-60','60+']
data['age_bin'] = pd.cut(data['age_at_test'], [17,30,40,50,60,150], labels = age_bin_names)

sleep_bin_names = ['0-6','6-8','8+']
data['typical_sleep_bin'] = pd.cut(data['typical_sleep_duration'], [-1,6,8,24], labels=sleep_bin_names)
data['prev_night_sleep_bin'] = pd.cut(data['prev_night_sleep_duration'], [-1,6,8,24], labels=sleep_bin_names)


# In[21]:


# Table of responses to each questionnaire item
demo_questions = ['gender', 'education', 'anxiety', 'depression', 'age_bin', 'typical_sleep_bin', 'prev_night_sleep_bin']
demo_index     = [(question,category) for question in demo_questions for category in data[question].value_counts().index]
demo_index     = pd.MultiIndex.from_tuples(demo_index)
demographics   = pd.DataFrame(index=demo_index, columns=['count'])
for question in demo_questions:
    demographics.loc[idx[question,:], 'count'] = data[question].value_counts().values
demographics.to_excel('../CSVs/TableS1.xlsx')


# In[22]:


demographics


# ## Other Variables

# In[23]:


# Load the optional questionnaire data, select only rows (subjects)
# that correspond to participants in the final sample.
other_qs = pd.read_pickle('../data/other_questionnaire.pickle.bz2')
other_qs = other_qs.reindex(data.index)
other_qs = other_qs.reset_index()


# In[24]:


# Responses may be stored as "multi-select" (i.e., comma separated options)
# like "english,french,spanish". So, break those apart and store each as a
# new response with the user's ID.
for q in ['country_of_origin', 'languages_spoken']:
    for index, row in other_qs.iterrows():
        if type(row[q]) is str and "," in row[q]:
            answers = row[q].split(',')
            row = row.to_frame().T
            new_rows = pd.concat([row]*len(answers), ignore_index=True)
            new_rows.loc[:,q] = answers
            other_qs = pd.concat([other_qs,new_rows], ignore_index=True)
            other_qs.loc[row.index,q] = np.nan


# In[25]:


# Tally the counts of languages and countries, but pool responses to 
# options that less than 1% of people selected into an "other" category.
response_threshold = data.shape[0]*.01 # 1% of our final sample size

# Language(s) spoken at home
language_counts = other_qs['languages_spoken'].value_counts()
languages_rarely_chosen = language_counts[language_counts<response_threshold]
other_response_count = pd.Series(data=[languages_rarely_chosen.sum()], index=['other'])
languages_frequently_chosen = language_counts[language_counts>response_threshold]
language_counts = languages_frequently_chosen.append(other_response_count)
language_counts.name = "What language(s) do you primarily speak at home?"
print("'Other' languages include %d possible options"%languages_rarely_chosen.size)

# Country (countries) they grew up in
country_counts = other_qs['country_of_origin'].value_counts()
countries_rarely_chosen = country_counts[country_counts<response_threshold]
other_response_count = pd.Series(data=[countries_rarely_chosen.sum()], index=['other'])
countries_frequently_chosen = country_counts[country_counts>response_threshold]
country_counts = countries_frequently_chosen.append(other_response_count)
country_counts.name = "What country (or countries) did you grow up in?"
print("'Other' countries include %d possible options"%countries_rarely_chosen.size)


# In[26]:


# Create a dataframe to save as an .xlsx
index_levels = [(q.name, a) for q in [language_counts, country_counts] for a in q.index.values]
index_levels = pd.MultiIndex.from_tuples(index_levels)
other_q_results = pd.DataFrame(index=index_levels, columns=['N'])
other_q_results.loc[:,'N'] = pd.concat([language_counts, country_counts], axis=0).values
other_q_results.to_excel('../CSVs/country_languages.xlsx')
other_q_results


# ## Distribution and Scatter Plots of Age & Sleep Durations

# In[27]:


f1a = ss.joint_plot_with_data_cloud(data, 'age_at_test', 'typical_sleep_duration', 'Age (years)', 'Typical Sleep Duration (hours)')
f1a.savefig('../images/Figure_1a.pdf', format='pdf')


# In[28]:


f1b = ss.joint_plot_with_data_cloud(data, 'age_at_test', 'prev_night_sleep_duration', 'Age (years)', 'Previous Night Sleep (hours)')
f1b.savefig('../images/Figure_1b.pdf', format='pdf')


# In[29]:


f1c = ss.joint_plot_with_data_cloud(data, 'prev_night_sleep_duration', 
                                          'typical_sleep_duration',
                                          'Previous Night Sleep (hours)',
                                          'Typical Sleep Duration (hours)', )
f1c.savefig('../images/Figure_1c.pdf', format='pdf')


# ### Save the final sample data (used in subsequent analysis)

# In[30]:


data.to_pickle('../data/final_sample.pickle.bz2')

