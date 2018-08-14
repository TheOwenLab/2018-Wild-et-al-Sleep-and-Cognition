""" This script contains all helper functions and constants used by Jupyter
    notebook analyses for the study, "Dissociable effects of self-reported 
    daily sleep duration on high-level cognitive abilities."
    
    Last Updated: 2018-06-19 by cwild

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import gaussian_kde
from patsy import ContrastMatrix
from collections import namedtuple

import pdb, traceback, sys
import itertools


idx = pd.IndexSlice

TEST_NAMES = ['spatial_span',
              'grammatical_reasoning',
              'double_trouble',
              'odd_one_out',
              'monkey_ladder',
              'rotations',
              'feature_match',
              'digit_span',
              'spatial_tree',
              'paired_associates',
              'polygons',
              'token_search']

QUESTIONNAIRE_ITEMS = ['age_at_test',
                       'education',
                       'gender',
                       'depression',
                       'anxiety',
                       'typical_sleep_duration',
                       'prev_night_sleep_duration']

# Varimax rotated principal component loadings (N=3 components) derived
# from the data of Hampshire et al. (2012). Each row is the loading of 
# each test (in the order of TEST_NAMES) on the 1st three components, 
# which have been interpreted as STM, Reasoning, and Verbal.
FACTOR_LOADINGS = np.array(
    [[     0.7014310832595965,   0.2114602614099599,  0.06335835494755199],
     [    0.06489246556607997,   0.3200071012586953,   0.6555755842264829],
     [    0.21528601455159402,   0.3526924332953027,   0.5250830836410911],
     [     0.1992332685797254,   0.5526963820789189, -0.11183812310895136],
     [     0.6968350824881397,   0.2277093563746992, 0.053842026805891285],
     [    0.12127517611619794,   0.6653527253557938,  0.09299029746874055],
     [    0.16663501355802068,   0.5504977638792241,   0.2190648077992447],
     [     0.2697350970669865, -0.20477442132929694,   0.7086057589964939],
     [     0.4015791806929251,  0.45692510567203554,  0.03807692316482178],
     [     0.5987583784444802, -0.03013309539255579,  0.23366793983112322],
     [-0.00047574212977022357,   0.5198168539290584,   0.3378749934189247],
     [     0.6001410364160199,   0.1816371349537731,  0.18355395542544883]])  

COMPOSITE_SCORE_NAMES = ['STM', 'Reasoning', 'Verbal', 'Overall']

# A set of four maximally distinct colors, generated from 
# tools.medialab.sciences-po.fr/iwanthue/
COMPOSITE_SCORE_COLORS = [
    (0.8509803921568627,  0.37254901960784315, 0.00784313725490196),
    (0.9058823529411765,  0.1607843137254902,  0.5411764705882353),
    (0.10588235294117647, 0.6196078431372549,  0.4666666666666667),
    (0.4588235294117647,  0.4392156862745098,  0.7019607843137254)]

def score_columns():
    """ Returns a list of the of columns names associated with test scores for the sleep study.
        Can be used to select test score columns from a full data frame.
        
    Args:
        None
    
    Returns:
        A list of strings.
        
    Example:
        >>> score_columns()
            ['spatial_span_score',
             'grammatical_reasoning_score',
             'double_trouble_score',
             'odd_one_out_score',
             'monkey_ladder_score',
             'rotations_score',
             'feature_match_score',
             'digit_span_score',
             'spatial_tree_score',
             'paired_associates_score',
             'polygons_score',
             'token_search_score']
    """
    return [test+"_score" for test in TEST_NAMES]

def create_histogram(data, column):
    """ Creates a histogram plot of values counts for the specified column in the provided
        data frame. Also places counts (as text) above bars.
    
    Args:
        data (pandas dataframe): the full data frame.
        column (string): which column of the data frame to build a histogram for.
    
    Returns:
        Nothing
    """
    plot_axis  = sns.countplot(x=column, data=data)
    categories = [tick.get_text() for tick in plot_axis.get_xticklabels()]
    all_counts = data[column].value_counts()

    for index, category in enumerate(categories):
        plot_axis.text(index, all_counts[category]+all_counts.max() * 0.02, str(all_counts[category]), horizontalalignment='center')

    plot_axis.set_xticklabels(categories, rotation=20)
    plot_axis.set_xlabel('')
    plot_axis.spines['top'].set_visible(False)
    plot_axis.spines['right'].set_visible(False)
    plt.show()

def joint_plot_with_data_cloud(data, x_column, y_column, x_title=None, y_title=None, bw=0.7, ax=None):
    """ Creates a scatter plot of y_column vs x_column, and uses a gaussian kernel to estimate
        the probability density of the data cloud, instead of just plotting a mass of opaque dots.
        Also places histograms (and estimated pdfs) for each variable along the corresponding axis.
        Also includes a line of best fit and associated statistics.
    """
    if x_title is None:
        x_title = x_column
    if y_title is None:
        y_title = y_column

    xy  = np.vstack([data[x_column],data[y_column]])
    z   = gaussian_kde(xy,bw_method=bw)(xy)
    fig = sns.jointplot(x_column, y_column, data=data, kind='reg', space=0,
                       marginal_kws=dict(bins=10,kde_kws=dict(bw=bw)), 
                       annot_kws=dict(stat='r'), size=3.0)

    del fig.ax_joint.collections[0]
    fig.ax_joint.scatter(data[x_column], data[y_column], c=z, cmap='viridis')
    fig.ax_joint.set_ylabel(y_title)
    fig.ax_joint.set_xlabel(x_title)
    plt.show()
    return fig 
    

def build_model_expression(regressors):
    """ Given a list (or set) of strings that are variables in a dataframe, builds an expression
        (i.e., a string) that specifies a model for multiple regression. The dependent variable (y)
        is filled with '%s' so that it can replaced as in a format string.
    
    Args:
        regressors (list or set of strings): names of independent variables that will be used 
            to model the dependent variable (score).
        
    Returns:
        string: the model expression
                
    Example:
        >>> build_model_expression(['age', 'gender', 'other'])
            '%s ~ age+gender+other'
    """
    return '%s ~ '+'+'.join(regressors)

def build_interaction_terms(*regressors):
    """ Given multiple lists (or sets) of regressors, returns a set of all interaction terms.
        The set of interaction terms can then be used to build a model expression. Note that
        because Python 'sets' are used, order of items may not be preserved.
    
    Args:
        *regressors (mutiple list or sets of strings)
            
    Returns:
        the set of all interaction terms.
            
    Examples:
        >>> build_interaction_terms(['age'], ['sleep_1', 'sleep_2'])
            {'age:sleep_1', 'age:sleep_2'}
        >>> build_interaction_terms(['age'], ['sleep_1', 'sleep_2'], ['gender_male', 'gender_other'])
            {'age:sleep_1:gender_male',
             'age:sleep_1:gender_other',
             'age:sleep_2:gender_male',
             'age:sleep_2:gender_other'}
    """
    return set([':'.join(pair) for pair in itertools.product(*regressors)])


def adjust_data(estimated_model, term_names):
    """ Given an estimated regression model, adjusts the dependent variable for the 
        specified terms. Basically, regress out the effects of those variables.
        
        Args:
            estimated_model (): An estimated OLS model
            term_names (list of strings): a list of term names
        
        Returns:
            A numpy array.
            
        Examples:
        
    """
    design = estimated_model.model.data.design_info
    X0_ind = [np.r_[design.term_name_slices[term]] for term in term_names]
    X0_ind = np.concatenate(X0_ind)
    X      = estimated_model.model.exog
    B      = estimated_model.params.values
    X0     = X[:, X0_ind]
    B0     = np.expand_dims(B[X0_ind], axis=1)
    Y0     = np.dot(X0, B0)
    Y      = np.expand_dims(estimated_model.model.endog, axis=1)
    Yc     = Y - Y0
    return Yc
    
def compare_models(model_comparisons, data, score_columns, alpha=0.05, n_comparisons=None, correction='bonferroni'):
    """ Performs statistical analyses to compare fully specified regression models to (nested)
        restricted models. Uses a likelihood ratio test to compare the models, and also a 
        Bayesian model comparison method.
        
    References:
        http://www.statsmodels.org/dev/regression.html
        http://www.statsmodels.org/dev/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
        http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.compare_lr_test.html
        
    Args:
        model_comparisons (list of dicts): each dict in the list represents a contrast
            between two models. The dict must have: a 'name' field; a 'h0' field that 
            is the model expression for the restricted (or null) model; and a 'h1' field
            that is the model expression string for the fully specified model.
        data (panads data frame): the full data frame with all independent variables
            (predictors) and dependent variables (scores). Model expression strings in
            the previous arg are built from names of columns in this data frame. 
        score_columns (list of strings): which columns of the data frame are dependent
            variables to be modelled using expressions in model_comparisons?
        alpha (float): What considered a significant p-value after correction.
        n_comparisons (float): If specified, this is used to correct the p-values returned
            by the LR tests (i.e., multiply each p-value by the number of comparisons).
            The resulting adjusted p-value is what is compared to the alpha argument.
            Note: this take precendence over the multiple correction type argument.
        correction (string): method for correcting for multiple comparisons. Can be
            None, or any of the options listed in the above documentation for multipletests.
                
    Returns:
        A pandas dataframe with one row per score, and a multindex column structure that 
        follows a (contrast_name, statistic_name) convention.
    
    """
    score_names = [column_name[:column_name.rfind("_")] for column_name in score_columns]
    contrasts   = [comparison['name'] for comparison in model_comparisons]
    statistics  = ['LR', 'p', 'p_adj', 'df', 'Delta R^2', 'Cohen f^2', 'BF_01', 'BF_10']
    results_df  = pd.DataFrame(index   = pd.MultiIndex.from_product([contrasts, score_names]), 
                               columns = statistics)
    for contrast in model_comparisons:
        for score_index, score in enumerate(score_columns):
            score_name = score_names[score_index]
            
            # Fit the fully specified model (h1) and the nested restricted model (h0) using OLS
            h0 = smf.ols(contrast['h0']%score, data=data).fit()
            h1 = smf.ols(contrast['h1']%score, data=data).fit()
    
            # Perform Likelihood Ratio test to compare h1 (full) model to h0 (restricted) one
            lr_test_result = h1.compare_lr_test(h0)
            bayesfactor_01 = bayes_factor_01_approximation(h1, h0, min_value=0.0000001, max_value=10000000)
            all_statistics = [lr_test_result[0],
                              lr_test_result[1],
                              np.nan,
                              lr_test_result[2], 
                              h1.rsquared - h0.rsquared,
                              cohens_f_squared(h1, h0),
                              bayesfactor_01,
                              1/bayesfactor_01]
            results_df.loc[(contrast['name'], score_name), :] = all_statistics
        
        # Correct p-values for multiple comparisons across all tests of this contrast?
        contrast_p_vals = results_df.loc[idx[contrast['name'],:], 'p']
        if n_comparisons is not None:
            adjusted_p_vals = np.clip(contrast_p_vals * n_comparisons, 0, 1)
            results_df.loc[idx[contrast['name'],:], 'p_adj'] = adjusted_p_vals
        elif correction is not None:
            adjusted_p_vals = multipletests(contrast_p_vals.values, alpha=alpha, method=correction)
            results_df.loc[idx[contrast['name'],:], 'p_adj'] = adjusted_p_vals[1]
            
    return results_df

def cohens_f_squared(full_model, restricted_model):
    """ Calculate Cohen's f squared effect size statistic. See this reference:
    
        Selya, A. S., Rose, J. S., Dierker, L. C., Hedeker, D., & Mermelstein, R. J. (2012). 
            A practical guide to calculating Cohen’s f 2, a measure of local effect size, 
            from PROC MIXED. Frontiers in Psychology, 3, 1–6.
    
    """
    return (full_model.rsquared-restricted_model.rsquared)/(1-full_model.rsquared)

def bayes_factor_01_approximation(full_model, restricted_model, min_value=0.001, max_value=1000):
    """ Estimate Bayes Factor using the BIC approximation outlined here:
        Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems of p values.
            Psychonomic Bulletin & Review, 14, 779–804.
        
    Args:
        full_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The estimated multiple regression model that represents H1 - the alternative hypothesis
        restricted_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            The estimated multiple regression model that represents H0 - the null hypothesis
        min_value (float): a cutoff to prevent values from getting too small.
        max_value (float): a cutoff to prevent values from getting too big
    
    Returns:
        A float - the approximate Bayes Factor in support of the null hypothesis
    """
    bf = np.exp((full_model.bic - restricted_model.bic)/2)
    return np.clip(bf, min_value, max_value)

def effective_number_of_comparisons(data):
    """ A method for adjusting statistical significance thresholds for multiple comparisons, without
        the assumption of independence of tests. Instead of using the actual number of tests, the 
        effective number of tests is estimated from the correlations among the variables.
        
        Nyholt, D. R. (2004). A Simple Correction for Multiple Testing for Single-Nucleotide Polymorphisms 
            in Linkage Disequilibrium with Each Other. The American Journal of Human Genetics, 74, 765–769.
            
    Args:
        data (NumPy matrix): variables as columns, observations as rows
    Returns:
        A float - the estimated number of effective tests.
    
    """
    num_scores = data.shape[1]
    score_corr = np.corrcoef(data, rowvar=False)
    eigen_vals = np.linalg.eigvals(score_corr)
    eff_num_scores = 1+(num_scores-1)*(1-np.var(eigen_vals)/num_scores)
    return eff_num_scores

def list_union(listA, listB):
    """ Find the union (intersection) of two lists (A & B). Note that because the python
        data type 'sets' is used, order may not be preserved.
    
    Args:
        listA (list): a list of strings, or whatever
        listB (list): a list of strings, or whatever
        
    Returns:
        A list that is the union of listA & listB
        
    Example:
        >>> list_union(['cat', 'dog','horse'], ['mouse', 'dog', 'snake', 'cat', 'pig'])
            ['dog', 'cat']
    """
    return list(set(listA) & set(listB))

def filter_df_with_stdevs(df, scores = TEST_NAMES, stdevs=[6,4]):
    """
    """
    
    for column in df:
        this_score = [score for score in scores if score in column]
        if this_score:
            for stdev in stdevs:
                df = df[np.abs(df[column]-df[column].mean())<=(stdev*df[column].std())]
    return df

def create_stats_figure(results, stat_name, p_name, alpha=0.05, log_stats=True, correction=None):
    """ Creates a matrix figure to summarize multple tests/scores. Each cell represents a contrast
        (or model comparison) for a specific effect (rows) for a given score (columns). Also
        draws asterisks on cells for which there is a statistically significant effect.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to display. Should
            be a rectangular dataframe with tests as rows and effects as columns (i.e., the 
            transpose of the resulting image). The dataframe index and column labels are used
            as labels for the resulting figure.
        stat_name (string): Which statistic to plot. There might be multiple columns for each
            effect (e.g., Likelihood Ratio, BFs, F-stats, etc.)
        p_name (string): The name of the column to use for p-values.
        alpha (float): what is the alpha for significant effects?
        log_stats (boolean): Should we take the logarithm of statistic values before creating 
            the image? Probably yes, if there is a large variance in value across tests and
            effects.
        correction (string): indicates how the alpha was corrected (e.g., FDR or bonferroni) so
            the legend can be labelled appropriately.
            
    Returns:
        A matplotlib figure.
        
    """
    
    score_index   = results.index.levels[1][results.index.labels[1]].unique()
    contrast_index = results.index.levels[0][results.index.labels[0]].unique()
    stat_values   = results.loc[:, stat_name].unstack().T.reindex(index=score_index, columns=contrast_index)
    p_values      = results.loc[:, p_name].unstack().T.reindex(index=score_index, columns=contrast_index)
    num_scores    = stat_values.shape[0]
    num_contrasts = stat_values.shape[1]
    image_values  = stat_values.values.astype('float32')
    image_values  = np.log10(image_values) if log_stats else image_values
    
    figure   = plt.figure(figsize=[num_scores*0.6, num_contrasts*0.6])
    plt_axis = figure.add_subplot(1,1,1) 
    imgplot  = plt_axis.imshow(image_values.T, aspect='auto', clim=[0,np.min([3,np.max(image_values)])])
    
    plt_axis.plot([num_scores-4.5,num_scores-4.5],[-0.5,num_contrasts-0.5], c='w')
    plt_axis.set_yticks(np.arange(0,num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    if log_stats:
        cbar.ax.set_ylabel('$Log_{10}(LR)$')
    else:
        cbar.ax.set_ylabel('LR')
    
    reject_h0 = (p_values.values.T < alpha).nonzero()
    legend_label = "p < %.04f" % alpha
    if correction is not None:
        legend_label += " (%s)" % correction 
    plt_axis.plot(reject_h0[1],reject_h0[0],'r*', markersize=10, label=legend_label)
    
    plt.legend(bbox_to_anchor=(1, 1.1), loc=4, borderaxespad=0.)
    plt.show()
    return figure

def create_bayes_factors_figure(results, log_stats=True):
    """ Creates a matrix figure to summarize Bayesian stats for multiple scores & tests.
        Each cell indicates the Bayes Factor (BF associated with a model comparison) for 
        a specific effect (rows) for a given score (columns). Also draws symbols on cells
        to indicate the interpretation of that BF.
        
    Args:
        results (Pandas dataframe): a dataframe that contains the statistics to display. Should
            be a rectangular dataframe with tests as rows and effects as columns (i.e., the 
            transpose of the resulting image). The dataframe index and column labels are used
            as labels for the resulting figure.
        log_stats (boolean): Should we take the logarithm of BF values before creating 
            the image? Probably yes, if there is a large variance in value across scores and
            effects.
            
    Returns:
        A matplotlib figure
    
    """
    
    score_index    = results.index.levels[1][results.index.labels[1]].unique()
    contrast_index = results.index.levels[0][results.index.labels[0]].unique()
    num_scores     = len(score_index)
    num_contrasts  = len(contrast_index)
    bf_values      = results.loc[:, 'BF_01'].unstack().T.reindex(index=score_index, columns=contrast_index).values.astype('float32')
    # Too small values cause problems for the image scaling
    np.place(bf_values, bf_values < 0.00001, 0.00001)
    
    figure   = plt.figure(figsize=[num_scores*0.6, num_contrasts*0.6])
    plt_axis = figure.add_subplot(1,1,1) 
    imgplot  = plt_axis.imshow(np.log10(bf_values.T), aspect='auto', cmap='coolwarm', clim=[-6.0, 6.0])
    plt_axis.plot([num_scores-4.5, num_scores-4.5], [-0.5, num_contrasts-0.5], c='w')
    plt_axis.set_yticks(np.arange(0, num_contrasts))
    plt_axis.set_yticklabels(list(contrast_index))
    plt_axis.set_xticks(np.arange(0, num_scores))
    plt_axis.set_xticklabels(list(score_index), rotation=45, ha='right')
    
    # Add a colour bar
    cbar = figure.colorbar(imgplot, ax=plt_axis, pad=0.2/num_scores)
    cbar.ax.set_ylabel('$Log_{10}(BF_{01})$')
    cbar.ax.text(0,1.05, "$H_0$")
    cbar.ax.text(0,-0.12, "$H_1$")
    
    # Use absolute BFs for determining weight of evidence
    abs_bfs = bf_values
    abs_bfs[abs_bfs==0] = 0.000001
    abs_bfs[abs_bfs<1] = 1/abs_bfs[abs_bfs<1]

    # Custom markers for the grid
    markers = [(2+i, 1+i%2, i/4*90.0) for i in range(1, 5)]
    markersize = 10
    
    # Positive evidence BF 3 - 20
    positive = (abs_bfs >= 3) & (abs_bfs < 20)
    xy = positive.nonzero()
    plt_axis.plot(xy[0],xy[1],'r', linestyle='none', marker=markers[0], label='positive', markersize=markersize)
    
    # Strong Evidence BF 20 - 150
    strong = (abs_bfs >= 20) & (abs_bfs < 150)
    xy = strong.nonzero()
    plt_axis.plot(xy[0],xy[1],'r', linestyle='none', marker=markers[1], label='strong', markersize=markersize)
    
    # Very strong evidence BF > 150
    very_strong = (abs_bfs >= 150)
    xy = very_strong.nonzero()
    plt_axis.plot(xy[0],xy[1],'r', linestyle='none', marker=markers[2], label='very strong', markersize=markersize)
    
    plt.legend(bbox_to_anchor=(0.5,1.05), loc='lower center', borderaxespad=0.,ncol=4, title='Bayes\' evidence')    
    plt.show()
    return figure

def categorical_factors(model_design):
    """ Given a model specification, returns a list of all factors that are categorical.
    
    Args:
        model_design (patsy.design_info.DesignInfo): the model design info for a regression
            model constructed using Patsy (e.g., default statsmodels)
            
    Returns:
        A list of factors.
    
    Examples:
        >>> model_design = estimated_delta_models[-1].model.data.design_info
        >>> ss.categorical_factors(model_design)
            [EvalFactor('anxiety'),
             EvalFactor('depression'),
             EvalFactor('gender'),
             EvalFactor('education')]
    
    """
    all_factors = list(model_design.factor_infos.values())
    return [factor.factor for factor in all_factors if factor.type == 'categorical']

def categorical_predicted_means(estimated_model):
    """ Given an estimated model, returns the predicted mean effect of each categorical factor
    """
    means = np.array([])
    design = estimated_model.model.data.design_info
    for factor in categorical_factors(design):
        factor_term  = [term for term in design.terms if factor in term.factors and len(term.factors)==1][0]
        factor_slice = design.term_slices[factor_term]
        proportions  = estimated_model.model.data.exog[:,factor_slice].mean(axis=0)
        proportions  = np.expand_dims(proportions, axis=0)
        factor_betas = estimated_model.params[factor_slice].values
        factor_betas = np.expand_dims(factor_betas, axis=1)
        means = np.append(means, np.dot(proportions, factor_betas))
    return means
        
def change_all_categorical_coding_to_mean(model, exclude=[], in_place=True):
    import copy
    new_model = model if in_place else copy.deepcopy(model)
    model_design = new_model.model.data.design_info

    for this_factor in categorical_factors(model_design):
        if this_factor.name() not in exclude:
            num_categories         = len(model_design.factor_infos[this_factor].categories)
            terms_with_this_factor = [term for term in model_design.terms if this_factor in term.factors]
            term_only_this_factor  = [term for term in terms_with_this_factor if len(term.factors)==1][0]
            factor_dmatrix_slice   = model_design.term_slices[term_only_this_factor]
            factor_column_means    = np.mean(new_model.model.data.exog[:,factor_dmatrix_slice], axis=0)
            for this_term in terms_with_this_factor:
                coding_contrast = model_design.term_codings[this_term][0].contrast_matrices[this_factor]
                coding_contrast.matrix = np.repeat([factor_column_means], num_categories, axis=0)
    
    return new_model

def generate_design_matrix_from_df(model, df):
    from patsy import dmatrix
    return dmatrix(model.data.design_info, df)

def generate_dummy_df(estimated_model, predictor, x=None, dx=0.01, constants={}):
    orig_df = estimated_model.model.data.frame
    x_pts   = x if x is not None else np.arange(orig_df[predictor].min(), orig_df[predictor].max(), dx)
    n_pts   = len(x_pts)

    model_factors = [factor.name() for factor in estimated_model.model.data.design_info.factor_infos]
    factors_in_df = list_union(model_factors, orig_df.columns.values)
    
    # Build a new data frame that only has rows for each of our x_pts
    category_vars = categorical_factors(estimated_model.model.data.design_info)
    category_var_names = [category.name() for category in category_vars]
    category_var_first = np.array([estimated_model.model.data.design_info.factor_infos[factor].categories[0] for factor in category_vars])
    category_var_first = np.expand_dims(category_var_first, axis=0)
    numeric_vars  = orig_df[factors_in_df].select_dtypes(include=np.number)
    
    dummy_df = pd.DataFrame(np.nan, index = x_pts, columns=factors_in_df)
    dummy_df.loc[:, category_var_names]    = np.repeat(category_var_first, n_pts, axis=0)
    dummy_df.loc[:, numeric_vars.columns]  = np.repeat(np.expand_dims(numeric_vars.mean(axis=0).values, 0), n_pts, axis=0)
    dummy_df.loc[:, predictor] = x_pts
    
    for column_name, column_value in constants.items():
        dummy_df[column_name] = column_value
    

    return dummy_df
    
def get_prediction_and_confidence(estimated_model, predictor, constants={}, x=None, dx=0.01, use_category_means=True, in_place=True):
    """
    

    
    """
    
    exog_data    = generate_dummy_df(estimated_model, predictor, x=x, dx=dx, constants=constants)
    new_model    = change_all_categorical_coding_to_mean(estimated_model, exclude=[predictor]+list(constants.keys()), in_place=in_place) if use_category_means else estimated_model
    exog_dmatrix = generate_design_matrix_from_df(new_model.model, exog_data)

    return estimated_model.get_prediction(exog_dmatrix, transform=False, row_labels=exog_data.index)

Point = namedtuple('Point', ['x', 'y'])
def calculate_parabola_vertex(estimated_model, factor_name, constants={}):
    """
       
    """
    
    mean_coding_model = change_all_categorical_coding_to_mean(estimated_model)
    exog_data         = generate_dummy_df(estimated_model, factor_name, x=[1], constants=constants)
    design_matrix = generate_design_matrix_from_df(mean_coding_model.model, exog_data)
    
    design = estimated_model.model.data.design_info

    a_factor = [factor for factor in design.factor_infos.keys() if factor.name() == 'np.power(%s, 2)'%factor_name][0]
    b_factor = [factor for factor in design.factor_infos.keys() if factor.name() == factor_name][0]
    a_terms  = [term for term in design.terms if a_factor in term.factors]
    b_terms  = [term for term in design.terms if b_factor in term.factors]
    c_terms  = [term for term in design.terms if term not in a_terms and term not in b_terms]
    a = np.sum([(estimated_model.params.iloc[slice]*design_matrix[0,slice]).sum() for slice in (design.term_slices[term] for term in a_terms)])
    b = np.sum([(estimated_model.params.iloc[slice]*design_matrix[0,slice]).sum() for slice in (design.term_slices[term] for term in b_terms)])
    c = np.sum([(estimated_model.params.iloc[slice]*design_matrix[0,slice]).sum() for slice in (design.term_slices[term] for term in c_terms)])
    
    x = -b / (2*a)
    y = a * np.power(x,2) + b * x + c
    return Point(x=x,y=y)

def fieller_ci(model, factor_name, t_0=1.96):
    """
    Equation 7 from:
        Lye, J., & Hirschberg, J. (2004). Confidence bounds for the extremum determined by a quadratic 
        regression. In Econometric Society 2004 Australasian Meetings. Econometric Society.
    """
    a_name = 'np.power(%s, 2)'%factor_name
    b_name = factor_name
    covs   = model.cov_params()
    cov_ab = covs.loc[a_name,b_name]
    var_a  = covs.loc[a_name,a_name]
    var_b  = covs.loc[b_name,b_name]
    t_a    = model.tvalues[a_name]
    t_a2   = np.power(t_a,2)
    t_b    = model.tvalues[b_name]
    t_b2   = np.power(t_b,2)
    t_02   = np.power(t_0,2)
    b_a    = model.params[a_name]
    b_b    = model.params[b_name]
    
    xx = np.sqrt(np.power((b_a*b_b-t_02*cov_ab),2)-var_a*var_b*(t_a2-t_02)*(t_b2-t_02))
    denom = 2*var_a*(t_a2-t_02)
    c1 = (t_02*cov_ab - b_a*b_b - xx)/denom 
    c2 = (t_02*cov_ab - b_a*b_b + xx)/denom 
    
    return np.array([c1,c2])

def inch2cm(*tupl):
    """ Converts a tuple of measurement in inches, to centimetres.
        Adapted from: 
            https://stackoverflow.com/questions/14708695/specify-figure-size-in-centimeter-in-matplotlib
    
    Args:
        *tupl (tuple): a bunch of quanities in inches
    
    Returns:
        *tupl (tuple): those numbers, but converted into centimetres.
        
    
    Examples:
        >>> ss.inch2cm(1,2,3)
            (2.5399986284007405, 5.079997256801481, 7.619995885202221)
    
    """
    cm = 0.393701
    if isinstance(tupl[0], tuple):
        return tuple(i/cm for i in tupl[0])
    else:
        return tuple(i/cm for i in tupl)