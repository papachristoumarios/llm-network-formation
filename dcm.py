import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize
import scipy.stats
  

def prepare_discrete_choice_model(results, candidates, bias=True, feature_names=['degree'], log_transform=True, exclude_log=[]):

    choice_sets = []
    choices = []

    chosen_set = set()
    dropped_set = set()
    df_records = []

    for result in results:
        num_choices = len(result)
        choice = np.ones((len(feature_names) + int(bias), num_choices))
        for i, r in enumerate(result):
            for j, feat_name in enumerate(feature_names):
                if log_transform and feat_name not in exclude_log:
                    choice[j, i] = np.log(r['similarity'][feat_name] + 1)
                else:
                    choice[j, i] = r['similarity'][feat_name]

            chosen_set |= {r['name']}
            dropped_set |= {r['name']} if r['dropped'] else set()

        choices.append(choice)

    for candidate in candidates:
        choice_set = np.ones((len(feature_names) + int(bias), len(candidate)))

        for i, c in enumerate(candidate):
            for j, feat_name in enumerate(feature_names):
                if log_transform and feat_name not in exclude_log:
                    choice_set[j, i] = np.log(c['similarity'][feat_name] + 1)
                else:
                    choice_set[j, i] = c['similarity'][feat_name]

            c['similarity']['y'] = 1 if c['name'] in chosen_set else 0
            c['similarity']['dropped'] = 1 if c['name'] in dropped_set else 0
            df_records.append(c['similarity'])

        choice_sets.append(choice_set)
        
    df = pd.DataFrame.from_records(df_records)

    if log_transform:
        for feat_name in feature_names:
            if feat_name not in exclude_log:
                df[feat_name] = np.log(df[feat_name] + 1)

    return choices, choice_sets, df


def prepare_discrete_choice_model(df, input_type, bias=True, feature_names=['degree'], log_transform=True, exclude_log=[]):
    if input_type == 'df':
        choice_sets = []
        choices = []

        alternative_set_ids = df['Alternative Set'].unique()

        for alternative_set_id in alternative_set_ids:
            subset = df[df['Alternative Set'] == alternative_set_id]

            choice = np.ones((len(feature_names) + int(bias), 1))
            choice[:len(feature_names), 0] = subset[subset['Chosen'] == 1][feature_names].values[0]

            for i, feat_name in enumerate(feature_names):
                if log_transform and feat_name not in exclude_log:
                    choice[i, 0] = np.log(choice[i, 0] + 1)
            choices.append(choice)

            choice_set = np.ones((len(feature_names) + int(bias), len(subset)))

            choice_set[:len(feature_names), :] = subset[feature_names].values.T

            for i, feat_name in enumerate(feature_names):
                if log_transform and feat_name not in exclude_log:
                    choice_set[i, :] = np.log(choice_set[i, :] + 1)

            choice_sets.append(choice_set)
    elif input_type == 'results_candidates':
        results, candidates = df
        choice_sets = []
        choices = []

        chosen_set = set()
        dropped_set = set()

        for result in results:
            num_choices = len(result)
            choice = np.ones((len(feature_names) + int(bias), num_choices))
            for i, r in enumerate(result):
                for j, feat_name in enumerate(feature_names):
                    if log_transform and feat_name not in exclude_log:
                        choice[j, i] = np.log(r['similarity'][feat_name] + 1)
                    else:
                        choice[j, i] = r['similarity'][feat_name]

                chosen_set |= {r['name']}
                dropped_set |= {r['name']} if r['dropped'] else set()

            choices.append(choice)

        for candidate in candidates:
            choice_set = np.ones((len(feature_names) + int(bias), len(candidate)))

            for i, c in enumerate(candidate):
                for j, feat_name in enumerate(feature_names):
                    if log_transform and feat_name not in exclude_log:
                        choice_set[j, i] = np.log(c['similarity'][feat_name] + 1)
                    else:
                        choice_set[j, i] = c['similarity'][feat_name]

                c['similarity']['y'] = 1 if c['name'] in chosen_set else 0
                c['similarity']['dropped'] = 1 if c['name'] in dropped_set else 0

            choice_sets.append(choice_set)
            
    return choices, choice_sets

def fit_discrete_choice_model(df, bias=True, feature_names=['degree', 'common_attributes', 'common_neighbors'], log_transform=True, exclude_log=[], calculate_p_values=True, calculate_average_marginal_effects=True, input_type='df'):

    choices, choice_sets = prepare_discrete_choice_model(df, bias=bias, feature_names=feature_names, log_transform=log_transform, exclude_log=exclude_log, input_type=input_type)

    theta = np.zeros(len(feature_names) + int(bias))

    ll = lambda x: -discrete_choice_model_log_likelihood(x, choice_sets, choices)

    res = scipy.optimize.minimize(ll, x0=theta, method='L-BFGS-B')

    theta = res.x

    log_likelihood = -res.fun

    standard_errors = (res.hess_inv.todense().diagonal() / len(choices)) ** 0.5
    covariance_matrix = res.hess_inv.todense() / len(choices)

    if calculate_p_values:
        p_values = np.zeros(len(feature_names))

        for i, feat_name in enumerate(feature_names):
            _, _, log_likelihood_null, _, _, _, _, _ = fit_discrete_choice_model(df, bias=bias, feature_names=list(set(feature_names) - {feat_name}), log_transform=log_transform, exclude_log=exclude_log, calculate_p_values=False, calculate_average_marginal_effects=False, input_type=input_type)
            p_values[i] = 1 - scipy.stats.chi2.cdf(2 * (log_likelihood - log_likelihood_null), df=1)
    else:
        p_values = None

    if calculate_average_marginal_effects:
        probabilities = discrete_choice_model_relative_probability(theta, choice_sets)
        ame, sdame, p_values_ame = marginal_effects(probabilities, theta, covariance_matrix, feature_names, bias=bias)
    else:
        ame = None
        sdame = None
        probabilities = None
        p_values_ame = None

    return theta, standard_errors, log_likelihood, p_values, probabilities, ame, sdame, p_values_ame

def discrete_choice_model_log_likelihood(theta, choice_sets, choices):

    log_likelihood = 0

    for choice_set, choice in zip(choice_sets, choices):
        choice_set_utilities = np.dot(theta, choice_set)
        Z = np.sum(np.exp(choice_set_utilities))

        if Z == 0:
            continue

        num_choices = choice.shape[1]

        for i in range(num_choices):
            choice_utility = np.dot(theta, choice[:, i])
            log_likelihood +=  (choice_utility - np.log(Z))

    return log_likelihood

def discrete_choice_model_relative_probability(theta, choice_sets):
    probabilities = []
    for choice_set in choice_sets:
        choice_set_utilities = np.dot(theta, choice_set)

        Z = np.sum(np.exp(choice_set_utilities))

        if Z == 0:
            continue

        probabilities.append(np.exp(choice_set_utilities) / Z)

    # num_samples x num_alternatives
    probabilities = np.array(probabilities)

    return probabilities

def marginal_effects(P, theta, cov_theta, feature_names, bias=True):
    N, J = P.shape  # N samples, J alternatives
    K = len(theta)  # Number of parameters
   
    AME = np.zeros(K)
    SE = np.zeros(K)

    # For each feature k
    for k in range(K):
        # Compute AME_k = sum over j of AME_{jk}
        ame_k = 0.0
        grad_k = np.zeros(K)  # gradient of AME_k w.r.t. theta

        for j in range(J):
            # Term = θ_k * P_ij * (1 - P_ij)
            p_term = P[:, j] * (1 - P[:, j])
            ame_jk = np.mean(theta[k] * p_term)
            ame_k += ame_jk

            # Gradient w.r.t. θ_k is mean(P_ij * (1 - P_ij))
            grad_k[k] += np.mean(p_term)

        # Store
        AME[k] = ame_k
        SE[k] = np.sqrt(grad_k @ cov_theta @ grad_k)

    # Calculate p-values
    z = AME / SE
    p_values = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))

    return AME, SE, p_values


def compare_models(df1, df2, on='Alternative Set', method='tv', bias=True, feature_names=['degree', 'common_attributes', 'common_neighbors'], log_transform=True, exclude_log=[], calculate_p_values=True, calculate_average_marginal_effects=True, input_type='df'):
    theta1, sd1, _, _, _, ame1, sdame1, p_values_ame1 = fit_discrete_choice_model(df1, bias=bias, feature_names=feature_names, log_transform=log_transform, exclude_log=exclude_log, calculate_p_values=calculate_p_values, calculate_average_marginal_effects=calculate_average_marginal_effects, input_type=input_type)
    theta2, sd2, _, _, _, ame2, sdame2, p_values_ame2 = fit_discrete_choice_model(df2, bias=bias, feature_names=feature_names, log_transform=log_transform, exclude_log=exclude_log, calculate_p_values=calculate_p_values, calculate_average_marginal_effects=calculate_average_marginal_effects, input_type=input_type)

    _, choice_sets1 = prepare_discrete_choice_model(df1, bias=bias, feature_names=feature_names, log_transform=log_transform, exclude_log=exclude_log, input_type=input_type)
    _, choice_sets2 = prepare_discrete_choice_model(df2, bias=bias, feature_names=feature_names, log_transform=log_transform, exclude_log=exclude_log, input_type=input_type)
    choice_sets_eval = choice_sets1 + choice_sets2
    
    P1 = discrete_choice_model_relative_probability(theta1, choice_sets_eval)
    P2 = discrete_choice_model_relative_probability(theta2, choice_sets_eval)

    if method == 'tv':
        distances = 0.5 * np.sum(np.abs(P1 - P2), axis=1)
    elif method == 'l2':
        distances = np.sqrt(np.sum((P1 - P2) ** 2, axis=1))
    elif method == 'kl':
        distances = np.sum(P1 * np.log(P1 / P2), axis=1)
    elif method == 'js':
        m = 0.5 * (P1 + P2)
        distances = 0.5 * (np.sum(P1 * np.log(P1 / m), axis=1) + np.sum(P2 * np.log(P2 / m), axis=1))
    else:
        raise ValueError("Unknown method: {}".format(method))

    # if bias is True, we remove the bias term from the parameters
    if bias:
        theta1 = theta1[:-1].copy()
        theta2 = theta2[:-1].copy()

    theta_spearman = np.corrcoef(np.argsort(theta1), np.argsort(theta2))[0, 1]

    return distances.mean(), distances.std(), theta_spearman, theta1, theta2, sd1, sd2, ame1, ame2, sdame1, sdame2, p_values_ame1, p_values_ame2
