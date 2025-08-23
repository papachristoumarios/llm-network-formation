import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import random
import math

# --------- Step 1: Compute network-based features ---------
def common_neighbors(G, u, v):
    return len(set(G[u]).intersection(G[v]))

def jaccard_coefficient(G, u, v):
    union = set(G[u]).union(G[v])
    inter = set(G[u]).intersection(G[v])
    return len(inter) / len(union) if union else 0

def adamic_adar(G, u, v):
    neighbors_u = set(G[u])
    neighbors_v = set(G[v])
    shared = neighbors_u & neighbors_v
    return sum(1 / math.log(len(G[w])) for w in shared if len(G[w]) > 1)

def preferential_attachment(G, u, v):
    return len(G[u]) * len(G[v])


def measure_similarity(G, u, v, profiles):
    profile1 = profiles[u]
    profile2 = profiles[v]

    similarity = 0

    for k in profile1.keys():
        if k != 'name' and k != 'neighbors' and k in profile2.keys():
            if isinstance(profile1[k], list):
                similarity += len(set(profile1[k]) & set(profile2[k]))
            elif profile1[k] == profile2[k]:
                similarity += 1
        
    return similarity

# --------- Step 2: Combine feature vectors ---------
def feature_pair(u, v, G, profiles):   
    x = np.array([
        measure_similarity(G, u, v, profiles),
        common_neighbors(G, u, v),
        jaccard_coefficient(G, u, v),
        adamic_adar(G, u, v),
        preferential_attachment(G, u, v)
    ])
    return x

# --------- Step 3: Create dataset ---------
def create_dataset(G, profiles, num_neg_samples_per_pos=1):
    X_pairs = []
    y = []
    nodes = list(G.nodes())
    seen_edges = set(G.edges())

    # Positive examples
    for u, v in G.edges():
        if u == v: continue
        X_pairs.append(feature_pair(u, v, G, profiles))
        y.append(1)

    # Negative examples
    for _ in range(len(G.edges()) * num_neg_samples_per_pos):
        while True:
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                X_pairs.append(feature_pair(u, v, G, profiles))
                y.append(0)
                break

    return np.array(X_pairs), np.array(y)

# --------- Step 4: Train logistic regression model ---------
def train_link_predictor(G, profiles, name):
    X_data, y_data = create_dataset(G, profiles)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

    # train same logit model with statsmodels
    X_train = pd.DataFrame(data=X_train, columns=['Feature Similarity', 'Common Neighbors', 'Jaccard Similarity (Neighbors)', 'Adamic-Adar Index', 'Preferential Attachment Score'])
    X_train_sm = sm.add_constant(X_train)  # Add constant for intercept
    y_train = pd.Series(y_train)
    model = sm.Logit(y_train, X_train_sm).fit(disp=0)
    

    # evaluate the model
    X_test = pd.DataFrame(data=X_test, columns=['Feature Similarity', 'Common Neighbors', 'Jaccard Similarity (Neighbors)', 'Adamic-Adar Index', 'Preferential Attachment Score'])
    X_test_sm = sm.add_constant(X_test)  # Add constant for intercept
    y_pred = model.predict(X_test_sm)
    auc = roc_auc_score(y_test, y_pred)

    with open(f'tables/combined_model_{name.lower()}_logit.txt', 'w+') as f:
        f.write(model.summary().as_text())
        f.write(f'\nROC AUC Score: {auc:.4f}\n')

    print(f"ROC AUC Score: {auc:.4f}")
    print("Statsmodels model summary:")
    print(model.summary().as_latex())

    return model

# --------- Step 5: Recommend friends ---------
def recommend_friends(model, G, profiles, node, k=5):
    candidates = set(G.nodes()) - set(G[node]) - {node}
    scores = []

    features_arr = []

    for cand in candidates:
        # make this a one-row array
        features = feature_pair(node, cand, G, profiles)
        features_arr.append(features)

    features_arr = np.array(features_arr)
    features_arr = pd.DataFrame(features_arr, columns=['Feature Similarity', 'Common Neighbors', 'Jaccard Similarity (Neighbors)', 'Adamic-Adar Index', 'Preferential Attachment Score'])
    features_arr = sm.add_constant(features_arr)  # Add constant for intercept

    prob = model.predict(features_arr)
    for i, cand in enumerate(candidates):
        scores.append((cand, prob[i]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in scores[:k]]
