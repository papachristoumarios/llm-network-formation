import numpy as np
import collections
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import glob
import pandas as pd
import os
import json
import seaborn as sns

def jaccard(X, Y):
    U = X.union(Y)
    I = X.intersection(Y)

    if len(U) == 0:
        return 1
    else:
        return len(I) / len(U)
    
def load_snap_ego_nets(input_dir, name):

    with open(f'{input_dir}/{name}/{name}_combined.json') as f:
        data = json.load(f)

    G = nx.node_link_graph(data)

    return {-1: G}

def preprocess_snap_ego_nets(input_dir, network_type='gplus', ego_nets=False, max_num_egonets=-1, max_graph_size=-1):
    """
    Load ego nets from files in input_dir.
    Returns a dictionary of networkx graphs.
    """
    

    sns.set_palette(['#2980b9', '#f1c40f', '#7f8c8d', '#d35400', '#34495e', '#e67e22'])
    # sns.set_theme()

    ego_nets = {}

    G_combined = nx.Graph()

    feats_combined = {}

    for i, file in enumerate(glob.glob(input_dir + '/*.edges'), 1):

        no_feat_nodes = []

        if i > max_num_egonets and max_num_egonets != -1:
            break
        else:
            print(f'Processing egonet {i}/{max_num_egonets if max_num_egonets != -1 else len(glob.glob(input_dir + "/*.edges"))}')

        name = file.split('/')[-1].split('.')[0]
        ego_net = nx.read_edgelist(file, nodetype=str, create_using=nx.Graph)
        
        ego_net.add_node(name)

        for n in ego_net.nodes():
            if n != name:
                ego_net.add_edge(name, n)

        featnames_dir = {}

        with open(input_dir + '/' + name + '.featnames') as f:
            featnames = f.read().splitlines()


        for featname in featnames:
            id, f = featname.split(' ', 1)

            # import pdb; pdb.set_trace()
            
            if network_type == 'twitter':
                k, v = f, ''
            elif network_type == 'gplus':
                k, v = f.split(':', 1)
            elif network_type == 'facebook':
                k, v = f.split(';', 1)

            if k not in featnames_dir:
                featnames_dir[int(id)] = (k, v)


        feats_dir = {}
        feats = []

        with open(input_dir + '/' + name + '.egofeat') as f:
            egofeat = [name] + f.read().splitlines()[0].split()
            feats.append(egofeat)

        with open(input_dir + '/' + name + '.feat') as f:
            for feat in f.read().splitlines():
                feats.append(feat.split())

        # with open(input_dir + '/' + name + '.circles') as f:
        #     circles = f.read().splitlines()

        # node2circle = collections.defaultdict(list)

        # for circle in circles:
        #     _, members = circle.split('\t', 1)
        #     members = members.split('\t')
        #     for member in members:
        #         node2circle[member].append(list(set(members) - set([member])))

        feats_freq = collections.defaultdict(int)

        for feat in feats:
            
            if network_type in ['gplus', 'facebook']:
                temp = {}
                for i, f in enumerate(feat[1:]):
                    if int(f) == 1 and featnames_dir[i][0] != 'gender':
                        temp[featnames_dir[i][0]] = featnames_dir[i][1]

                        feats_freq[featnames_dir[i][0]] += 1

            elif network_type == 'twitter':
                temp = {'hashtags': [], 'follows': []}

                for i, f in enumerate(feat[1:]):
                    if int(f) == 1:
                        if featnames_dir[i][0].startswith('#'):
                            temp['hashtags'].append(featnames_dir[i][0])
                        elif featnames_dir[i][0].startswith('@'):
                            temp['follows'].append(featnames_dir[i][0])

                        feats_freq[featnames_dir[i][0]] += 1

            # temp['neighbors'] = list(ego_net.neighbors(feat[0]))
            # temp['social_circles'] = node2circle.get(feat[0], [])

            if len(temp) == 0:
                no_feat_nodes.append(feat[0])
            else:
                feats_dir[feat[0]] = temp

        ego_net.remove_nodes_from(no_feat_nodes)

        # import pdb; pdb.set_trace()

        nx.set_node_attributes(ego_net, feats_dir, 'features')
        
        G_combined = nx.compose(G_combined, ego_net)
        feats_combined = {**feats_combined, **feats_dir}

        if max_graph_size != -1 and G_combined.number_of_nodes() > max_graph_size:
            break

    nx.set_node_attributes(G_combined, feats_combined, 'features')

    # Get largest connected component
    G_combined = G_combined.subgraph(max(nx.connected_components(G_combined), key=len))

    # Plot feat frequency
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))


    # If twitter plot top 10 features
    if network_type == 'twitter':
        top_feats = dict(sorted(feats_freq.items(), key=lambda x: x[1], reverse=True)[:10])

        sns.barplot(x=list(top_feats.keys()), y=list(top_feats.values()), ax=ax[0])
    else:
        sns.barplot(x=list(feats_freq.keys()), y=list(feats_freq.values()), ax=ax[0])

    ax[0].set_title('Feature frequency')
    ax[0].set_xlabel('Feature')

    # rotate x ticks
    for tick in ax[0].get_xticklabels():
        tick.set(rotation=90)

    if network_type == 'twitter':
        jaccard_similarities_hashtags = []
        jaccard_similarities_follows = []

        for (u, v) in G_combined.edges():

            feats_u = G_combined.nodes[u]['features']
            feats_v = G_combined.nodes[v]['features']

            hashtags_u = set(feats_u['hashtags'])
            hashtags_v = set(feats_v['hashtags'])

            follows_u = set(feats_u['follows'])
            follows_v = set(feats_v['follows'])

            # calculate jaccard similarity
            jaccard_similarities_hashtags.append(jaccard(hashtags_u, hashtags_v))
            jaccard_similarities_follows.append(jaccard(follows_u, follows_v))

        sns.histplot(jaccard_similarities_hashtags, ax=ax[1], stat='probability')
        sns.histplot(jaccard_similarities_follows, ax=ax[2], stat='probability')

        ax[1].set_title('Jaccard Sim. Hashtags')
        ax[1].set_xlabel('Jaccard Similarity')

        ax[2].set_title('Jaccard Sim. Follows')
        ax[2].set_xlabel('Jaccard Similarity')


    else:
        similarities = []

        for (u, v) in G_combined.edges():
            feats_u = G_combined.nodes[u]['features']
            feats_v = G_combined.nodes[v]['features']

            # count values that match
            common_feats = sum([1 for k in feats_u if k in feats_v and feats_u[k] == feats_v[k]])

            similarities.append(common_feats)

        sns.histplot(similarities, ax=ax[1], stat='probability', discrete=True)

        ax[1].set_title('Similarity distribution')
        ax[1].set_xlabel('Number of common features')

        # Assortativities with respect to each feature
        features_expanded = collections.defaultdict(dict)

        for u in G_combined.nodes():
            for k in G_combined.nodes[u]['features']:
                features_expanded[k][u] = G_combined.nodes[u]['features'][k]

        assortativities = {}

        for k in features_expanded:
            nx.set_node_attributes(G_combined, features_expanded[k], k)
            assortativities[k] = nx.attribute_assortativity_coefficient(G_combined, k)
        
        sns.barplot(x=list(assortativities), y=list(assortativities.values()), ax=ax[2])

        # rotate x ticks
        for tick in ax[2].get_xticklabels():
            tick.set(rotation=90)

        ax[2].set_title('Assortativities')
        ax[2].set_xlabel('Feature')

    # remove top and right spines
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    fig.suptitle(f'{network_type.title()}') 

    fig.savefig(input_dir + f'/{network_type}_feat_freq_sim.png', bbox_inches='tight')

    # Store graph JSON 
    with open(input_dir + f'/{network_type}_combined.json', 'w+') as f:
        data = nx.node_link_data(G_combined)
        f.write(json.dumps(data))

    return G_combined

def load_facebook100(input_dir, name, num_egonets=10, egonets_radius=2, sample_egonets=True):

    feat_names = ['status', 'gender', 'major', 'second_major', 'accomodation', 'year', 'high_school']

    filename = input_dir + '/' + name + '.mat'
    mat = sio.loadmat(filename)

    if 'A' in mat:
        A = mat['A']

    feats_dir = {}         

    if 'local_info' in mat:
        local_info = mat['local_info']


    for i in range(len(local_info)):
        feat = local_info[i]
        

        temp = {}
        for feat_name, f in zip(feat_names, feat):
            if f != 0:
                if feat_name == 'status':
                    if f == 1:
                        temp[feat_name] = 'student'
                    elif f == 2:
                        temp[feat_name] = 'faculty'
                elif feat_name == 'accomodation':
                    if f == 1:
                        temp[feat_name] = 'house'
                    elif f == 2:
                        temp[feat_name] = 'dorm'
                elif feat_name == 'gender':
                    continue
                    # if f == 1:
                    #     temp[feat_name] = 'male'
                    # elif f == 2:
                    #     temp[feat_name] = 'female'
                else:
                    temp[feat_name] = int(f)

        feats_dir[i] = temp


    G = nx.from_numpy_array(A.toarray())
    nx.set_node_attributes(G, feats_dir, 'features')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    similarities = []

    for (u, v) in G.edges():
        feats_u = G.nodes[u]['features']
        feats_v = G.nodes[v]['features']

        # count values that match
        common_feats = sum([1 for k in feats_u if k in feats_v and feats_u[k] == feats_v[k]])

        similarities.append(common_feats)

    sns.histplot(similarities, ax=ax[0], stat='probability', discrete=True)

    ax[0].set_title('Similarity distribution')
    ax[0].set_xlabel('Number of common features')

    # Assortativities with respect to each feature
    features_expanded = collections.defaultdict(dict)

    for u in G.nodes():
        for k in G.nodes[u]['features']:
            features_expanded[k][u] = G.nodes[u]['features'][k]

    assortativities = {}

    for k in features_expanded:
        nx.set_node_attributes(G, features_expanded[k], k)
        assortativities[k] = nx.attribute_assortativity_coefficient(G, k)
    
    sns.barplot(x=list(assortativities), y=list(assortativities.values()), ax=ax[1])

    # rotate x ticks
    for tick in ax[1].get_xticklabels():
        tick.set(rotation=90)

    ax[1].set_title('Assortativities')
    ax[1].set_xlabel('Feature')


    # remove top and right spines
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    fig.suptitle(f'{name.title()}') 

    fig.savefig(input_dir + f'/{name}_feat_freq_sim.png', bbox_inches='tight')

    if sample_egonets:
        networks = sample_ego_nets(G, n_samples=num_egonets, radius=egonets_radius)
    else:
        networks = {-1: G}
        
    return networks

def sample_ego_nets(G, n_samples=10, radius=2, seed=0):
    """
    Sample n_samples ego nets from G.
    """
    random.seed(seed)
    ego_nets = {}
    nodes = list(G.nodes())
    for i in range(n_samples):
        ego_net = nx.ego_graph(G, random.choice(nodes), radius=radius)
        ego_nets[i] = ego_net

    return ego_nets

if __name__ == '__main__':
    input_dir = 'datasets/facebook'
    ego_nets = preprocess_snap_ego_nets(input_dir, network_type='facebook', ego_nets=True, max_num_egonets=20, max_graph_size=2000)

    # for name in ['Caltech36', 'Swarthmore42', 'UChicago30']:
    #     print(name)
    #     load_facebook100('datasets/facebook100', name, sample_egonets=False, num_egonets=-1, egonets_radius=-1)