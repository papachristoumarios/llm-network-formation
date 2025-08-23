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
import powerlaw as pwl
import matplotlib


def plot_stats(G, title):

    sns.set_palette(['#2980b9', '#f1c40f', '#7f8c8d', '#d35400', '#34495e', '#e67e22'])

    MEDIUM_SIZE = 24
    SMALL_SIZE = 0.85 * MEDIUM_SIZE
    BIGGER_SIZE = 1.5 * MEDIUM_SIZE
    SMALLEST_SIZE = 0.5 * MEDIUM_SIZE

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLEST_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLEST_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    degress = [G.degree(x) for x in G.nodes()]

    powerlaw_fit = pwl.Fit(degress, discrete=True)

    ax[0].set_title('Degree Distribution')

    powerlaw_fit.plot_ccdf(linewidth=3, ax=ax[0], color='#e74c3c', label='Empirical')
    powerlaw_fit.power_law.plot_ccdf(ax=ax[0], color='#e74c3c', linestyle='--', label='Fit')

    ax[0].legend()
    ax[0].xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))

    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.setp(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')


    clustering_coeffs = np.array(list(nx.clustering(G).values()))

    # histogram with matplotlib

    ax[1].hist(clustering_coeffs, bins=20, alpha=0.75, color='#2ecc71')

    ax[1].set_title('Clustering Coefficient')
    # ax[1].set_xlabel('Clustering Coeff.')


    # Assortativities with respect to each feature
    features_expanded = collections.defaultdict(dict)

    for u in G.nodes():
        for k in G.nodes[u]['features']:
            features_expanded[k][u] = G.nodes[u]['features'][k]

    assortativities = {}

    for k in features_expanded:
        nx.set_node_attributes(G, features_expanded[k], k)
        assortativities[k] = nx.attribute_assortativity_coefficient(G, k)
    
    sns.barplot(x=list(assortativities), y=list(assortativities.values()), ax=ax[2])

    # rotate x ticks
    for tick in ax[2].get_xticklabels():
        tick.set(rotation=45)

    ax[2].set_title('Assortativities')

    # remove top and right spines
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    fig.suptitle(title, y=1.05) 

    fig.savefig( f'figures/{title}_stats.png', bbox_inches='tight', dpi=300)

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

def preprocess_snap_ego_nets(input_dir, network_type='gplus', ego_nets=False, max_num_egonets=-1):
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

    # plot_stats(G, f'Facebook100 {name.title()}')

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

def load_andorra(input_dir):
    df_edges = pd.read_csv(input_dir + '/andorra.txt', sep=' ', header=None)
    df_attributes = pd.read_csv(input_dir + '/andorra_attributes_bin.txt', sep=' ', names=['Phone Type', 'Location', 'Usage'], header=None)

    phone_types_dict = {0: 'Apple', 1: 'Samsung', 2: 'Other'}
    usage_dict = {0: 'High', 1: 'Low'}
    
    df_attributes['Phone Type'] = df_attributes['Phone Type'].apply(lambda x: phone_types_dict[x])
    df_attributes['Usage'] = df_attributes['Usage'].apply(lambda x: usage_dict[x])
    
    G = nx.from_pandas_edgelist(df_edges, source=0, target=1)

    nx.set_node_attributes(G, df_attributes.to_dict('index'), 'features')   

    # plot_stats(G, 'Andorra') 

    # print(nx.info(G))

    return {-1: G}

def load_mobiled(input_dir):
    df_edges = pd.read_csv(input_dir + '/mobiled.txt', sep=' ', header=None)
    df_attributes = pd.read_csv(input_dir + '/mobiled_attributes.txt', sep=' ', names=['Employee Type'], header=None)

    employee_types_dict = {0: 'Manager', 1: 'Employee'}
    df_attributes['Employee Type'] = df_attributes['Employee Type'].apply(lambda x: employee_types_dict[x])

    G = nx.from_pandas_edgelist(df_edges, source=0, target=1)

    nx.set_node_attributes(G, df_attributes.to_dict('index'), 'features')   

    # plot_stats(G, 'MobileD') 

    # print(nx.info(G))

    return {-1: G}


if __name__ == '__main__':
    load_andorra('datasets/andorra')
    load_mobiled('datasets/mobiled')

    for name in ['Caltech36', 'Swarthmore42', 'UChicago30']:
        load_facebook100('datasets/facebook100', name, num_egonets=10, egonets_radius=2, sample_egonets=True)