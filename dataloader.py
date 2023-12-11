import numpy as np
import collections
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import glob
import pandas as pd
import os

def load_ego_nets(input_dir, network_type='gplus'):
    """
    Load ego nets from files in input_dir.
    Returns a dictionary of networkx graphs.
    """
    ego_nets = {}
    for file in glob.glob(input_dir + '/*.edges'):
        name = file.split('/')[-1].split('.')[0]
        if network_type in ['twitter', 'gplus']:
            ego_net = nx.read_edgelist(file, nodetype=str, create_using=nx.DiGraph)
        elif network_type == 'facebook':
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
            
            if network_type == 'twitter':
                k, v = f, ''
            else:
                k, v = f.split(':', 1)

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

        with open(input_dir + '/' + name + '.circles') as f:
            circles = f.read().splitlines()

        node2circle = collections.defaultdict(list)

        for circle in circles:
            _, members = circle.split('\t', 1)
            members = members.split('\t')
            for member in members:
                node2circle[member].append(list(set(members) - set([member])))

        for feat in feats:
            
            if network_type in ['gplus', 'facebook']:
                temp = {}
                for i, f in enumerate(feat[1:]):
                    if int(f) == 1:
                        temp[featnames_dir[i][0]] = featnames_dir[i][1]

            elif network_type == 'twitter':
                temp = {'hashtags': [], 'follows': []}

                for i, f in enumerate(feat[1:]):
                    if int(f) == 1:
                        if featnames_dir[i][0].startswith('#'):
                            temp['hashtags'].append(featnames_dir[i][0])
                        elif featnames_dir[i][0].startswith('@'):
                            temp['follows'].append(featnames_dir[i][0])


            # temp['neighbors'] = list(ego_net.neighbors(feat[0]))
            temp['social_circles'] = node2circle.get(feat[0], [])

            feats_dir[feat[0]] = temp

        nx.set_node_attributes(ego_net, feats_dir, 'features')

        ego_nets[name] = ego_net

    ego_nets['combined'] = nx.compose_all(list(ego_nets.values()))

    nx.write_gml(ego_nets['combined'], f'datasets/{network_type}_combined.gml')

    return ego_nets

def load_facebook100(input_dir):

    networks = {}

    feat_names = ['status', 'gender', 'major', 'second_major', 'accomodation', 'year', 'high_school']

    for file in glob.glob(input_dir + '/*.mat'):

        name = file.split('/')[-1].split('.')[0]
        mat = sio.loadmat(file)

        if 'A' in mat:
            A = mat['A']


        feats_dir = {}            

        if 'local_info' in mat:
            local_info = mat['local_info']

        print(local_info.shape)

        for i in range(len(local_info)):
            feat = local_info[i]
            
            print(feat.shape)

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
                        if f == 1:
                            temp[feat_name] = 'male'
                        elif f == 2:
                            temp[feat_name] = 'female'
                    else:
                        temp[feat_name] = f
    
            feats_dir[i] = temp

        import pdb; pdb.set_trace()

        G = nx.from_numpy_matrix(A)
        networks[name] = G

    return networks

def load_insurance_dataset(input_dir):

    dataset_dir = os.path.join(input_dir, 'data', 'data')

    df = pd.read_stata(os.path.join(dataset_dir, '0422allinforawnet.dta'))

    df1 = pd.read_stata(os.path.join(dataset_dir, '0422structure_all.dta'))

    df2 = pd.read_stata(os.path.join(dataset_dir, '0422price.dta'))

    df3 = pd.read_stata(os.path.join(dataset_dir, '0422survey.dta'))

    import pdb; pdb.set_trace()

load_insurance_dataset('datasets/insurance-dataset')