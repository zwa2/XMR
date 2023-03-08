import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import torch
from rdkit import Chem
from collections import defaultdict
from molecularGNN_smiles.main.preprocess import create_atoms, create_ijbonddict, extract_fingerprints, split_dataset
import numpy as np

def spearman_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))
        

def load_ontology(file_name, gene2id_mapping):

    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}

    file_handle = open(file_name)

    gene_set = set()

    for line in file_handle:

        line = line.rstrip().split()
        if len(line) == 0:
            continue
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        else:
            if line[1] not in gene2id_mapping:
                continue

            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[ line[0] ] = set()

            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

            gene_set.add(line[1])

    file_handle.close()

    print('There are %d genes' % len(gene_set))

    fin = False
    while not fin:
        fin = True
        for term in dG.nodes():
        
            term_gene_set = set()

            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]

            deslist = nxadag.descendants(dG, term)

            for child in deslist:
                if child in term_direct_gene_map:
                    term_gene_set = term_gene_set | term_direct_gene_map[child]

            # jisoo
            if len(term_gene_set) == 0:
                dG.remove_node(term)
                fin = False
                break

    for term in dG.nodes():

        term_gene_set = set()

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nxadag.descendants(dG, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        # jisoo
        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term: %s' % term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)


    # leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
    leaves = [n for n,d in dG.in_degree() if d==0]
    # leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print('There are %d roots: %s' % (len(leaves), leaves[0]))
    print('There are %d terms' % len(dG.nodes()))
    print('There are %d connected components' % len(connected_subG_list))

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, leaves[0], term_size_map, term_direct_gene_map


def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split(' ')

            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([float(tokens[2])])

    return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping



def build_input_vector(row_data, num_col, original_features):

    cuda_features = torch.zeros(len(row_data), num_col)

    for i in range(len(row_data)):
        data_ind = row_data[i]
        cuda_features.data[i] = original_features.data[data_ind]
   
    return cuda_features


def create_datasets(device, cat, radius=2):

    dir_dataset = './data/'
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):

        print(filename)
        
        with open(dir_dataset + filename, 'r') as f:
            data_original = f.read().strip().split('\n')

        dataset = []

        for data in data_original:
            _, smiles, property = data.strip().split()

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            property = torch.FloatTensor([[float(property)]]).to(device)

            dataset.append((fingerprints, adjacency, molecular_size, property))

        return dataset

    dataset_train = create_dataset(f'{cat}_train.txt')
    dataset_test = create_dataset(f'{cat}_val.txt')

    N_fingerprints = len(fingerprint_dict)

    return dataset_train, dataset_test, N_fingerprints

