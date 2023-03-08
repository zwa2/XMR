import numpy as np
import torch
import torch.utils.data as du
import torch.nn as nn
import os
from util import *
from xmr import *
import argparse
import gc

# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

        term_mask_map[term] = mask_gpu

    return term_mask_map

# solution for 1/2||x-y||^2_2 + c||x||_0
def proximal_l0(yvec, c):
    yvec_abs =  torch.abs(yvec)
    csqrt = torch.sqrt(2*c)
    
    xvec = (yvec_abs>=csqrt).float()*yvec
    return xvec

# solution for 1/2||x-y||^2_2 + c||x||_g
def proximal_glasso_nonoverlap(yvec, c):
    ynorm = torch.norm(yvec, p='fro')
    if ynorm > c:
        xvec = (yvec/ynorm)*(ynorm-c)
    else:
        xvec = torch.zeros_like(yvec)
    return xvec

# solution for ||x-y||^2_2 + c||x||_2^2
def proximal_l2(yvec, c):
    return (1./(1.+c))*yvec

# prune the structure by palm
def optimize_palm(model, dG, root, reg_l0, reg_glasso, reg_decay, lr=0.001, lip=0.001):
    dG_prune = dG.copy()
    for name, param in model.named_parameters():
        if "direct" in name:
            # mutation side
            # l0 for direct edge from gene to term
            param_tmp = param.data - lip*param.grad.data
            param_tmp2 = proximal_l0(param_tmp, torch.tensor(reg_l0*lip).cuda(CUDA_ID))
            param.data = param_tmp2
        elif "GO_linear_layer" in name:
            # group lasso for
            dim = model.num_hiddens_genotype
            term_name = name.split('_')[0]
            child = model.term_neighbor_map[term_name]
            for i in range(len(child)):
                term_input = param.data[:,i*dim:(i+1)*dim]
                term_input_grad = param.grad.data[:,i*dim:(i+1)*dim]
                term_input_tmp = term_input - lip*term_input_grad
                term_input_update = proximal_glasso_nonoverlap(term_input_tmp, reg_glasso*lip)
                param.data[:,i*dim:(i+1)*dim] = term_input_update
                num_n0 =  len(torch.nonzero(term_input_update, as_tuple =False))
                if num_n0 == 0 :
                    dG_prune.remove_edge(term_name, child[i])
            # weight decay for direct
            direct_input = param.data[:,len(child)*dim:]
            direct_input_grad = param.grad.data[:,len(child)*dim:]
            direct_input_tmp = direct_input - lr*direct_input_grad
            direct_input_update = proximal_l2(direct_input_tmp, reg_decay)
            param.data[:,len(child)*dim:] = direct_input_update
        else:
            # other param weigth decay
            param_tmp = param.data - lr*param.grad.data
            param.data = proximal_l2(param_tmp, 2*reg_decay*lr)
    
    del param_tmp, param_tmp2, child, term_input, term_input_grad, term_input_tmp, term_input_update
    del direct_input, direct_input_grad, direct_input_tmp, direct_input_update
    
# check network statisics
def check_network(model, dG, root):
    dG_prune = dG.copy()
    for name, param in model.named_parameters():
        if "GO_linear_layer" in name:
            # group lasso for
            dim = model.num_hiddens_genotype
            term_name = name.split('_')[0]
            child = model.term_neighbor_map[term_name]
            for i in range(len(child)):
                term_input = param.data[:,i*dim:(i+1)*dim]
                num_n0 =  len(torch.nonzero(term_input, as_tuple =False))
                if num_n0 == 0 :
                    dG_prune.remove_edge(term_name, child[i])
    
    print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
    NodesLeft = list()
    for nodetmp in dG_prune.nodes:
        for path in nx.all_simple_paths(dG_prune, source=root, target=nodetmp):
            NodesLeft.extend(path)
    NodesLeft = list(set(NodesLeft))
    sub_dG_prune = dG_prune.subgraph(NodesLeft)
    print("Pruned graph has %d nodes and %d edges" % (sub_dG_prune.number_of_nodes(), sub_dG_prune.number_of_edges()))
    
    return sub_dG_prune


def training_acc(model, optimizer, train_loader, train_label_gpu, gene_dim, cuda_cells, drug_dim, cuda_drugs, CUDA_ID):
    #Train
    model.train()
    train_predict = torch.zeros(0,0).cuda(CUDA_ID)

    for i, (inputdata, labels) in enumerate(train_loader):
        cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer

        cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
        cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)
        
        print(i)
        # Here term_NN_out_map is a dictionary
        aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

        if train_predict.size()[0] == 0:
            train_predict = aux_out_map['final'].data
        else:
            train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

        total_loss = 0
        for name, output in aux_out_map.items():
            loss = nn.MSELoss()
            if name == 'final':
                total_loss += loss(output, cuda_labels)
            else: # change 0.2 to smaller one for big terms
                total_loss += 0.2 * loss(output, cuda_labels)
        print(i, total_loss)
        
    train_corr = spearman_corr(train_predict, train_label_gpu)
        
    print("pretrained model %f total loss, %f training acc" % (total_loss, train_corr))
        
        
def test_acc(model, test_loader, batch_size, drug_test, test_label_gpu, gene_dim, cuda_cells, CUDA_ID):
    model.eval()
    test_predict = torch.zeros(0,0).cuda(CUDA_ID)

    for i, (inputdata, labels) in enumerate(test_loader):
        # Convert torch tensor to Variable
        cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
        cuda_cell_features.cuda(CUDA_ID)

        drugdata = list(zip(*drug_test[i*batch_size:(i+1)*batch_size]))
        aux_out_map, _ = model(cuda_cell_features, drugdata)

        if test_predict.size()[0] == 0:
            test_predict = aux_out_map['final'].data
        else:
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

    test_corr = spearman_corr(test_predict, test_label_gpu)
    del aux_out_map, inputdata, labels, test_predict, cuda_cell_features
    torch.cuda.empty_cache()
    
    return test_corr
    
def grad_hook_masking(grad, mask):
    grad = grad.mul_(mask)
    del mask

def sparse_direct_gene(model, GOlist):
    GO_direct_spare_gene = {}
    for go in GOlist:
        GO_direct_spare_gene[go] = list()
    
    preserved_gene = list()
    for name, param in model.named_parameters():
        if "direct" in name:
            GOname = name.split('_')[0]
            if GOname in GOlist:
                param_tmp = torch.sum(param.data, dim=0)
                indn0 = torch.nonzero(param_tmp, as_tuple=True)[0]
                indn0 = [indn0[i].item() for i in range(len(indn0))]
                GO_direct_spare_gene[GOname].extend(indn0)
                preserved_gene.extend(indn0)
                
    return GO_direct_spare_gene, preserved_gene

# train a DrugCell model 
def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_final_drug, num_hiddens_final, cell_features, drug_train, drug_test, N_fingerprints, pathway):

    pathway = pathway.split("/")[-1]
    pathway = pathway.split(".")[0]
    print("Pathway: ", pathway)

    dGc = dG.copy()

    # separate the whole data into training and test data
    train_feature, train_label, test_feature, test_label = train_data

    # copy labels (observation) to GPU - will be used to
    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))
    
    # create dataloader for training/test data
    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)
    
    # create a torch objects containing input features for cell lines and drugs
    cuda_cells = torch.from_numpy(cell_features)
    
    # dcell neural network
    model = XMR(N_fingerprints, term_size_map, term_direct_gene_map, dG, gene_dim, root, num_hiddens_genotype, num_hiddens_drug, num_final_drug, num_hiddens_final, CUDA_ID)
    
    # load model to GPU
    model.cuda(CUDA_ID)

    # define optimizer
    # optimize drug NN
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)
        
    # load pretrain model
    pretrained_model = f"{model_save_folder}/{pathway}_best.pkl"
    
    if os.path.isfile(pretrained_model):
        print("Pre-trained model exists:" + pretrained_model)
        model.load_state_dict(torch.load(pretrained_model,map_location=torch.device('cuda', CUDA_ID))) #param_file
        base_test_acc = test_acc(model, test_loader, batch_size, drug_test, test_label_gpu, gene_dim, cuda_cells, CUDA_ID)
        results = base_test_acc.cpu().detach().numpy()
        print("Pre Acc.: ", results)
        # sys.exit()
    else:
        print("Pre-trained model does not exist, so before pruning we have to pre-train a model.")
        sys.exit()
    
    PrunedG = check_network(model, dGc, root)
    GOLeft = list(PrunedG.nodes)
    GOLeft_num = len(GOLeft)
    GO_direct_gene, Prev_gene_tmp = sparse_direct_gene(model, GOLeft)
    Prev_gene = [Prev_gene_tmp[i] for i in range(len(Prev_gene_tmp))]
    Prev_gene_unique = list(set(Prev_gene))
    NumGeneLeft = len(Prev_gene_unique)
    print("Gene num: ", NumGeneLeft)
    print("Prev_gene_unique: ", Prev_gene_unique)

    reg_adj_dict = {
        "Cell_Cycle": 500,
        "Disease": 1000,
        "DNA_Repair": 500,
        "Signal_Transduction": 1000,
        "Metabolism": 1000,
    }

    reg_l0 = NumGeneLeft / reg_adj_dict[pathway]
    reg_glasso = GOLeft_num / reg_adj_dict[pathway]

    best_retrain_corr = base_test_acc
    for epoch in range(train_epochs):

        # prune step
        for prune_epoch in range(10):
	        #Train
            model.train()
            train_predict = torch.zeros(0,0).cuda(CUDA_ID)

            for i, (inputdata, labels) in enumerate(train_loader):
                cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))
                
	            # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer

                cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
                cuda_cell_features.cuda(CUDA_ID)

	            # Here term_NN_out_map is a dictionary
                drugdata = list(zip(*drug_train[i*batch_size:(i+1)*batch_size]))
                aux_out_map, _ = model(cuda_cell_features, drugdata)

                if train_predict.size()[0] == 0:
                    train_predict = aux_out_map['final'].data
                else:
                    train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

                total_loss = 0
                for name, output in aux_out_map.items():
                    loss = nn.MSELoss()
                    if name == 'final':
                        total_loss += loss(output, cuda_labels)
                    else: # change 0.2 to smaller one for big terms
                        total_loss += 0.2 * loss(output, cuda_labels)

                total_loss.backward()

                for name, param in model.named_parameters():
                    if '_direct_gene_layer.weight' not in name:
                        continue
                    term_name = name.split('_')[0]
                    param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
                    param.data = torch.mul(param.data, term_mask_map[term_name])
                
                optimize_palm(model, dGc, root, reg_l0=reg_l0, reg_glasso = reg_glasso, reg_decay=0.001, lr=0.001, lip=0.001)
                
            del total_loss, cuda_cell_features
            del aux_out_map, inputdata, labels
            torch.cuda.empty_cache()

            train_corr = spearman_corr(train_predict, train_label_gpu)
            prune_test_corr = test_acc(model, test_loader, batch_size, drug_test, test_label_gpu, gene_dim, cuda_cells, CUDA_ID)
            print(">>>>>%d epoch run Pruning step %d: model train acc %f test acc %f" % (epoch, prune_epoch, train_corr, prune_test_corr))
            del train_predict, prune_test_corr
            torch.cuda.empty_cache()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    
        for retain_epoch in range(10):
            handle_list = list()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "direct" in name:
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        handle_list.append(handle)
                    if "GO_linear_layer" in name:
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        handle_list.append(handle)
            
            model.train()
            train_predict = torch.zeros(0,0).cuda(CUDA_ID)

            for i, (inputdata, labels) in enumerate(train_loader):
                cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

                optimizer.zero_grad()

                cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
                cuda_cell_features.cuda(CUDA_ID)

                drugdata = list(zip(*drug_train[i*batch_size:(i+1)*batch_size]))
                aux_out_map, _ = model(cuda_cell_features, drugdata)

                if train_predict.size()[0] == 0:
                    train_predict = aux_out_map['final'].data
                else:
                    train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

                total_loss = 0
                for name, output in aux_out_map.items():
                    loss = nn.MSELoss()
                    if name == 'final':
                        total_loss += loss(output, cuda_labels)
                    else:
                        total_loss += 0.2 * loss(output, cuda_labels)
                optimizer.zero_grad()
                total_loss.backward()

                for name, param in model.named_parameters():
                    if '_direct_gene_layer.weight' not in name:
                        continue
                    term_name = name.split('_')[0]
                    param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
                    param.data = torch.mul(param.data, term_mask_map[term_name])

                optimizer.step()
                
            del total_loss, cuda_cell_features
            del aux_out_map, inputdata, labels
            torch.cuda.empty_cache()

            for handle in handle_list:
                handle.remove()
            torch.cuda.empty_cache()

            gc.collect()

            train_corr = spearman_corr(train_predict, train_label_gpu)
            retrain_test_corr = test_acc(model, test_loader, batch_size, drug_test, test_label_gpu, gene_dim, cuda_cells, CUDA_ID)
            print(">>>>>%d epoch Retraining step %d: model training acc %f test acc %f" % (epoch, retain_epoch, train_corr, retrain_test_corr))

        PrunedG = check_network(model, dGc, root)
        GOLeft = list(PrunedG.nodes)
        GOLeft_num = len(GOLeft)
        print("GO num: ", GOLeft_num)
        if GOLeft_num == 0:
            exit(0)
        GO_direct_gene, Prev_gene_tmp = sparse_direct_gene(model, GOLeft)
        Prev_gene = [Prev_gene_tmp[i] for i in range(len(Prev_gene_tmp))]
        Prev_gene_unique = list(set(Prev_gene))
        NumGeneLeft = len(Prev_gene_unique)
        print("Gene num: ", NumGeneLeft)

        # save models
        if best_retrain_corr < retrain_test_corr:
            best_retrain_corr = retrain_test_corr
            print("Prev_gene_unique: ", Prev_gene_unique)

            if GOLeft_num < 15:
                print("GOLeft: ", GOLeft)
                print(nx.to_dict_of_dicts(PrunedG))
                print("GO_direct_gene: ", GO_direct_gene)
            
            print("Pathway: ", pathway)
            torch.save(model.state_dict(), f"./{model_save_folder}/{pathway}_pruned.pkl")
        



parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=200)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.005)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=200)
parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)
parser.add_argument('-cancer_type', help='Cancer type', type=str)
parser.add_argument('-genotype_hiddens', help='The number of neurons in each VNN layer', type=int, default=3)
parser.add_argument('-drug_hiddens', help='The number of neurons in each GNN layer', type=int, default=512)
parser.add_argument('-drug_final_hiddens', help='The number of neurons of GNN final layer', type=int, default=3)
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=6)
parser.add_argument('-cellline', help='Mutation information for cell lines', type=str)

print("Start....")

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=3)

# load input data
cat = opt.cancer_type
train_file = f"./data/{cat}_train.txt"
test_file = f"./data/{cat}_val.txt"
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(train_file, test_file, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)
print('Total number of genes = %d' % len(gene2id_mapping))

cell_features = np.genfromtxt(opt.cellline, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = opt.drug_hiddens
num_final_drug = opt.drug_final_hiddens
num_hiddens_final = opt.final_hiddens

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')

print('-'*100)
print('Preprocessing the dataset.')
drug_train, drug_test, N_fingerprints = create_datasets(device, cat)
print('-'*100)
print('The preprocess has finished!')

CUDA_ID = opt.cuda

train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, opt.modeldir, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_final_drug, num_hiddens_final, cell_features, drug_train, drug_test, N_fingerprints, opt.onto)
