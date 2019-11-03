import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
#from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from metrics import *

import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
#setup_seed(100)

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


def reg_scores(label, pred):
	label = label.reshape(-1)
	pred = pred.reshape(-1)
	return rmse(label, pred), pearson(label, pred), spearman(label, pred)


def load_blosum62():
	blosum_dict = {}
	f = open('blosum62.txt')
	lines = f.readlines()
	f.close()
	skip =1 
	for i in lines:
		if skip == 1:
			skip = 0
			continue
		parsed = i.strip('\n').split()
		blosum_dict[parsed[0]] = np.array(parsed[1:]).astype(float)
	return blosum_dict


#padding functions
def pad_label(arr_list, ref_list):
	N = ref_list.shape[1]
	a = np.zeros((len(arr_list), N))
	for i, arr in enumerate(arr_list):
		n = len(arr)
		a[i,0:n] = arr
	return a


def pad_label_2d(label, vertex, sequence):
	dim1 = vertex.size(1)
	dim2 = sequence.size(1)
	a = np.zeros((len(label), dim1, dim2))
	for i, arr in enumerate(label):
		a[i, :arr.shape[0], :arr.shape[1]] = arr
	return a


def pack2D(arr_list):
	N = max([x.shape[0] for x in arr_list])
	M = max_nb#max([x.shape[1] for x in arr_list])
	a = np.zeros((len(arr_list), N, M))
	for i, arr in enumerate(arr_list):
		n = arr.shape[0]
		m = arr.shape[1]
		a[i,0:n,0:m] = arr
	return a


def pack1D(arr_list):
	N = max([x.shape[0] for x in arr_list])
	a = np.zeros((len(arr_list), N))
	for i, arr in enumerate(arr_list):
		n = arr.shape[0]
		a[i,0:n] = arr
	return a


def get_mask(arr_list):
	N = max([x.shape[0] for x in arr_list])
	a = np.zeros((len(arr_list), N))
	for i, arr in enumerate(arr_list):
		a[i,:arr.shape[0]] = 1
	return a


#embedding selection function
def add_index(input_array, ebd_size):
	batch_size, n_vertex, n_nbs = np.shape(input_array)
	add_idx = np.array(range(0,(ebd_size)*batch_size,ebd_size)*(n_nbs*n_vertex))
	add_idx = np.transpose(add_idx.reshape(-1,batch_size))
	add_idx = add_idx.reshape(-1)
	new_array = input_array.reshape(-1)+add_idx
	return new_array

#function for generating batch data
def batch_data_process(data):
	vertex, edge, atom_adj, bond_adj, nbs, sequence = data
	
	vertex_mask = get_mask(vertex)
	vertex = pack1D(vertex)
	edge = pack1D(edge)
	atom_adj = pack2D(atom_adj)
	bond_adj = pack2D(bond_adj)
	nbs_mask = pack2D(nbs)
	
	#pad proteins and make masks
	seq_mask = get_mask(sequence)
	sequence = pack1D(sequence+1)
	
	#add index
	atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
	bond_adj = add_index(bond_adj, np.shape(edge)[1])
	
	#convert to torch cuda data type
	vertex_mask = Variable(torch.FloatTensor(vertex_mask)).cuda()
	vertex = Variable(torch.LongTensor(vertex)).cuda()
	edge = Variable(torch.LongTensor(edge)).cuda()
	atom_adj = Variable(torch.LongTensor(atom_adj)).cuda()
	bond_adj = Variable(torch.LongTensor(bond_adj)).cuda()
	nbs_mask = Variable(torch.FloatTensor(nbs_mask)).cuda()
	
	seq_mask = Variable(torch.FloatTensor(seq_mask)).cuda()
	sequence = Variable(torch.LongTensor(sequence)).cuda()
	
	return vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence


# load data
def data_from_index(data_pack, idx_list):
	fa, fb, anb, bnb, nbs_mat, seq_input = [data_pack[i][idx_list] for i in range(6)]
	aff_label = data_pack[6][idx_list].astype(float).reshape(-1,1)
	#cid, pid = [data_pack[i][idx_list] for i in range(7,9)]
	pairwise_mask = data_pack[9][idx_list].astype(float).reshape(-1,1)
	pairwise_label = data_pack[10][idx_list]
	return [fa, fb, anb, bnb, nbs_mat, seq_input, aff_label, pairwise_mask, pairwise_label]


def split_train_test_clusters(measure, clu_thre, n_fold):
	# load cluster dict
	cluster_path = '../preprocessing/'
	with open(cluster_path+measure+'_compound_cluster_dict_'+str(clu_thre), 'rb') as f:
		C_cluster_dict = pickle.load(f)
	with open(cluster_path+measure+'_protein_cluster_dict_'+str(clu_thre), 'rb') as f:
		P_cluster_dict = pickle.load(f)
	
	C_cluster_set = set(list(C_cluster_dict.values()))
	P_cluster_set = set(list(P_cluster_dict.values()))
	C_cluster_list = np.array(list(C_cluster_set))
	P_cluster_list = np.array(list(P_cluster_set))
	np.random.shuffle(C_cluster_list)
	np.random.shuffle(P_cluster_list)
	# n-fold split
	c_kf = KFold(len(C_cluster_list), n_fold, shuffle=True)
	p_kf = KFold(len(P_cluster_list), n_fold, shuffle=True)
	#c_kf = KFold(n_fold,shuffle=True)
	#p_kf = KFold(n_fold,shuffle=True)
	c_train_clusters, c_test_clusters = [], []
	for train_idx, test_idx in c_kf: #.split(C_cluster_list):
		c_train_clusters.append(C_cluster_list[train_idx])
		c_test_clusters.append(C_cluster_list[test_idx])
	p_train_clusters, p_test_clusters = [], []
	for train_idx, test_idx in p_kf: #.split(P_cluster_list):
		p_train_clusters.append(P_cluster_list[train_idx])
		p_test_clusters.append(P_cluster_list[test_idx])
	
	
	#pair_kf = KFold(n_fold,shuffle=True)
	pair_list = []
	for i_c in C_cluster_list:
		for i_p in P_cluster_list:
			pair_list.append('c'+str(i_c)+'p'+str(i_p))
	pair_list = np.array(pair_list)
	np.random.shuffle(pair_list)
	pair_kf = KFold(len(pair_list), n_fold, shuffle=True)
	pair_train_clusters, pair_test_clusters = [], []
	for train_idx, test_idx in pair_kf: #.split(pair_list):
		pair_train_clusters.append(pair_list[train_idx])
		pair_test_clusters.append(pair_list[test_idx])
	
	return pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


def load_data(measure, setting, clu_thre, n_fold):
	# load data
	with open('../preprocessing/pdbbind_all_combined_input_'+measure,'rb') as f:
		data_pack = pickle.load(f)
	cid_list = data_pack[7]
	pid_list = data_pack[8]
	n_sample = len(cid_list)
	
	# train-test split
	train_idx_list, valid_idx_list, test_idx_list = [], [], []
	print 'setting:', setting
	if setting == 'imputation':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			pair_train_valid, pair_test = pair_train_clusters[fold], pair_test_clusters[fold]
			pair_valid = np.random.choice(pair_train_valid, int(len(pair_train_valid)*0.125), replace=False)
			pair_train = set(pair_train_valid)-set(pair_valid)
			pair_valid = set(pair_valid)
			pair_test = set(pair_test)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample):
				if 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_train:
					train_idx.append(ele)
				elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_valid:
					valid_idx.append(ele)
				elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			print 'fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx)
			
	elif setting == 'new_protein':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
			p_valid = np.random.choice(p_train_valid, int(len(p_train_valid)*0.125), replace=False)
			p_train = set(p_train_valid)-set(p_valid)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample): 
				if P_cluster_dict[pid_list[ele]] in p_train:
					train_idx.append(ele)
				elif P_cluster_dict[pid_list[ele]] in p_valid:
					valid_idx.append(ele)
				elif P_cluster_dict[pid_list[ele]] in p_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			print 'fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx)
			
	elif setting == 'new_compound':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
			c_valid = np.random.choice(c_train_valid, int(len(c_train_valid)*0.125), replace=False)
			c_train = set(c_train_valid)-set(c_valid)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample):
				if C_cluster_dict[cid_list[ele]] in c_train:
					train_idx.append(ele)
				elif C_cluster_dict[cid_list[ele]] in c_valid:
					valid_idx.append(ele)
				elif C_cluster_dict[cid_list[ele]] in c_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			print 'fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx)
	
	elif setting == 'new_new':
		assert n_fold ** 0.5 == int(n_fold ** 0.5)
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))
		
		for fold_x in range(int(n_fold ** 0.5)):
			for fold_y in range(int(n_fold ** 0.5)):
				c_train_valid, p_train_valid = c_train_clusters[fold_x], p_train_clusters[fold_y]
				c_test, p_test = c_test_clusters[fold_x], p_test_clusters[fold_y]
				c_valid = np.random.choice(list(c_train_valid), int(len(c_train_valid)/3), replace=False)
				c_train = set(c_train_valid)-set(c_valid)
				p_valid = np.random.choice(list(p_train_valid), int(len(p_train_valid)/3), replace=False)
				p_train = set(p_train_valid)-set(p_valid)
				
				train_idx, valid_idx, test_idx = [], [], []
				for ele in range(n_sample):
					if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
						train_idx.append(ele)
					elif C_cluster_dict[cid_list[ele]] in c_valid and P_cluster_dict[pid_list[ele]] in p_valid:
						valid_idx.append(ele)
					elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
						test_idx.append(ele)
				train_idx_list.append(train_idx)
				valid_idx_list.append(valid_idx)
				test_idx_list.append(test_idx)
				print 'fold', fold_x*int(n_fold ** 0.5)+fold_y, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx)
	return data_pack, train_idx_list, valid_idx_list, test_idx_list


# network utils
def loading_emb(measure):
	#load intial atom and bond features (i.e., embeddings)
	f = open('../preprocessing/pdbbind_all_atom_dict_'+measure)
	atom_dict = pickle.load(f)
	f.close()
	
	f = open('../preprocessing/pdbbind_all_bond_dict_'+measure)
	bond_dict = pickle.load(f)
	f.close()
	
	f = open('../preprocessing/pdbbind_all_word_dict_'+measure)
	word_dict = pickle.load(f)
	f.close()
	
	print 'atom dict size:', len(atom_dict), ', bond dict size:', len(bond_dict), ', word dict size:', len(word_dict)
	
	init_atom_features = np.zeros((len(atom_dict), atom_fdim))
	init_bond_features = np.zeros((len(bond_dict), bond_fdim))
	init_word_features = np.zeros((len(word_dict), 20))
	
	for key,value in atom_dict.items():
		init_atom_features[value] = np.array(list(map(int, key)))
	
	for key,value in bond_dict.items():
		init_bond_features[value] = np.array(list(map(int, key)))
	
	blosum_dict = load_blosum62()
	for key, value in word_dict.items():
		if key not in blosum_dict:
			continue
		init_word_features[value] = blosum_dict[key] #/ float(np.sum(blosum_dict[key]))
	init_atom_features = Variable(torch.FloatTensor(init_atom_features)).cuda()
	init_bond_features = Variable(torch.FloatTensor(init_bond_features)).cuda()
	init_word_features = Variable(torch.cat((torch.zeros(1,20),torch.FloatTensor(init_word_features)),dim=0)).cuda()
	
	return init_atom_features, init_bond_features, init_word_features


#Model parameter intializer
def weights_init(m):
	if isinstance(m, nn.Conv1d) or isinstance(m,nn.Linear):
		nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
		nn.init.constant_(m.bias, 0)


#Custom loss
class Masked_BCELoss(nn.Module):
	def __init__(self):
		super(Masked_BCELoss, self).__init__()
		self.criterion = nn.BCELoss(reduce=False)
	def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
		batch_size = pred.size(0)
		loss_all = self.criterion(pred, label)
		loss_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))*pairwise_mask.view(-1, 1, 1)
		loss = torch.sum(loss_all*loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)
		return loss

