import numpy as np
import pickle
# ============================ pdbbind ======================
def read_fasta():
	uniprot_seq_dict = {}
	f = open('out6.2_target_uniprot_pdbbind.fasta')
	for line in f.readlines():
		if line[0] == '>':
			pdbid = line.split('_')[0][1:]
			name = line.strip().split('_')[-1]
		else:
			seq = line.strip()
			uniprot_seq_dict[pdbid] = (seq,name)
	f.close()
	return uniprot_seq_dict

def get_pdbid_to_ligand():
	pdbid_to_ligand = {}
	with open('./pdbbind_index/INDEX_general_PL.2018') as f:
		for line in f.readlines():
			if line[0] != '#':
				ligand = line.strip().split('(')[1].split(')')[0]
				if '-mer' in ligand:
					continue
				elif '/' in ligand:
					ligand = ligand.split('/')[0]
				if len(ligand) != 3:
					#print(line[:4], ligand)
					continue
				pdbid_to_ligand[line[:4]] = ligand
	print('pdbid_to_ligand',len(pdbid_to_ligand))
	return pdbid_to_ligand
pdbid_to_ligand = get_pdbid_to_ligand()
	
def get_result_dict():
	result_dict = {}
	f = open('./smith-waterman-src/out6.4_pdbbind_align.txt')
	i = -1
	seq_target, seq_query, align = '', '', ''
	pdb_ratio_dict = {}
	for line in f.readlines():
		i += 1
		if i%4 == 0:
			if 'target_name' in line:
				if len(seq_target) != 0:
					result_dict[target_name] = (seq_target, seq_query, align, target_start, query_start)
				target_name = line.strip().split(' ')[-1]
				#print('target_name',target_name)
				seq_target, seq_query, align = '', '', ''
			else:
				seq_target += line.split('\t')[1]
				#print('seq_target',seq_target)
		elif i%4 == 1:
			if 'query_name' in line:
				query_name = line.strip().split(' ')[-1]
				#print('query_name',query_name)
			else:
				align += line.strip('\n').split('\t')[1]
				#print('align',align)
		elif i%4 == 2:
			if 'optimal_alignment_score' in line:
				for item in line.strip().split('\t'):
					if item.split(' ')[0] == 'target_begin:':
						target_start = int(item.split(' ')[1])
					elif item.split(' ')[0] == 'query_begin:':
						query_start = int(item.split(' ')[1])
				#print('query_start',query_start,'target_start',target_start)
			else:
				seq_query += line.split('\t')[1]
	f.close()
	return result_dict

def seq_with_gap_to_idx(seq):
	idx_list = []
	i = 0
	for aa in seq:
		if aa == '-':
			idx_list.append(-1)
		else:
			idx_list.append(i)
			i += 1
	return idx_list

def get_target_idx(target_idx_list, query_idx_list, align, target_start, query_start):
	pdb_to_uniprot_idx = []
	for i in range(target_start-1):
		pdb_to_uniprot_idx.append(-1)
	for i in range(len(target_idx_list)):
		if target_idx_list[i] != -1:
			if align[i]  == '|' and query_idx_list[i] != -1:
				pdb_to_uniprot_idx.append(query_idx_list[i] + query_start-1)
			else:
				pdb_to_uniprot_idx.append(-1)
	return pdb_to_uniprot_idx

def get_pdb_to_uniprot_map(result_dict):
	#pdb_ratio_dict = {}
	pdb_to_uniprot_map_dict = {}
	for name in result_dict:
		pdbid, chain = name.split('_')
		#if pdbid != '2qbw':
			#continue
		seq_target, seq_query, align, target_start, query_start = result_dict[name]
		#print(seq_target, seq_query, align, target_start, query_start )
		ratio = float(align.count('|'))/float(len(seq_target.replace('-','')))
		if ratio < 0.9:
			continue
		
		target_idx_list = seq_with_gap_to_idx(seq_target)
		query_idx_list = seq_with_gap_to_idx(seq_query)
		#print('seq_target',seq_target, len(seq_query.replace('-','')),  len(seq_target.replace('-','')))
		pdb_to_uniprot_idx = get_target_idx(target_idx_list, query_idx_list, align, target_start, query_start)
		#print('pdb_to_uniprot_idx',pdb_to_uniprot_idx)
		if pdbid in pdb_to_uniprot_map_dict:
			pdb_to_uniprot_map_dict[pdbid][chain] = pdb_to_uniprot_idx
		else:
			pdb_to_uniprot_map_dict[pdbid] = {}
			pdb_to_uniprot_map_dict[pdbid][chain] = pdb_to_uniprot_idx

	return pdb_to_uniprot_map_dict

def get_pocket_in_uniprot_seq(pocket_dict, protein_dict, pdb_to_uniprot, uniprot_seq):	
	#print('pdb_to_uniprot',pdb_to_uniprot['A'])
	pocket_in_uniprot_seq_set = set()
	pocket_in_uniprot_seq_list = []
	residue_record = ''
	
	for chain in pocket_dict:
		if chain not in  pdb_to_uniprot:
			continue
		pocket_seq, pocket_idx = pocket_dict[chain]    # chain, idx (pdb) of interact residue
		#residue_record += pocket_seq
		full_seq, full_idx = protein_dict[chain]       # pdb seuqnce, idx of chain
		#print('pocket_idx',pocket_idx)
		i = -1
		for idx in pocket_idx:
			i += 1
			if full_idx.count(idx) != 1:             # some positions in pdb sequence may have the same idx
				print('idx_list.count(idx) != 1', idx)
			seq_pos = full_idx.index(idx)   # position along pdb sequence
			if seq_pos >= len(pdb_to_uniprot[chain]):
				continue
			if pdb_to_uniprot[chain][seq_pos] == -1:
				continue
			if pdb_to_uniprot[chain][seq_pos] not in pocket_in_uniprot_seq_set:
				pocket_in_uniprot_seq_set.add(pdb_to_uniprot[chain][seq_pos])
				pocket_in_uniprot_seq_list.append(pdb_to_uniprot[chain][seq_pos])
				residue_record += pocket_seq[i]
	return pocket_in_uniprot_seq_list, residue_record


# main
with open('out5_pocket_dict','rb') as f:
	pdbbind_pocket_dict = pickle.load(f)

pdbid_to_ligand = get_pdbid_to_ligand()
result_dict = get_result_dict()
print('result_dict',len(result_dict))
pdb_to_uniprot_map_dict = get_pdb_to_uniprot_map(result_dict)
print('pdb_to_uniprot_map_dict',len(pdb_to_uniprot_map_dict))

i = 0
count_not_in_dataset = 0
count_not_same_seq = 0
count_not_align = 0
uniprot_seq_dict = read_fasta()
pdbbind_pocket_dict_final = {}

for pdbid in pdbbind_pocket_dict:
	i += 1
	if pdbid not in uniprot_seq_dict:
		count_not_in_dataset += 1
		continue
	if pdbid not in pdb_to_uniprot_map_dict:
		count_not_align += 1
		continue
	ligand = pdbid_to_ligand[pdbid]
	uniprot_seq, uniprot_id = uniprot_seq_dict[pdbid]    # get uniprot sequence
	pdb_to_uniprot = pdb_to_uniprot_map_dict[pdbid]
	
	pocket_dict = pdbbind_pocket_dict[pdbid]['pocket'] 
	protein_dict = pdbbind_pocket_dict[pdbid]['protein'] 
	pocket_in_uniprot_seq_list, residue_record = get_pocket_in_uniprot_seq(pocket_dict, protein_dict, pdb_to_uniprot, uniprot_seq)
	
	residue_check = ''.join(np.array([aa for aa in uniprot_seq])[pocket_in_uniprot_seq_list].tolist())
	if residue_record != residue_check:
		count_not_same_seq +=1 
		continue
	
	pdbbind_pocket_dict_final[pdbid] = {}
	pdbbind_pocket_dict_final[pdbid]['ligand'] = ligand
	pdbbind_pocket_dict_final[pdbid]['pocket_in_uniprot_seq'] = pocket_in_uniprot_seq_list
	pdbbind_pocket_dict_final[pdbid]['uniprot_id'] = uniprot_id
	pdbbind_pocket_dict_final[pdbid]['uniprot_seq'] = uniprot_seq


print('pdbbind_pocket_dict_final', len(pdbbind_pocket_dict_final))
print('count_not_in_dataset',count_not_in_dataset)
print 'count_not_same_seq', count_not_same_seq
print 'count_not_align', count_not_align
with open('out8_final_pocket_dict','wb') as f:
	pickle.dump(pdbbind_pocket_dict_final,f,protocol=0)