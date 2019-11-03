from rdkit import Chem
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import os
import pickle


def get_pdbid_list():
	pdbid_list = []
	with open('out1.2_pdbid_list.txt') as f:
		for line in f.readlines():
			pdbid_list.append(line.strip())
	print('pdbid_list',len(pdbid_list))
	return pdbid_list

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

def get_bonds(pdbid, ligand, atom_idx_list):
	bond_list = []
	f = open('./plip_results/output_'+pdbid+'.txt')
	isheader = False
	for line in f.readlines():
		if line[0] == '*':
			bond_type = line.strip().replace('*','')
			isheader = True
		if line[0] == '|':
			if isheader:
				header = line.replace(' ','').split('|')
				isheader = False
				continue
			lines = line.replace(' ','').split('|')
			if ligand not in lines[5]:
				continue
			aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(lines[4]), lines[5], lines[6]
			if bond_type in ['Hydrogen Bonds', 'Water Bridges'] :
				atom_idx1, atom_idx2 = int(lines[12]), int(lines[14])
				if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
					continue
				if atom_idx1 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
				elif atom_idx2 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
				else:
					print(pdbid, ligand, bond_type, 'error: atom index in plip result not in atom_idx_list')
					print(atom_idx1, atom_idx2)
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
			elif bond_type == 'Hydrophobic Interactions':
				atom_idx_ligand, atom_idx_protein = int(lines[8]), int(lines[9])
				if  atom_idx_ligand not in atom_idx_list: 
					continue
				elif atom_idx_ligand not in atom_idx_list:
					print('error: atom index in plip result not in atom_idx_list')
					print('Hydrophobic Interactions', atom_idx_ligand, atom_idx_protein)
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
			elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
				atom_idx_ligand_list = list(map(int, lines[11].split(',')))
				if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
					print(bond_type, 'error: atom index in plip result not in atom_idx_list')
					print(atom_idx_ligand_list)
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
			elif bond_type == 'Salt Bridges':
				atom_idx_ligand_list = list(set(map(int, lines[10].split(','))))
				if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
					print('error: atom index in plip result not in atom_idx_list')
					print('Salt Bridges', atom_idx_ligand_list, set(atom_idx_ligand_list).intersection(set(atom_idx_list)))
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
			elif bond_type == 'Halogen Bonds':
				atom_idx1, atom_idx2 = int(lines[11]), int(lines[13])
				if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
					continue
				if atom_idx1 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
				elif atom_idx2 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
				else:
					print('error: atom index in plip result not in atom_idx_list')
					print('Halogen Bonds', atom_idx1, atom_idx2)
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
			else:
				print('bond_type',bond_type)
				print(header)
				print(lines)
				return None
	f.close()
	if len(bond_list) != 0:
		return bond_list

def get_atoms_from_pdb(ligand, pdbid):   
	# from pdb protein structure, get ligand index list for bond extraction
	p = PDBParser()
	atom_idx_list = []
	atom_name_list = []
	structure = p.get_structure(pdbid, './pdb_files/'+pdbid+'.pdb')
	seq_dict = {}
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			id_list = []
			for res in chain:
				if ligand == res.get_resname():
					if res.get_id()[0] == ' ':
						continue
					for atom in res:
						atom_idx_list.append(atom.get_serial_number())
						atom_name_list.append(atom.get_id())
	if len(atom_idx_list) != 0:
		return atom_idx_list, atom_name_list
	else:
		return None, None

def get_mol_from_ligandpdb(ligand):
	if not os.path.exists('./pdb_files/'+ligand+'_ideal.pdb'):
		return None, None, None
	name_order_list = []
	name_to_idx_dict, name_to_element_dict = {}, {}
	p = PDBParser()
	structure = p.get_structure(ligand, './pdb_files/'+ligand+'_ideal.pdb')
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			for res in chain:
				if ligand == res.get_resname():
					#print(ligand,res.get_resname(),res.get_full_id())
					for atom in res:
						name_order_list.append(atom.get_id())
						name_to_element_dict[atom.get_id()] = atom.element
						name_to_idx_dict[atom.get_id()] = atom.get_serial_number()-1
	#print('check', name_to_idx_dict.items())
	if len(name_to_idx_dict) == 0:
		return None, None, None
	return name_order_list, name_to_idx_dict, name_to_element_dict

def get_interact_atom_name(atom_idx_list, atom_name_list,bond_list):
	interact_atom_name_list = []
	interact_bond_type_list = []
	interact_atom_name_set = set()
	assert len(atom_idx_list) == len(atom_name_list)
	for bond in bond_list:
		for atom_idx in bond[-1]:
			atom_name = atom_name_list[atom_idx_list.index(atom_idx)]
			#if atom_name not in interact_atom_name_set:
			interact_atom_name_set.add(atom_name)
			interact_atom_name_list.append(atom_name)
			interact_bond_type_list.append((atom_name, bond[0]))
	return interact_atom_name_list, interact_bond_type_list

def get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list):
	atom_idx_list = []
	atom_name_list = []
	atom_element_list = []
	atom_interact_list = []
	for name in name_order_list:
		idx = atom_name_to_idx_dict[name]
		atom_idx_list.append(idx)
		atom_name_list.append(name)
		atom_element_list.append(atom_name_to_element_dict[name])
		atom_interact_list.append(int(name in interact_atom_name_list))
	return atom_idx_list, atom_name_list, atom_element_list, atom_interact_list

def get_seq(pdbid):
	p = PDBParser()
	structure = p.get_structure(pdbid, './pdb_files/'+pdbid+'.pdb')
	seq_dict = {}
	idx_to_aa_dict = {}
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			if chain_id == ' ':
				continue
			seq = ''
			id_list = []
			for res in chain:
				if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':   # remove HETATM
					continue
				try:
					seq+=three_to_one(res.get_resname())
					idx_to_aa_dict[chain_id+str(res.get_id()[1])+res.get_id()[2].strip()] = three_to_one(res.get_resname())
				except:
					print('unexpected aa name', res.get_resname())
				id_list.append(res.get_id()[1])
			seq_dict[chain_id] = (seq,id_list)
	return seq_dict, idx_to_aa_dict

def get_interact_residue(idx_to_aa_dict, bond_list):
	interact_residue = []
	for bond in bond_list:
		if bond[1]+str(bond[3]) not in idx_to_aa_dict:
			continue
		aa = idx_to_aa_dict[bond[1]+str(bond[3])]
		assert three_to_one(bond[2]) == aa
		interact_residue.append((bond[1]+str(bond[3]), aa, bond[0]))
	if len(interact_residue) != 0:
		return interact_residue
	else:
		return None

no_valid_ligand = 0
no_such_ligand_in_pdb_error = 0
no_interaction_detected_error = 0
no_ideal_pdb_error = 0
empty_atom_interact_list = 0
protein_seq_error = 0

i = 0
interaction_dict = {}
pdbid_list = get_pdbid_list()
for pdbid in pdbid_list:
	i += 1
	print(i, pdbid)
	if pdbid not in pdbid_to_ligand:
		no_valid_ligand += 1
		continue
	ligand = pdbid_to_ligand[pdbid]
	
	# get bond
	atom_idx_list, atom_name_list =  get_atoms_from_pdb(ligand, pdbid)  # for bond atom identification
	if atom_idx_list is None:
		no_such_ligand_in_pdb_error += 1
		print('no such ligand in pdb','pdbid', pdbid, 'ligand', ligand)
		continue
	bond_list = get_bonds(pdbid, ligand, atom_idx_list)
	if bond_list is None:
		print('empty bond list: pdbid', pdbid, 'ligand', ligand, 'atom_idx_list', len(atom_idx_list))
		no_interaction_detected_error += 1
		continue
	interact_atom_name_list, interact_bond_type_list = get_interact_atom_name(atom_idx_list, atom_name_list,bond_list)

	name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict = get_mol_from_ligandpdb(ligand)
	if atom_name_to_idx_dict == None:
		no_ideal_pdb_error+=1
		continue
	atom_idx_list, atom_name_list, atom_element_list, atom_interact_list \
	= get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list)
	if len(atom_idx_list) == 0:
		empty_atom_interact_list+=1
		continue
	
	# get sequence interaction information
	seq_dict, idx_to_aa_dict = get_seq(pdbid)
	interact_residue_list = get_interact_residue(idx_to_aa_dict, bond_list)
	if interact_residue_list is None:
		protein_seq_error += 1
		continue
	
	interaction_dict[pdbid+'_'+ligand] = {}
	interaction_dict[pdbid+'_'+ligand]['bond'] = bond_list
	interaction_dict[pdbid+'_'+ligand]['atom_idx'] = atom_idx_list
	interaction_dict[pdbid+'_'+ligand]['atom_name'] = atom_name_list
	interaction_dict[pdbid+'_'+ligand]['atom_element'] = atom_element_list
	interaction_dict[pdbid+'_'+ligand]['atom_interact'] = atom_interact_list
	interaction_dict[pdbid+'_'+ligand]['atom_bond_type'] = interact_bond_type_list
	
	interaction_dict[pdbid+'_'+ligand]['sequence'] = seq_dict
	interaction_dict[pdbid+'_'+ligand]['residue_interact'] = interact_residue_list
	

print('interaction_dict', len(interaction_dict))
print('no_valid_ligand error', no_valid_ligand)
print('no_such_ligand_in_pdb_error', no_such_ligand_in_pdb_error)
print('no_interaction_detected_error', no_interaction_detected_error)
print('no_ideal_pdb_error',no_ideal_pdb_error)
print('empty_atom_interact_list',empty_atom_interact_list)
print('protein_seq_error',protein_seq_error)

with open('out4_interaction_dict', 'wb') as f:
	pickle.dump(interaction_dict, f, protocol=0)
