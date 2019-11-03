# =================================== protein-ligand complex =======================

def get_pdbid_list():
	pdbid_list = []
	with open('./pdbbind_index/INDEX_general_PL.2018') as f:
		for line in f.readlines():
			if line[0] != '#':
				pdbid_list.append(line.strip().split()[0])
	print('pdbid_list',len(pdbid_list))
	return pdbid_list
pdbid_list = get_pdbid_list()

fw = open('out1.2_pdbid_list.txt','w')
for pdbid in pdbid_list:
	fw.write(pdbid+'\n')
fw.close()

fw = open('out1.2_pdbbind_wget_complex.txt','w')
for pdbid in pdbid_list:
	fw.write('https://files.rcsb.org/download/'+pdbid+'.pdb\n')
fw.close()

# ================================= ligand pdb file =================================
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

fw = open('out1.2_pdbbind_wget_ligand.txt','w')
for pdbid in pdbid_list:
	if pdbid in pdbid_to_ligand:
		ligand = pdbid_to_ligand[pdbid]
		fw.write('https://files.rcsb.org/ligands/download/'+ligand+'_ideal.pdb\n')
fw.close()

"""
# ============================= for retrival uniprot sequences ===============
# first get pdbbind_all_mapping.tab from uniprot using the pdbid list from the above step
def get_uniprotid():
	uniprotid_set = set()
	with open('./index/INDEX_general_PL_name.2018') as f:
		for line in f.readlines():
			if line[0] != '#':
				lines = line.strip().split('  ')
				if lines[2] != '------':
					uniprotid_set.add(lines[2])
	print('uniprotid_set step1',len(uniprotid_set))
	
	with open('pdbbind_all_mapping.tab') as f:
		for line in f.readlines()[1:]:
			lines = line.split('\t')
			uniprotid_set.add(lines[1])
	print('uniprotid_set step2',len(uniprotid_set))
	return uniprotid_set

uniprotid_set = get_uniprotid()
uniprotid_list = list(uniprotid_set)
fw = open('0_uniprotid_list.txt','w')
for uniprotid in uniprotid_list:
	fw.write(uniprotid+'\n')
fw.close()
"""