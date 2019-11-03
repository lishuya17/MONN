import pickle

def get_pdbid_to_uniprot():
	pdbid_to_uniprot = {}
	with open('out2_pdbbind_all_datafile.tsv') as f:
		for line in f.readlines():
			try:
				pdbid, uniprotid, ligand, inchi, seq, measure, value = line.strip().split('\t')
			except:
				print(line.strip().split('\t'))
				assert 0
			pdbid_to_uniprot[pdbid] = (uniprotid, seq)
	print('pdbid_to_uniprot',len(pdbid_to_uniprot))
	return pdbid_to_uniprot
pdbid_to_uniprot = get_pdbid_to_uniprot()

# =========================================== for PDB =====================================
with open('out4_interaction_dict','rb') as f:
	interaction_dict = pickle.load(f)
fw1 = open('out6.1_query_pdb.fasta', 'w')
fw2 = open('out6.1_target_uniprot_pdb.fasta', 'w')
for name in interaction_dict:
	pdbid = name.split('_')[0]
	if pdbid not in pdbid_to_uniprot:
		continue
	chain_dict = interaction_dict[name]['sequence']
	uniprotid, uniprotseq = pdbid_to_uniprot[pdbid]
	#print('chain_dict', len(chain_dict))
	for chain_id in chain_dict:
		if len(chain_dict[chain_id][0]) == 0:
			continue
		fw1.write('>'+pdbid+'_'+chain_id+'\n')
		fw1.write(chain_dict[chain_id][0]+'\n')
		fw2.write('>'+pdbid+'_'+chain_id+'_'+uniprotid+'\n')
		fw2.write(uniprotseq+'\n')
fw1.close()
fw2.close()


# =========================================== for PDBbind =====================================
with open('out5_pocket_dict','rb') as f:
	interaction_dict = pickle.load(f)
fw1 = open('out6.2_query_pdbbind.fasta', 'w')
fw2 = open('out6.2_target_uniprot_pdbbind.fasta', 'w')
for pdbid in interaction_dict:
	if pdbid not in pdbid_to_uniprot:
		continue
	try:
		chain_dict = interaction_dict[pdbid]['protein']
	except:
		continue
	uniprotid, uniprotseq = pdbid_to_uniprot[pdbid]
	#print('chain_dict', len(chain_dict))
	for chain_id in chain_dict:
		if len(chain_dict[chain_id][0]) == 0:
			continue
		fw1.write('>'+pdbid+'_'+chain_id+'\n')
		fw1.write(chain_dict[chain_id][0]+'\n')
		fw2.write('>'+pdbid+'_'+chain_id+'_'+uniprotid+'\n')
		fw2.write(uniprotseq+'\n')

fw1.close()
fw2.close()
