## Codes for "MONN: a Multi-Objective Neural Network for Predicting Pairwise Non-Covalent Interactions and Binding Affinities between Compounds and Proteins"

The benchmark dataset described in this paper can be found in ./data/, and the creation of this dataset can be reproduced by the protocol in ./create_dataset/.

Before run the MONN model in ./src/, please first use ./src/preprocessing_and_clustering.py to produce necessary files.

### Requirements:
Python2.7

rdkit (for preprocessing)

Pytorch >= 0.4.0

scikit-learn

### Training \& Evaluation
```
cd ./src/
python CPI_train.py
```
