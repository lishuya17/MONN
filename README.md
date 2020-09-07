## Codes for "MONN: a Multi-Objective Neural Network for Predicting Pairwise Non-Covalent Interactions and Binding Affinities between Compounds and Proteins"

The benchmark dataset described in this paper can be found in ./data/, and the creation of this dataset can be reproduced by the protocol in ./create_dataset/.

Before running the MONN model in ./src/, please first use ./src/preprocessing_and_clustering.py to produce necessary files.

For cross validation, e.g., using IC50 data, new-compound setting and clustering threshold 0.3, run:

```python CPI_train.py IC50 new_compound 0.3```

### Requirements:
Python2.7

rdkit (for preprocessing)

Pytorch >= 0.4.0

scikit-learn

### License

This software is copyrighted by Machine Learning and Computational Biology Group @ Tsinghua University.

The algorithm and data can be used only for NON COMMERCIAL purposes.
