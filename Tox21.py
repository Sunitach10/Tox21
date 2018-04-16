
from __future__ import print_function

import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem

np.random.seed(123)


#SDF file to mol and Dataframe
def sdf_to_df(filepass):
    mols = [mol for mol in Chem.SDMolSupplier(filepass) if mol is not None]
    for id, mol in enumerate(mols):
        if id == 0:
            dicts = mol.GetPropsAsDict()
            df = pd.DataFrame(dicts, index=[id,])
        else:
            dicts = mol.GetPropsAsDict()
            dfplus = pd.DataFrame(dicts, index=[id,])
            df = df.append(dfplus)
    return mols, df

#Return list of indices of Nonetypes when SDF file is coneverted to mol by RDkit.
def search_nonetype_id(more, less):
    result = []
    for i in range(len(more)):
        if more['Sample ID'].values[i] in  less['Sample ID'].values:
            pass
        else:
            result.append(i)
    return result


## Train sets ##


train_x, train_df = sdf_to_df('./tox21_10k_data_all.sdf')

#train_x = [Chem.AddHs(mol) for mol in train_x if mol is not None] # Uncomment: Adds hydrogens to the graph of a molecule.
train_label = train_df.drop(train_df.columns[:3], axis=1)
train_label = train_label.values


## Valid sets ##


valid_x, valid_df = sdf_to_df('./tox21_10k_challenge_test.sdf')

#valid_x = [Chem.AddHs(mol) for mol in valid_x if mol is not None] # Uncomment: Adds hydrogens to the graph of a molecule.

#Drop useless columns.
valid_label = valid_df.drop(valid_df.columns[:2], axis=1)
valid_label = valid_label.values



## Test sets ##


test_x, test_df = sdf_to_df('./tox21_10k_challenge_score.sdf')

#test_x = [Chem.AddHs(mol) for mol in test_x if mol is not None] # Uncomment: Adds hydrogens to the graph of a molecule.

test_df_label = pd.read_table('./tox21_10k_challenge_score.csv')
none_ids = search_nonetype_id(test_df_label, test_df)

#Drop indices of Nonetype and useless columns.
test_label = test_df_label.drop(none_ids)
test_label = test_label.drop(test_label.columns[:1], axis=1)

#'x' to 'NaN'
test_label = np.where(test_label == 'x' , np.nan, test_label)
test_label = test_label.astype(np.float64)







