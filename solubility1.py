import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen

class Featurizer:
    import pandas as pd

    def __init__(self, train_smiles):
        self.scaler = StandardScaler()
        train_descriptors = self.get_descriptors(train_smiles)
        self.scaler.fit(train_descriptors)

    def featurize(self, smiles):
        descriptors = self.get_descriptors(smiles)
        scaled_descriptors = self.scaler.transform(descriptors)
        return scaled_descriptors

    def get_descriptors(self, smiles):
        df = pd.DataFrame({'SMILES':smiles})
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        df['mol_wt'] = df['mol'].apply(rdMolDescriptors.CalcExactMolWt)             # Molecular weight
        df['logp'] = df['mol'].apply(Crippen.MolLogP)                               # LogP (lipophilicity)
        df['num_heavy_atoms'] = df['mol'].apply(rdMolDescriptors.CalcNumHeavyAtoms) # Number of heavy atoms
        df['num_HBD'] = df['mol'].apply(rdMolDescriptors.CalcNumHBD)                # Number of hydrogen bond donors
        df['num_HBA'] = df['mol'].apply(rdMolDescriptors.CalcNumHBA)                # Number of hydrogen bond acceptors
        df['aromatic_rings'] = df['mol'].apply(rdMolDescriptors.CalcNumAromaticRings) # Number of aromatic rings
        return  df[['mol_wt', 'num_heavy_atoms', 'num_HBD', 'num_HBA', 'aromatic_rings', 'logp']]

with open('data/svr.pkl', 'rb') as f:
    svr_loaded = pkl.load(f)

with open('data/lr.pkl', 'rb') as l:
    lr_loaded = pkl.load(l)

st.title('Solubility prediction machine')

# Input SMILES
input_smiles = st.text_input("Input a SMILES string")

# Generate prediction
if st.button("GENERATE"):
    if not input_smiles.strip():
        st.write(":red[Please input a valid SMILES string.]")
    else:
        # Initialize Featurizer
        train_smiles = ["CCO", "C=O", "O=C=O"]  # Example training data
        featurizer = Featurizer(train_smiles)

        try:
            # Featurize and predict
            prediction = featurizer.featurize([input_smiles])
            y_pred = svr_loaded.predict(prediction)

            # Store results in session state
            st.session_state["output"] = pd.DataFrame({'SMILES': [input_smiles], 'Solubility': y_pred})
            st.write(st.session_state["output"])

        except Exception as e:
            st.write(f":red[An error occurred: {e}]")

# Clear all data
if st.button("Clear All"):
    st.session_state.clear()  # Reset all session state
    st.experimental_rerun()  # Refresh the app