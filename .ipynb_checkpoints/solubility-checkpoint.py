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

# Initialize session state for output data
if "output" not in st.session_state:
    st.session_state["output"] = None

st.write("Hello! You have encountered a magic ball (just imagine it's a biochemically correct version of the Magic 8-Ball) that enables you to predict the solubility of chosen molecules based on their SMILES!")

st.write("**Input one or more SMILES strings, separated by commas, to check the solubility, AND BE READY FOR MAGIC!**")

chosen_smiles = [st.text_input("Input a SMILES string")]
featurizer = Featurizer(chosen_smiles)

if st.button("**GENERATE**", icon="🔥"):
    prediction = featurizer.featurize(chosen_smiles)   # Extract features for the new dataset
    y_pred = svr_loaded.predict(prediction)  # Use the SVR model to predict the solubility
    guess = pd.DataFrame({'SMILES': chosen_smiles, 'Solubility': y_pred})
    st.write(guess)

    if not chosen_smiles:    # if the user did not enter a smiles string, display a warning message
        st.write(':red[Enter SMILES to discover the full power of the machine]')
        st.stop()   # stop the execution of the script