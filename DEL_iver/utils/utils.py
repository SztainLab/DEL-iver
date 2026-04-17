from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Sheridan
import numpy as np
import pickle

def load_bb_train_dicts(bbs_dict_1_file, bbs_dict_2_file, bbs_dict_3_file):
    """Load the (previously generated) building block training dictionaries from pickle files. 
        Args:
            bbs_dict_1 (filepath): pickle (.p) file that contains a dictionary of train fingerprints for bb1 e.g. {0: FP0, 1: FP1, ...} 
            bbs_dict_2 (filepath): pickle (.p) file that contains a dictionary of train fingerprints for bb2 e.g. {0: FP0, 1: FP1, ...} 
            bbs_dict_3 (filepath): pickle (.p) file that contains a dictionary of train fingerprints for bb3 e.g. {0: FP0, 1: FP1, ...} 
        Returns:
            returns the three train dictionaries, bbs_1, bbs_2, bbs_3
    """
    with open(bbs_dict_1_file, 'rb') as file:
        bbs_1 = pickle.load(file)

    with open(bbs_dict_2_file, 'rb') as file:
        bbs_2 = pickle.load(file)

    with open(bbs_dict_3_file, 'rb') as file:
        bbs_3 = pickle.load(file)
    return bbs_1, bbs_2, bbs_3

def load_bb_test_dicts(bbs_dict_1_file, bbs_dict_2_file, bbs_dict_3_file):
    """Load the (previously generated) building block test dictionaries from pickle files. 
        Args:
            bbs_dict_1 (filepath): pickle (.p) file that contains a dictionary of test fingerprints for bb1 e.g. {0: FP0, 1: FP1, ...} 
            bbs_dict_2 (filepath): pickle (.p) file that contains a dictionary of test fingerprints for bb2 e.g. {0: FP0, 1: FP1, ...} 
            bbs_dict_3 (filepath): pickle (.p) file that contains a dictionary of test fingerprints for bb3 e.g. {0: FP0, 1: FP1, ...} 
        Returns:
            returns the three test dictionaries, bbs_1, bbs_2, bbs_3
    """
    with open(bbs_dict_1_file, 'rb') as file:
        bbs_1 = pickle.load(file)

    with open(bbs_dict_2_file, 'rb') as file:
        bbs_2 = pickle.load(file)

    with open(bbs_dict_3_file, 'rb') as file:
        bbs_3 = pickle.load(file)
    return bbs_1, bbs_2, bbs_3
    


def validate_required_data(obj, required_fields):
    missing = []

    for field in required_fields:
        # if Enum, use its value; otherwise assume string
        field_name = field.value if hasattr(field, "value") else field
        try:
            value = getattr(obj, field_name)
        except AttributeError:
            missing.append(f"{field_name} (not defined)")
            continue

        if value is None:
            missing.append(field_name)
        elif hasattr(value, "empty") and value.empty:
            missing.append(f"{field_name} (empty)")
        elif isinstance(value, (list, dict, set)) and len(value) == 0:
            missing.append(f"{field_name} (empty)")

    if missing:
        raise ValueError(
            f"Missing required data for {type(obj).__name__}: {', '.join(missing)}"
        )








def retrieve_mol_fp(smiles, fingerprint_type, fingerprint_length=1024):
    """Generate and return the molecule's fingerprint as a numpy array.
        Args:
            smiles (str): SMILES string of molecule
            fingerprint_type (str): 'ECFP4', 'FCFP4', 'MACCS'
            fingerprint_length (int): length of bit array
        Returns:
            np.ndarray: The fingerprint as a NumPy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles}.")
        
    if fingerprint_type == 'ECFP4':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fingerprint_length)
    elif fingerprint_type == 'FCFP4':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=fingerprint_length)
    elif fingerprint_type == 'MACCS':
        fp = list(MACCSkeys.GenMACCSKeys(mol).ToBitString()[1:])
    elif fingerprint_type == 'APDP':
        ap_fp = Pairs.GetAtomPairFingerprint(mol)
        dp_fp = Sheridan.GetBPFingerprint(mol)
        fp = np.zeros(fingerprint_length, dtype=np.uint8)
        ap_nonzero_elements = ap_fp.GetNonzeroElements().keys()
        dp_nonzero_elements = dp_fp.GetNonzeroElements().keys()
        for i in ap_nonzero_elements:
            fp[i % fingerprint_length] = 1
        for i in dp_nonzero_elements:
            fp[(i + 8388608) % fingerprint_length] = 1
    else:
        raise ValueError("Unsupported fingerprint type. Choose 'ECFP4', 'FCFP4', 'MACCS', or 'APDP'.")
    
    return np.array(fp, dtype=np.uint8)

# Generate fingerprint dictionary based on a dictionary that has {0: 'SMILES0', 1: 'SMILES1', ...}
def generate_fp_dict(smiles_dict: dict, fingerprint_type: str, fingerprint_length=1024):
    """Generate fingerprint dictionary.
        Args:
            smiles_dict (dict): dictionary of SMILES e.g. {0: 'SMILES0', 1: 'SMILES1', ...}
            fingerprint_type (str): 'ECFP4', 'FCFP4', 'MACCS'
            fingerprint_length (int): length of bit array, default = 1024
        Returns:
            fp_dict (dict): a dictionary of fps e.g. {0: FP0, 1: FP1, ...}
    """
    fp_dict = {}
    for key, smiles in smiles_dict.items():
        fp_dict[key] = retrieve_mol_fp(smiles, fingerprint_type, fingerprint_length)
    return fp_dict
    
# Replace Dy group with PEG linker
def replace_Dy(smiles):
    """Replace the Dy in a SMILES string with a PEG linker.
        Args:
            smiles (str): SMILES string of molecule containing Dy
        Returns:
            edited_smiles (str): SMILES string of molecule with Dy replaced by PEG linker
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles
    dy_mol = Chem.MolFromSmiles('[Dy]')
    index = mol.GetSubstructMatch(dy_mol)
    if not index:
        print("DNA tag 'Dy' not found. Skipping.")
        return smiles
    
    peg_mol = Chem.MolFromSmiles('CCOCCOCCO')
    edited_mol = Chem.ReplaceSubstructs(mol, 
                                        dy_mol, 
                                        peg_mol)[0]
    
    return Chem.MolToSmiles(edited_mol)


def determine_ddr_type(ddr):
    """
    Return the underlying data stored in a Data_Reader.

    Returns
    -------
    pd.DataFrame OR iterator
    """

    if ddr is None:
        raise ValueError("ddr cannot be None")

    if getattr(ddr, "df", None) is not None:
        return ddr.df

    if getattr(ddr, "iter", None) is not None:
        return ddr.iter

    raise ValueError("Data_Reader has neither df nor iter set.")