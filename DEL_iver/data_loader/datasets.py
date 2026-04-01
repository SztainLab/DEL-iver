import torch
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np

from utils import retrieve_mol_fp

# A dataset that is built using array of BB indices (e.g., [[0, 0, 1], [0, 0, 24], etc) and 3 dictionary of building blocks FPs
class TrainBBFPDataset_v1(Dataset):
    """Create a training dataset of building block fingerprints (BBFP), inherited from torch.utils.data Dataset class.

    Attributes:
    indices -- array of building block indices, e.g. [[0, 0, 1], [0, 0, 24], [0, 2, 25]]
    bbs_1_fps -- a dictionary of building block 1 fingerprints, e.g. {0: FP0, 1: FP1,...}
    bbs_2_fps -- a dictionary of building block 2 fingerprints, e.g. {0: FP0, 1: FP1,...}
    bbs_3_fps -- a dictionary of building block 3 fingerprints, e.g. {0: FP0, 1: FP1,...}
    labels -- array-like list containing binary labels for the molecules (hit=1, non-hit=0)
    
    Methods:
    __len__(self) -- returns the length of the dataset
    __getitem__(self, idx) -- returns two items: the concatenated fingerprint at the idx index: concatenated_fp (concatenation of bbs_1_fps[indices[idx][0]], bbs_2_fps[indices[idx][1]], bbs_3_fps[indices[idx][2]]) and the label at idx
    """
    def __init__(self, indices, bbs_1_fps, bbs_2_fps, bbs_3_fps, labels):
        self.indices = indices
        self.bbs_1_fps = bbs_1_fps
        self.bbs_2_fps = bbs_2_fps
        self.bbs_3_fps = bbs_3_fps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.indices[idx]
        
        # Fetch the fingerprints for each building block type
        fp1 = torch.tensor(self.bbs_1_fps[index[0]], dtype=torch.float32)
        fp2 = torch.tensor(self.bbs_2_fps[index[1]], dtype=torch.float32)
        fp3 = torch.tensor(self.bbs_3_fps[index[2]], dtype=torch.float32)

        # Concatenate the fingerprints
        concatenated_fp = torch.cat((fp1, fp2, fp3), dim=0)
        
        return concatenated_fp, torch.tensor(self.labels[idx], dtype=torch.float32)
    
# A dataset that is built using array of BB indices (e.g., [[0, 0, 1], [0, 0, 24], etc) and 3 dictionary of building blocks FPs
# Unlike version 1, this version uses the fact that BB2 and BB3 can switch places to take the union of the two fingerprints
# before concatenating them to BB1
class TrainBBFPDataset_v2(Dataset):
    """Create a training dataset of building block fingerprints (BBFP), inherited from torch.utils.data Dataset class. 
    Similar to version1 except version2 uses the fact that BB2 and BB3 can switch places to take the union of the BB2 and BB3 fingerprints
    before concatenating them to BB1. 

    Attributes:
    indices -- array of building block indices, e.g. [[0, 0, 1], [0, 0, 24], [0, 2, 25]]
    bbs_1_fps -- a dictionary of building block 1 fingerprints, e.g. {0: FP0, 1: FP1,...}
    bbs_2_fps -- a dictionary of building block 2 fingerprints, e.g. {0: FP0, 1: FP1,...}
    bbs_3_fps -- a dictionary of building block 3 fingerprints, e.g. {0: FP0, 1: FP1,...}
    labels -- array-like list containing binary labels for the molecules (hit=1, non-hit=0)
    
    Methods:
    __len__(self) -- returns the length of the dataset
    __getitem__(self, idx) -- returns two items: the concatenated fingerprint at the idx index: concatenated_fp (concatenation of bbs_1_fps[indices[idx][0]], union(bbs_2_fps[indices[idx][1]], bbs_3_fps[indices[idx][2]])) and the label at idx
    """
    def __init__(self, indices, bbs_1_fps, bbs_2_fps, bbs_3_fps, labels):
        self.indices = indices
        self.bbs_1_fps = bbs_1_fps
        self.bbs_2_fps = bbs_2_fps
        self.bbs_3_fps = bbs_3_fps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.indices[idx]
        
        # Fetch the fingerprints for each building block type
        # Perform union of BB2 and BB3
        fp1 = torch.tensor(self.bbs_1_fps[index[0]], dtype=torch.float32)
        fp_comb = torch.tensor(np.bitwise_or(self.bbs_2_fps[index[1]], self.bbs_3_fps[index[2]]), dtype=torch.float32)

        # Concatenate the fingerprints
        concatenated_fp = torch.cat((fp1, fp_comb), dim=0)
        
        return concatenated_fp, torch.tensor(self.labels[idx], dtype=torch.float32)
    
# A dataset that is built from a list of fingerprints and labels
class TrainMolFPDataset(Dataset):
    """Create a training dataset of full molecule fingerprints, inherited from torch.utils.data Dataset class.

    Attributes:
    fps -- a list of full molecule chemical fingerprints
    labels -- a lit of labels corresponding to the full molecules (hit=1, non-hit=0) 
    
    Methods:
    __len__(self) -- returns the length of the dataset
    __getitem__(self, idx) -- returns two items: the concatenated fingerprint at the idx index: concatenated_fp (concatenation of bbs_1_fps[indices[idx][0]], bbs_2_fps[indices[idx][1]], bbs_3_fps[indices[idx][2]]) and the label at idx
    """
    def __init__(self, fps: list, labels: list):
        self.fps = fps
        self.labels = labels

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        return torch.tensor(self.fps[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)