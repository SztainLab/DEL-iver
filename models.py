import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention module
class AttentionModule(nn.Module):
    """Create AttentionModule for attention layers in models, inherited from torch.nn.Module class.

    Attributes:
    attention_weights -- attention weights determined by linear layer and input dimension
    
    Methods:
    forward(self, inputs) -- returns output of attention layer applied to inputs
    """
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        weights = F.softmax(self.attention_weights(inputs), dim=0)
        return (inputs * weights).sum(dim=0)

class Simple_BuildingBlock_MLP(nn.Module):
    """MLP model class that uses the concatenated fingerprints from 3 building blocks to predict the probability of binding to one target. This class is inherited from the nn.Module class.
        Model architecture (3 layers):
            - Dimension per layer: fingerprint length * number of building blocks --> 512 --> 256 --> 1
            - The first two layers are linear layers with relu activation
            - The last layer is a linear layer with sigmoid activation to produce the output prediction
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """
    def __init__(self, fingerprint_length, num_bb=3):
        super(Simple_BuildingBlock_MLP, self).__init__()
        self.fc1 = nn.Linear(fingerprint_length * num_bb, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Simple_BuildingBlock_MLP_w_DropOut(nn.Module):
    """MLP model class that uses the concatenated fingerprints from 3 building blocks to predict the probability of binding to one target. This class is inherited from the nn.Module class.
        
        This model differs from Simple_BuildingBlock_MLP in that: 
            - It contains 4 layers instead of 3
            - Between each layer, a dropout with 50% probability is applied
        
        Model architecture (4 layers):
            - Dimension per layer: fingerprint length * number of building blocks --> 1024 --> 512 --> 256 --> 1
            - The first two layers are linear layers with relu activation
            - The last layer is a linear layer with sigmoid activation to produce the output prediction
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """
    def __init__(self, fingerprint_length, num_bb=3):
        super(Simple_BuildingBlock_MLP_w_DropOut, self).__init__()
        self.fc1 = nn.Linear(fingerprint_length * num_bb, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x

class PermutationInvariant_BuildingBlock_MLP(nn.Module):
    """MLP model class that uses the concatenated fingerprints from 3 building blocks and shared layers for BBs 2 and 3 to predict the probability of binding to one target. This class is inherited from the nn.Module class.
        
        This model differs from Simple_BuildingBlock_MLP in that: 
            - It contains 1 layer for BB1 processing, 1 shared layer for BB2 and BB3 processing, and then 2 layers for concatenated BB1+BB2+BB3 features
        
        Model architecture (5 layers):
            - BB1 processing layer: fingerprint length --> 512 then relu activation
            - BB2 and BB3 processing layer: for BB2 and BB3, fingerprint length --> 512 then relu activation
            - Add the features of BB2 and BB3 from the shared processing layer (bitwise)
            - Concatenate the BB1 feature vector with the BB2+BB3 aggregated feature vector for a vector of size 1024
            - Dimension per layer after BB processing layers: 1024 --> 256 --> 1
                - The first layer is linear relu activation
                - The last layer is a linear layer with sigmoid activation to produce the output prediction
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """
    def __init__(self, fingerprint_length):
        super(PermutationInvariant_BuildingBlock_MLP, self).__init__()
        self.fp_length = fingerprint_length
        # Shared layers for the second and third building blocks
        self.shared_fc = nn.Linear(fingerprint_length, 512)
        
        # Individual layer for the first building block
        self.fc1_bb1 = nn.Linear(fingerprint_length, 512)
        
        # Following layers process the combined information
        self.fc2 = nn.Linear(512 * 2, 256)  # Input size doubled due to combination of bb1 and aggregated bb2&3
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        # Split the concatenated input tensor into three parts
        bb1 = x[:, :self.fp_length]  # First bits for bb1
        bb2 = x[:, self.fp_length:self.fp_length*2]  # Next bits for bb2
        bb3 = x[:, self.fp_length*2:]  # Last bits for bb3

        # Process bb1 through its dedicated layer
        bb1_processed = F.relu(self.fc1_bb1(bb1))

        # Apply shared layer to bb2 and bb3
        bb2_processed = F.relu(self.shared_fc(bb2))
        bb3_processed = F.relu(self.shared_fc(bb3))

        # Sum the features of bb2 and bb3 due to permutation invariance
        aggregated_bb2_bb3 = bb2_processed + bb3_processed

        # Concatenate bb1_processed and aggregated_bb2_bb3
        combined_features = torch.cat((bb1_processed, aggregated_bb2_bb3), dim=1)

        # Further processing the combined features
        x = F.relu(self.fc2(combined_features))
        x = torch.sigmoid(self.fc3(x))
        return x

# Permutation invariant MLP version 2  
class BBFP_PermInvarNN_v2(nn.Module):
    """MLP model that uses shared layers for the first, second, and third building block fingerprints to create permutation invariance.
        - Shared linear layer with relu activation applied to bb1, bb2, and bb3 
        - Bitwise add outputs of bb1, bb2, and bb3 after shared layer 
        - Linear layer with relu activation
        - Linear layer with sigmoid activation
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """
    def __init__(self, fingerprint_length):
        super(BBFP_PermInvarNN_v2, self).__init__()
        self.fp_length = fingerprint_length
        # Shared layers for the all building blocks
        self.shared_fc = nn.Linear(fingerprint_length, 512)
        
        # Following layers process the combined information
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        # Split the concatenated input tensor into three parts
        bb1 = x[:, :self.fp_length]  # First bits for bb1
        bb2 = x[:, self.fp_length:self.fp_length*2]  # Next bits for bb2
        bb3 = x[:, self.fp_length*2:]  # Last bits for bb3

        # Process bb1 through its dedicated layer
        bb1_processed = F.relu(self.shared_fc(bb1))
        bb2_processed = F.relu(self.shared_fc(bb2))
        bb3_processed = F.relu(self.shared_fc(bb3))

        # Sum the features of bb1 and bb2 and bb3 
        aggregated_bbs = bb1_processed + bb2_processed + bb3_processed

        # Further processing the combined features
        x = F.relu(self.fc2(aggregated_bbs))
        x = torch.sigmoid(self.fc3(x))
        return x

# Permutation invariant MLP version 3
class BBFP_PermInvarNN_v3(nn.Module):
    """MLP model that uses shared layers for the second and third building block fingerprints to create permutation invariance between BB2 and BB3.
    Also applies attention.
        - Linear layer with relu activation for bb1 (no reduction in size)
        - Shared linear layer with relu activation applied to bb2 and bb3 (no reduction in size)
        - Attention layer to aggregate outputs of shared layer from bb2 and bb3
        - Concatenate fp1 after first layer with aggregated attention output from bb2 and bb3
        - Three linear layers with relu activation
        - One linear layer (no activation)
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """ 
    def __init__(self, fingerprint_length):
        super(BBFP_PermInvarNN_v3, self).__init__()
        self.fp_length = fingerprint_length

        # Shared layers for the second and third building blocks
        self.shared_fc = nn.Linear(fingerprint_length, fingerprint_length)
        # Individual layer for the first building block
        self.fc1_bb1 = nn.Linear(fingerprint_length, fingerprint_length)
        # Attention layer
        self.attention = AttentionModule(fingerprint_length)
        
        # Following layers process the combined information
        self.fc2 = nn.Linear(fingerprint_length * 2, fingerprint_length)  # Input size doubled due to combination of bb1 and aggregated bb2&3
        self.fc3 = nn.Linear(fingerprint_length, fingerprint_length // 2)
        self.fc4 = nn.Linear(fingerprint_length // 2, fingerprint_length // 4)
        self.fc5 = nn.Linear(fingerprint_length // 4, 1)

    def forward(self, x):
        # Split the concatenated input tensor into three parts
        bb1 = x[:, :self.fp_length]  # First bits for bb1
        bb2 = x[:, self.fp_length:self.fp_length*2]  # Next bits for bb2
        bb3 = x[:, self.fp_length*2:]  # Last bits for bb3

        # Process bb1 through its dedicated layer
        bb1_processed = F.relu(self.fc1_bb1(bb1))

        # Apply shared layer to bb2 and bb3
        bb2_processed = F.relu(self.shared_fc(bb2))
        bb3_processed = F.relu(self.shared_fc(bb3))

        # Use attention on bb2 and bb3
        aggregated_bb2_bb3 = self.attention(torch.stack([bb2_processed, bb3_processed]))

        # Concatenate bb1_processed and aggregated_bb2_bb3
        combined_features = torch.cat((bb1_processed, aggregated_bb2_bb3), dim=1)

        # Further processing the combined features
        x = F.relu(self.fc2(combined_features))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
# Permutation invariant MLP version 4
class BBFP_PermInvarNN_v4(nn.Module):
    """MLP model that uses shared layers for the second and third building block fingerprints to create permutation invariance between BB2 and BB3.
    Also applies attention and batch normalization
        - Linear layer with relu activation for bb1 with batch normalization (no reduction in size)
        - Shared linear layer with relu activation applied to bb2 and bb3 with batch normalization (no reduction in size)
        - Attention layer to aggregate outputs of shared layer from bb2 and bb3
        - Concatenate fp1 after first layer with aggregated attention output from bb2 and bb3
        - Three linear layers with relu activation with batch normalization
        - One linear layer (no activation)
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """ 
    def __init__(self, fingerprint_length):
        super(BBFP_PermInvarNN_v4, self).__init__()
        self.fp_length = fingerprint_length

        # Shared layers for the second and third building blocks
        self.shared_fc = nn.Linear(fingerprint_length, fingerprint_length)
        self.shared_bn = nn.BatchNorm1d(fingerprint_length)  # Batch norm for shared_fc

        # Individual layer for the first building block
        self.fc1_bb1 = nn.Linear(fingerprint_length, fingerprint_length)
        self.bn1_bb1 = nn.BatchNorm1d(fingerprint_length)  # Batch norm for fc1_bb1

        # Attention layer
        self.attention = AttentionModule(fingerprint_length)

        # Following layers process the combined information
        self.fc2 = nn.Linear(fingerprint_length * 2, fingerprint_length)  # Input size doubled due to combination of bb1 and aggregated bb2&3
        self.bn2 = nn.BatchNorm1d(fingerprint_length)  # Batch norm for fc2
        self.fc3 = nn.Linear(fingerprint_length, fingerprint_length // 2)
        self.bn3 = nn.BatchNorm1d(fingerprint_length // 2)  # Batch norm for fc3
        self.fc4 = nn.Linear(fingerprint_length // 2, fingerprint_length // 4)
        self.bn4 = nn.BatchNorm1d(fingerprint_length // 4)  # Batch norm for fc4
        self.fc5 = nn.Linear(fingerprint_length // 4, 1)

    def forward(self, x):
        # Split the concatenated input tensor into three parts
        bb1 = x[:, :self.fp_length]  # First bits for bb1
        bb2 = x[:, self.fp_length:self.fp_length*2]  # Next bits for bb2
        bb3 = x[:, self.fp_length*2:]  # Last bits for bb3

        # Process bb1 through its dedicated layer
        bb1_processed = F.relu(self.bn1_bb1(self.fc1_bb1(bb1)))

        # Apply shared layer to bb2 and bb3
        bb2_processed = F.relu(self.shared_bn(self.shared_fc(bb2)))
        bb3_processed = F.relu(self.shared_bn(self.shared_fc(bb3)))

        # Use attention on bb2 and bb3
        aggregated_bb2_bb3 = self.attention(torch.stack([bb2_processed, bb3_processed]))

        # Concatenate bb1_processed and aggregated_bb2_bb3
        combined_features = torch.cat((bb1_processed, aggregated_bb2_bb3), dim=1)

        # Further processing the combined features
        x = F.relu(self.bn2(self.fc2(combined_features)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

# Model that uses fingerprints of molecules to predict one target
class MolFP_SimpleNN(nn.Module):
    """MLP model that uses full molecule fingerprints to predict on one target.
        - Three linear layers with relu activation
        - One linear layer with sigmoid activation
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """ 
    def __init__(self, fingerprint_length):
        super(MolFP_SimpleNN, self).__init__()
        self.fc1 = nn.Linear(fingerprint_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    
# Model that predicts read count from building block fingerprints
class BBFP_PermInvarNN_v3_ReadCount(nn.Module):
    """MLP model that uses shared layers for the second and third building block fingerprints to create permutation invariance between BB2 and BB3.
    Predicts read count rather than binding probability.
        - Linear layer with relu activation for bb1 (no reduction in size)
        - Shared linear layer with relu activation applied to bb2 and bb3 (no reduction in size)
        - Attention layer to aggregate outputs of shared layer from bb2 and bb3
        - Concatenate fp1 after first layer with aggregated attention output from bb2 and bb3
        - Three linear layers with relu activation with batch normalization
        - One linear layer
    
    Methods:
    forward(self, x) -- returns the output of x after passing through the model
    """ 
    def __init__(self, fingerprint_length):
        super(BBFP_PermInvarNN_v3, self).__init__()
        self.fp_length = fingerprint_length

        # Shared layers for the second and third building blocks
        self.shared_fc = nn.Linear(fingerprint_length, fingerprint_length)
        # Individual layer for the first building block
        self.fc1_bb1 = nn.Linear(fingerprint_length, fingerprint_length)
        # Attention layer
        self.attention = AttentionModule(fingerprint_length)
        
        # Following layers process the combined information
        self.fc2 = nn.Linear(fingerprint_length * 2, fingerprint_length)  # Input size doubled due to combination of bb1 and aggregated bb2&3
        self.fc3 = nn.Linear(fingerprint_length, fingerprint_length // 2)
        self.fc4 = nn.Linear(fingerprint_length // 2, fingerprint_length // 4)
        self.fc5 = nn.Linear(fingerprint_length // 4, 1)

    def forward(self, x):
        # Split the concatenated input tensor into three parts
        bb1 = x[:, :self.fp_length]  # First bits for bb1
        bb2 = x[:, self.fp_length:self.fp_length*2]  # Next bits for bb2
        bb3 = x[:, self.fp_length*2:]  # Last bits for bb3

        # Process bb1 through its dedicated layer
        bb1_processed = F.relu(self.fc1_bb1(bb1))

        # Apply shared layer to bb2 and bb3
        bb2_processed = F.relu(self.shared_fc(bb2))
        bb3_processed = F.relu(self.shared_fc(bb3))

        # Use attention on bb2 and bb3
        aggregated_bb2_bb3 = self.attention(torch.stack([bb2_processed, bb3_processed]))

        # Concatenate bb1_processed and aggregated_bb2_bb3
        combined_features = torch.cat((bb1_processed, aggregated_bb2_bb3), dim=1)

        # Further processing the combined features
        x = F.relu(self.fc2(combined_features))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x