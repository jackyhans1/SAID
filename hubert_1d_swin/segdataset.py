import os
import torch
from torch.utils.data import Dataset

class SegmentedAudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_seq_length=None):
        """
        Args:
            file_paths (list): List of paths to .pt feature files.
            labels (list): List of labels corresponding to the feature files.
            max_seq_length (int): Maximum sequence length for padding (optional).
        """
        self.file_paths = file_paths
        self.labels = labels
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feature_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load pre-extracted features from .pt file
        features = torch.load(feature_path)  # Features shape: [Batch, Sequence, Feature_Dimension] or [Sequence, Feature_Dimension]

        # Ensure the tensor is 2D (Sequence, Feature_Dimension)
        if len(features.shape) == 3 and features.shape[0] == 1:
            # Remove batch dimension if present
            features = features.squeeze(0)
        elif len(features.shape) != 2:
            raise ValueError(f"Unexpected feature shape: {features.shape}. Expected 2D tensor.")

        seq_length = features.shape[0]

        # Apply padding if max_seq_length is specified
        if self.max_seq_length:
            if seq_length > self.max_seq_length:
                features = features[:self.max_seq_length, :]
                mask = torch.ones(self.max_seq_length, dtype=torch.bool)
            else:
                pad_length = self.max_seq_length - seq_length
                padding = torch.zeros(pad_length, features.shape[1])
                features = torch.cat((features, padding), dim=0)
                mask = torch.cat((torch.ones(seq_length, dtype=torch.bool), torch.zeros(pad_length, dtype=torch.bool)))
        else:
            mask = torch.ones(seq_length, dtype=torch.bool)

        return features, mask, label


# Collate function for DataLoader
def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    Pads all sequences in the batch to the length of the longest sequence.

    Args:
        batch (list): List of tuples (features, mask, label).

    Returns:
        torch.Tensor: Padded feature tensor of shape [Batch, Max_Sequence_Length, Feature_Dimension].
        torch.Tensor: Attention mask tensor of shape [Batch, Max_Sequence_Length].
        torch.Tensor: Label tensor of shape [Batch].
    """
    features, masks, labels = zip(*batch)

    # Get the maximum sequence length in the batch
    max_seq_length = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]

    # Pad all sequences to the maximum length
    padded_features = torch.zeros(len(features), max_seq_length, feature_dim)
    attention_masks = torch.zeros(len(features), max_seq_length, dtype=torch.bool)

    for i, (f, m) in enumerate(zip(features, masks)):
        padded_features[i, :f.shape[0], :] = f
        attention_masks[i, :m.shape[0]] = m

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_features, attention_masks, labels
