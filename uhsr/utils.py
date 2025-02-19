import numpy as np

def logistic_norm(scores, a=5):
    """
    Apply logistic normalization to map scores into the (0,1) range.
    
    Parameters:
        scores (np.array): Raw scores.
        a (float): Steepness parameter; higher values produce a steeper transition around the mean.
    
    Returns:
        np.array: Normalized scores.
    """
    return 1 / (1 + np.exp(-a * (scores - np.mean(scores))))
