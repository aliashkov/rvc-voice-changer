import numpy as np

def calculate_stats(arr):
    """
    Calculate the mean and standard deviation of a given array.
    
    Args:
    arr (array-like): Input array of numbers
    
    Returns:
    tuple: (mean, standard deviation)
    """
    # Convert input to numpy array if it's not already
    arr = np.array(arr)
    
    # Calculate mean
    mean = np.mean(arr)
    
    # Calculate standard deviation
    std_dev = np.std(arr)
    
    return mean, std_dev
