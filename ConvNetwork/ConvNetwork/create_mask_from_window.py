
import numpy as np

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = (x == np.max(x))
    
    return mask

if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(2,3)
    mask = create_mask_from_window(x)
    print('x = ', x)
    # [[ 1.62434536 -0.61175641 -0.52817175] [-1.07296862 0.86540763 -2.3015387 ]]
    print("mask = ", mask)
    # [[ True False False] [False False False]]