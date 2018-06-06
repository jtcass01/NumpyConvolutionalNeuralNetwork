import numpy as np
import matplotlib.pyplot as plt

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    
    return X_pad

if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)

    print ("x.shape =", x.shape)
    # (4,3,3,2)
    print ("x_pad.shape =", x_pad.shape)
    # (4,7,7,2)
    print ("x[1,1] =", x[1,1])
    # [[ 0.90085595 -0.68372786] [-0.12289023 -0.93576943] [-0.26788808 0.53035547]]
    print ("x_pad[1,1] =", x_pad[1,1])
    # [[ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.] [ 0. 0.]]

    print(x[0,:,:,0])
    print(x_pad[0,:,:,0])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0,:,:,0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0,:,:,0])