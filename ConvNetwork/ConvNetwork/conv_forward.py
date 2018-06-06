import numpy as np

from zero_pad import zero_pad
from conv_single_step import conv_single_step

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    print("Input matrix shape",A_prev.shape)
    print("Weight shape", W.shape)

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    print()
    print("input_matrix", A_prev[0,:,:,0].shape, A_prev[0,:,:,0])
    print("vertical_0", A_prev[0,0,:,0].shape, A_prev[0,0,:,0])
    print("horizontal_0", A_prev[0,:,0,0].shape, A_prev[0,:,0,0])

    print()
    print("weight_matrix", W[:,:,0,0].shape, W[:,:,0,0])
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad)/stride)+1
    n_W = int((n_W_prev - f + 2 * pad)/stride)+1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]                               # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    print()
    print("output_matrix", Z[0,:,:,0].shape, Z[0,:,:,0])

    return Z, cache


def test1():
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
                   "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    # 0.0489952035289
    print("Z[3,2,1] =", Z[3,2,1])
    # [-0.61490741 -6.7439236 -2.55153897 1.75698377 3.56208902 0.53036437 5.18531798 8.75898442]
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    # [-0.20075807 0.18656139 0.41005165]


def test2():
    # Give A_prev 1 training example and 1 channel, so that
    # A_prev.shape = (1, n_H_prev, n_W_prev, 1)
    # n_H_prev and n_W_prev can be anything, 
    # as can any elements of A_prev
    n_H_prev = 5
    n_W_prev = 7
    A_prev = np.arange(n_H_prev*n_W_prev) + 1
    A_prev = A_prev.reshape((1, n_H_prev, n_W_prev, 1))

    # Make W a single 1-channel, 3x3 filter of all zeros,  
    # except for a 1 in the center
    W = np.zeros((3,3,1,1))
    W[1,1,0,0] = 1

    # Zero out b, with the appropriate dimensionality
    b = np.zeros((1,1,1,1))

    # Same padding: f = 3 ==> pad = (f-1)/2 = 1, stride = 1
    hparameters = {"pad":1, "stride":1}

    # Run conv_forward()
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

    # If every element of the inner 2 dimensions of Z 
    # doesn't match every element of the inner 2 
    # dimensions of A_prev, something's wrong
    print(Z[0,:,:,0] == A_prev[0,:,:,0])


def test3():
    # Give A_prev 1 training example and 1 channel, so that
    # A_prev.shape = (1, n_H_prev, n_W_prev, 1)
    # n_H_prev and n_W_prev can be anything, 
    # as can any elements of A_prev
    n_H_prev = 5
    n_W_prev = 7
    A_prev = np.arange(n_H_prev*n_W_prev) + 1
    A_prev = A_prev.reshape((1, n_H_prev, n_W_prev, 1))

    # Make W a single 1-channel, 3x3 filter of all zeros,  
    # except for a 1 in the center
    W = np.zeros((3,3,1,1))
    W[1,1,0,0] = 1

    # Zero out b, with the appropriate dimensionality
    b = np.zeros((1,1,1,1))

    # Same padding: f = 3 ==> pad = (f-1)/2 = 1, stride = 1
    hparameters = {"pad":1, "stride":2}

    # Run conv_forward()
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

    # If every element of the inner 2 dimensions of Z 
    # doesn't match every *second* element of the inner 2 
    # dimensions of A_prev, something's wrong
    print(Z[0,:,:,0] == A_prev[0,::2,::2,0]) 


if __name__ == "__main__":
    test1()
    test2()
    test3()