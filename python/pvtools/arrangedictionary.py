import numpy as np
import math

def arrangedictionary(weights: np.ndarray, upscaling:int=1):
    """arrangedictionary(weights, upscaling)
    Given a 4-D numpy array, of shape (numFeatures,nyp,nxp,nfp),
    arranges the dictionary elements into a 3-D array of shape
    (nrows * nyp, ncols * nxp, nfp), where ncols and nrows are chosen
    automatically so that each is close to sqrt(numFeatures) but ncols * nrows
    is greater than or equal to numFeatures. Each feature is individually
    rescaled linearly into the interval [-1, 1], with zero values remaining at
    zero. (If a feature's values are all close to zero, that feature is
    unchanged).

    If the optional argument 'upscaling' is positive, the resulting array is
    upsampled by that factor; i.e. each pixel is replaced by a pixel-by-pixel
    square, and the return value has shape
    (ncols * nxp * upscaling)-by-(nrows * nyp * upscaling)-by-nfp.
    """
    nxp = weights.shape[2]  # width
    nyp = weights.shape[1]  # height
    nfp = weights.shape[3]  # post-synaptic features
    nf = weights.shape[0]  # dictionary size
    nrows = math.floor(math.sqrt(nf))    # number of patches down
    ncols = math.ceil(nf / nrows)       # number of patches across

    # get the weight values and set up an empty dictionary image to fill out
    dictionary = np.zeros((nyp * nrows, nxp * ncols, nfp))

    # fill out the dictionary patch by patch
    for i in range(nf):
        patch = weights[i]

        # get row, col of where the dictionary patch belongs in the dictionary array
        row, col = np.unravel_index(i, (nrows, ncols))

        # normalize
        abs_max = np.max(np.abs(patch))
        if not math.isclose(abs_max, 0):
           patch = patch / abs_max 

        # put processed patch into dictionary image
        dictionary[row*nyp: (row+1)*nyp, col*nxp: (col+1)*nxp] = patch

    # upscale it by a factor of the 'upscaling' input argument
    if upscaling > 0:
      dictionary = dictionary.repeat(upscaling, axis=0).repeat(upscaling, axis=1)
    return dictionary
