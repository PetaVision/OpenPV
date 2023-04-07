import numpy

def weights_tableau(w):
    """Take a numpatches-by-nxp-by-nyp-by-nfp array representing a
    dictionary of weight patches and arrange them as an
    (M*nxp)-by-(N*nyp)-by-nfp array.
    M and N are chosen automatically to be close to sqrt(numpatches)
    and so that M*N <= numpatches.
    """
    if (not isinstance(w, numpy.ndarray)) or w.ndim != 4:
        raise Exception("weight_tableau argument is not a 4-dimensional array")

    (numpatches, nxp, nyp, nfp) = w.shape
    num_patch_rows = int(numpy.floor(numpy.sqrt(numpatches)));
    num_patch_columns = int(numpy.ceil(numpatches / num_patch_rows));

    weight_patch_array = numpy.zeros([num_patch_rows * nyp, num_patch_columns * nxp, nfp])
    for p in range(numpatches):
        patch_p = w[p]
        min_patch = patch_p.min()
        max_patch = patch_p.max()
        rescaled_patch = (patch_p - min_patch) / (max_patch - min_patch)
        col_index = p % num_patch_columns
        row_index = (p - col_index) // num_patch_columns
        col_patch_start = col_index * nxp
        row_patch_start = row_index * nyp
        weight_patch_array[row_patch_start:(row_patch_start + nyp), col_patch_start:(col_patch_start + nxp), :] = rescaled_patch

    weight_patch_8bit = numpy.uint8(255 * weight_patch_array)
    return weight_patch_8bit
