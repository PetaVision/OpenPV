import os
import numpy

def readlayerprobe(probe_name, directory = '.', batch_element = 0):
    """probeData = readlayerprobe(probe_name, directory, batch_element)
    probe_name is the name of the probe as it appears in the output file(s).
    directory is the path to the directory holding the probe's output file(s).
    batch_element is the batch element or list of batch elements to read
      The default batch_element is 0.
    
    The return value is a structure containing three fields,
    'time', 'values', and 'num_neurons'.
    'time' is an N-by-1 numpy array consisting of the timestamps.
    'values' is an N-by-B numpy array where N is the length of the 'time' field
      and B is the length of the batch_element input argument.
    'num_neurons' is an integer giving the number of neurons in the layer.
    """
    result = {}
    if isinstance(batch_element, int):
        batch_list = [batch_element]
    elif isinstance(batch_element, list):
        batch_list = batch_element
    elif isinstance(batch_element, range):
        batch_list = list(batch_element)
    B = len(batch_list)
    for ib in range(B):
        b = batch_list[ib]
        filename = probe_name + "_batchElement_" + str(b) + ".txt"
        filepath = os.path.join(directory, filename)
        filedata = numpy.loadtxt(filepath, delimiter=',')
        if ib == 0:
            result['time'] = filedata[:,0]
            result['values'] = numpy.zeros([len(result['time']), B])
            result['num_neurons'] = filedata[1, 2]
        result['values'][:, ib] = filedata[:,3]

    return result
