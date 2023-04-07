import os
import numpy

def readenergyprobe(probe_name, directory = '.', batch_element = 0):
    """probeData = readenergyprobe(probe_name, directory, batch_element)
    probe_name is the name of the probe as it appears in the output file(s).
    directory is the path to the directory holding the probe's output file(s).
    batch_element is the batch element or list of batch elements to read
      The default batch_element is 0.
    
    The return value is a structure containing two fields, 'time' and 'values'.
    'time' is an N-by-1 numpy array consisting of the timestamps.
    'values' is an N-by-B numpy array where N is the length of the 'time' field
    and B is the length of the batch_element input argument.
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
        filedata = numpy.loadtxt(filepath, delimiter=',', skiprows=1)
        if ib == 0:
            result['time'] = filedata[:,0]
            result['values'] = numpy.zeros([len(result['time']), B])
        result['values'][:, ib] = filedata[:,2]

    return result
