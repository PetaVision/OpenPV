import pvtools as pv
import numpy as np
import scipy.sparse as sp
import pdb


datasetDir = "/home/sheng/mountData/datasets/cifar/cifar-10-batches-py/"
outputDir = "/home/sheng/mountData/datasets/cifar/pvp/"

trainFiles = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
testFiles = "test_batch"


def unpackFile(filename):
    import cPickle
    fo = open(filename, 'rb')
    output = cPickle.load(fo)
    fo.close()

    data = output["data"]
    label = output["labels"]

    (numInstances, numFeatures) = data.shape
    #cifar data is 32x32, with 3 color channels
    dataMat = np.reshape(data, [numInstances, 3, 32, 32])

    #Permute to petavision shape, where nf is the fastest spinning dimension
    dataMat = np.transpose(dataMat, (0, 2, 3, 1))

    #Set label to petavision dimensions, where the layer is 1x1x10
    sparseValues = np.ones((numInstances))
    labelMat = sp.coo_matrix((sparseValues, (range(numInstances), label)), shape=(numInstances, 10))

    return (dataMat, labelMat)


if __name__ == "__main__":
    trainData = None
    trainLabel = None
    testData = None
    testLabel = None

    #Get trainData and trainLabel
    for trainfilename in trainFiles:
        infile = datasetDir + trainfilename
        (data, label) = unpackFile(infile)
        if(trainData == None):
            assert(trainLabel == None)
            trainData = data
            trainLabel = label
        else:
            assert(not trainData == None)
            assert(not trainLabel == None)
            trainData = np.concatenate((trainData, data), axis=0)
            trainLabel = sp.vstack([trainLabel, label])

    #Get testData and testLabel
    infile = datasetDir + testFiles
    (testData, testLabel) = unpackFile(infile)

    #Write out to pvp files
    trainDataObj = {}
    trainLabelObj = {}
    testDataObj = {}
    testLabelObj = {}

    trainDataObj["values"] = trainData
    trainDataObj["time"] = range(trainData.shape[0])

    trainLabelObj["values"] = trainLabel
    trainLabelObj["time"] = range(trainData.shape[0])

    testDataObj["values"] = testData
    testDataObj["time"] = range(testData.shape[0])

    testLabelObj["values"] = testLabel
    testLabelObj["time"] = range(testData.shape[0])

    print "Writing train data"
    pv.writepvpfile(outputDir + "cifarTrainData.pvp", trainDataObj)
    print "Writing train label"
    pv.writepvpfile(outputDir + "cifarTrainLabels.pvp", trainLabelObj, shape=(1, 1, 10))
    print "Writing test data"
    pv.writepvpfile(outputDir + "cifarTestData.pvp", testDataObj)
    print "Writing test label"
    pv.writepvpfile(outputDir + "cifarTestLabels.pvp", testLabelObj, shape=(1, 1, 10))






