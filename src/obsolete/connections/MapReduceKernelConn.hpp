/*
 * MapReduceKernelConn.hpp
 *
 *  Created on: Aug 16, 2013
 *      Author: garkenyon
 */

#ifndef MAPREDUCEKERNELCONN_HPP_
#define MAPREDUCEKERNELCONN_HPP_

#include "HyPerConn.hpp"
#include "../layers/Movie.hpp"

namespace PV {

class MapReduceKernelConn: public PV::HyPerConn {
public:
	MapReduceKernelConn();
	virtual ~MapReduceKernelConn();
    MapReduceKernelConn(const char * name, HyPerCol * hc);
    static const int MAX_WEIGHT_FILES = 1024;  // arbitrary limit...

protected:
    int initialize(const char * name, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
    virtual void ioParam_movieLayerName(enum ParamsIOFlag ioFlag);
    virtual void ioParam_dWeightsListName(enum ParamsIOFlag ioFlag);
    virtual void ioParam_num_dWeightFiles(enum ParamsIOFlag ioFlag);
    virtual void ioParam_dWeightFileIndex(enum ParamsIOFlag ioFlag);
	virtual int communicateInitInfo();
	virtual int reduceKernels(int arborID);
private:
	int initialize_base();
	char * dWeightsListName;
	char * dWeightsFilename;
	char dWeightsList[MAX_WEIGHT_FILES][PV_PATH_MAX];
	int num_dWeightFiles;
	int dWeightFileIndex;
	char * movieLayerName;
	Movie * movieLayer;

};

} /* namespace PV */
#endif /* MAPREDUCEKERNELCONN_HPP_ */
