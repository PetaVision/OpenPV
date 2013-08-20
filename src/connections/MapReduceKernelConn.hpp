/*
 * MapReduceKernelConn.hpp
 *
 *  Created on: Aug 16, 2013
 *      Author: garkenyon
 */

#ifndef MAPREDUCEKERNELCONN_HPP_
#define MAPREDUCEKERNELCONN_HPP_

#include "KernelConn.hpp"
#include "../layers/Movie.hpp"

namespace PV {

class MapReduceKernelConn: public PV::KernelConn {
public:
	MapReduceKernelConn();
	virtual ~MapReduceKernelConn();
	MapReduceKernelConn(const char * name, HyPerCol * hc,
			const char * pre_layer_name, const char * post_layer_name,
			const char * filename = NULL, InitWeights *weightInit = NULL, const char * movieLayerName = NULL);
    static const int MAX_WEIGHT_FILES = 1024;  // arbitrary limit...

protected:
	int initialize(const char * name, HyPerCol * hc,
	        const char * pre_layer_name, const char * post_layer_name,
	        const char * filename, InitWeights *weightInit, const char * movieLayerName);
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
