/*
 * VaryingKernelConn.hpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#ifndef VARYINGKERNELCONN_HPP_
#define VARYINGKERNELCONN_HPP_

#include "../PetaVision/src/connections/KernelConn.hpp"

namespace PV {

class VaryingKernelConn : public KernelConn {

public:
   VaryingKernelConn(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename, InitWeights *weightInit);
   virtual ~VaryingKernelConn();
   virtual int allocateDataStructures();

protected:
   int initialize(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename, InitWeights *weightInit=NULL);
   virtual int setParams(PVParams * inputParams);
   virtual void readPlasticityFlag(PVParams * params);
   virtual void readShmget_flag(PVParams * params);
   virtual int calc_dW(int axonId);

}; // end class VaryingKernelConn

}  // end namespace PV block


#endif /* VARYINGKERNELCONN_HPP_ */
