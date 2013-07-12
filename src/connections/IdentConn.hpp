/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "KernelConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class InitIdentWeights;

class IdentConn : public KernelConn {
public:
   IdentConn(const char * name, HyPerCol *hc,
         const char * pre_layer_name, const char * post_layer_name);

   virtual int communicateInitInfo();
   virtual int updateWeights(int axonID) {return PV_SUCCESS;}

protected:
   IdentConn();
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
         const char * pre_layer_name, const char * post_layer_name,
         const char * filename);
   virtual int initNormalize();

   virtual int setParams(PVParams * inputParams);
   virtual void readNumAxonalArbors(PVParams * params);
   virtual void readPlasticityFlag(PVParams * params);
   virtual void readStochasticReleaseFlag(PVParams * params);
   virtual void readPreActivityIsNotRate(PVParams * params);
   virtual void readShrinkPatches(PVParams * params);
   virtual void readWriteCompressedWeights(PVParams * params);
   virtual void readWriteCompressedCheckpoints(PVParams * params);
   virtual void readSelfFlag(PVParams * params);
   virtual void readCombine_dW_with_W_flag(PVParams * params);
   virtual int  readPatchSize(PVParams * params);
   virtual int  readNfp(PVParams * params);
   virtual void readKeepKernelsSynchronized(PVParams * params);
   virtual void readWeightUpdatePeriod(PVParams * params);
   virtual void readInitialWeightUpdateTime(PVParams * params);

};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
