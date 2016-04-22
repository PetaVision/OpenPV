/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "HyPerConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class InitIdentWeights;

class IdentConn : public HyPerConn {
public:
   IdentConn(const char * name, HyPerCol *hc);

   virtual int communicateInitInfo();
   virtual int updateWeights(int axonID) {return PV_SUCCESS;}
   //virtual int deliver();

protected:
   IdentConn();
   int initialize_base();
   int initialize(const char * name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag);
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);
   virtual void ioParam_selfFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag);

   void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
   void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);
   void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);

   virtual int setWeightInitializer();

   // IdentConn does not need to checkpoint
   virtual int checkpointRead(const char * cpDir, double* timef) { return PV_SUCCESS; }
   virtual int checkpointWrite(const char * cpDir) { return PV_SUCCESS; }

   virtual void handleDefaultSelfFlag();

   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID);
}; // class IdentConn

BaseObject * createIdentConn(char const * name, HyPerCol * hc);

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
