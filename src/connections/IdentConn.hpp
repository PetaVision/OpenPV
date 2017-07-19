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
   IdentConn(const char *name, HyPerCol *hc);

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int updateWeights(int axonID) override { return PV_SUCCESS; }
   // virtual int deliver();

  protected:
   IdentConn();
   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);

#ifdef PV_USE_CUDA
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;
#endif // PV_USE_CUDA
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_selfFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) override;

   void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) override;
   void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) override;

   // IdentConn does not need to checkpoint

   virtual void handleDefaultSelfFlag() override;

   virtual int registerData(Checkpointer *checkpointer) override;

   virtual int deliverPresynapticPerspective(PVLayerCube const *activity, int arborID) override;
}; // class IdentConn

} // end of block for namespace PV

#endif /* IDENTCONN_HPP_ */
