/*
 * CloneConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef CLONECONN_HPP_
#define CLONECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class CloneConn : public HyPerConn {

  public:
   CloneConn(const char *name, HyPerCol *hc);
   virtual ~CloneConn();

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual int updateState(double time, double dt) override;

   virtual int writeWeights(double time) override { return PV_SUCCESS; }
   virtual int writeWeights(const char *filename, bool verifyWrites) override { return PV_SUCCESS; }
   virtual int outputState(double time) override { return PV_SUCCESS; }

   HyPerConn *getOriginalConn() { return originalConn; }

   virtual int allocateDataStructures() override;
   virtual int finalizeUpdate(double timed, double dt) override;
   // virtual void initPatchToDataLUT();

   virtual long *getPostToPreActivity() override { return originalConn->getPostToPreActivity(); }

#ifdef PV_USE_CUDA
   virtual PVCuda::CudaBuffer *getDeviceWData() override { return originalConn->getDeviceWData(); }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   virtual PVCuda::CudaBuffer *getCudnnWData() override { return originalConn->getCudnnWData(); }
#endif

   // If this layer needs to allocate device weights, set orig conn's alloc weights
   virtual void setAllocDeviceWeights() override { originalConn->setAllocDeviceWeights(); }
   // Vice versa
   virtual void setAllocPostDeviceWeights() override { originalConn->setAllocPostDeviceWeights(); }
#endif // PV_USE_CUDA

  protected:
#ifdef PV_USE_CUDA
   virtual int allocatePostDeviceWeights() override;
   virtual int allocateDeviceWeights() override;
#endif

   CloneConn();
   virtual int allocatePostConn() override;
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;
   virtual void
   ioParam_triggerFlag() { /* deprecated as of Aug 17, 2015.  See HyPerConn::ioParam_triggerFlag. */
   }
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;
   virtual PVPatch ***initializeWeights(PVPatch ***patches, float **dataStart) override;
   virtual int cloneParameters();
   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override { return PV_SUCCESS; }
   virtual int constructWeights() override;
   void constructWeightsOutOfMemory();
   virtual int createAxonalArbors(int arborId);

   virtual int setPatchSize() override;

   virtual int registerData(Checkpointer *checkpointer) override;

   char *originalConnName;
   HyPerConn *originalConn;

  private:
   int initialize_base();
   int deleteWeights();

}; // end class CloneConn

} // end namespace PV

#endif /* CLONECONN_HPP_ */
