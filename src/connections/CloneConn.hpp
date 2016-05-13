/*
 * CloneConn.hpp
 *
 *  Created on: May 24, 2011
 *      Author: peteschultz
 */

#ifndef CLONECONN_HPP_
#define CLONECONN_HPP_

#include "HyPerConn.hpp"
#include "../weightinit/InitCloneKernelWeights.hpp"

namespace PV {

class CloneConn : public HyPerConn {

public:
   CloneConn(const char * name, HyPerCol * hc);
   virtual ~CloneConn();

   virtual int communicateInitInfo();

   virtual int updateState(double time, double dt);

   virtual int writeWeights(double time, bool last=false){return PV_SUCCESS;}
   virtual int writeWeights(const char * filename){return PV_SUCCESS;}
   virtual int checkpointWrite(const char * cpDir){return PV_SUCCESS;}
   virtual int checkpointRead(const char * cpDir, double *timef){return PV_SUCCESS;}
   virtual int outputState(double time, bool last = false){return PV_SUCCESS;}

   HyPerConn * getOriginalConn(){return originalConn;}

   virtual int allocateDataStructures();
   virtual int finalizeUpdate(double timed, double dt);
   //virtual void initPatchToDataLUT();

   virtual long * getPostToPreActivity(){
      return originalConn->getPostToPreActivity();
   }

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
#ifdef PV_USE_OPENCL
   virtual CLBuffer * getDeviceWData(){
#endif
#ifdef PV_USE_CUDA
   virtual PVCuda::CudaBuffer * getDeviceWData(){
#endif
      return originalConn->getDeviceWData();
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   virtual PVCuda::CudaBuffer * getCudnnWData(){
      return originalConn->getCudnnWData();
   }
#endif

   //If this layer needs to allocate device weights, set orig conn's alloc weights
   virtual void setAllocDeviceWeights(){
      originalConn->setAllocDeviceWeights();
   }
   //Vice versa
   virtual void setAllocPostDeviceWeights(){
      originalConn->setAllocPostDeviceWeights();
   }
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

protected:

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int allocatePostDeviceWeights();
   virtual int allocateDeviceWeights();
#endif

   CloneConn();
   virtual int allocatePostConn();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerFlag() {/* deprecated as of Aug 17, 2015.  See HyPerConn::ioParam_triggerFlag. */}
   virtual void ioParam_triggerLayerName() {triggerFlag = false; triggerLayerName = NULL;}
   virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);
   virtual int setWeightInitializer();
   virtual PVPatch *** initializeWeights(PVPatch *** patches, pvdata_t ** dataStart);
   virtual int cloneParameters();
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr) { return PV_SUCCESS; }
   virtual int constructWeights();
   void constructWeightsOutOfMemory();
   virtual int createAxonalArbors(int arborId);

   virtual int  setPatchSize(); // virtual int setPatchSize(const char * filename); // filename is now a member variable.

   char * originalConnName;
   HyPerConn * originalConn;

private:
   int initialize_base();
   int deleteWeights();

}; // end class CloneConn

BaseObject * createCloneConn(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* CLONECONN_HPP_ */
