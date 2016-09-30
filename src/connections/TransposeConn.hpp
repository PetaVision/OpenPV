/*
 * TransposeConn.hpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#ifndef TRANSPOSECONN_HPP_
#define TRANSPOSECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class TransposeConn: public HyPerConn {
public:
   TransposeConn();
   TransposeConn(const char * name, HyPerCol * hc);
   virtual ~TransposeConn();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   inline HyPerConn * getOriginalConn() {return originalConn;}

   virtual bool needUpdate(double timed, double dt);
   virtual int updateState(double time, double dt);
   virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime);
   virtual int finalizeUpdate(double time, double dt);

#ifdef PV_USE_CUDA
   //If this layer needs to allocate device weights, set orig conn's alloc post weights
   virtual void setAllocDeviceWeights(){
      originalConn->setAllocPostDeviceWeights();
   }
   //Vice versa
   virtual void setAllocPostDeviceWeights(){
      originalConn->setAllocDeviceWeights();
   }
#endif // PV_USE_CUDA

   virtual long * getPostToPreActivity(){
      return originalConn->postConn->getPostToPreActivity();
   }

#ifdef PV_USE_CUDA
   virtual PVCuda::CudaBuffer * getDeviceWData(){
      return originalConn->postConn->getDeviceWData();
   }
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   virtual PVCuda::CudaBuffer * getCudnnWData(){
      return originalConn->postConn->getCudnnWData();
   }
#endif

protected:

#ifdef PV_USE_CUDA
   virtual int allocatePostDeviceWeights();
   virtual int allocateDeviceWeights();
#endif


    int initialize_base();
    int initialize(const char * name, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
    virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
    virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
    virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
    virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
    virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {/* triggerFlag is deprecated as of Aug 17, 2015.  See HyPerConn::ioParam_triggerFlag for details*/}
    virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
    virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);
    virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag);
    virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
    virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);
    virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
    virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
    virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
    virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
    virtual int setPatchSize();
    virtual int setNeededRNGSeeds() {return 0;}
    virtual int setInitialValues();
    virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvwdata_t ** dataStart);
    virtual int calc_dW(int arborId){return PV_BREAK;};
    virtual int constructWeights();
    virtual int allocatePostConn();

    virtual int checkpointWrite(const char * cpDir){return PV_SUCCESS;};
    virtual int checkpointRead(const char * cpDir, double *timef){return PV_SUCCESS;};

private:
    int deleteWeights();

// Member variables
protected:
    char * originalConnName;
    HyPerConn * originalConn;
}; // end class TransposeConn

}  // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
