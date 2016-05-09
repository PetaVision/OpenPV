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

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   //If this layer needs to allocate device weights, set orig conn's alloc post weights
   virtual void setAllocDeviceWeights(){
      originalConn->setAllocPostDeviceWeights();
   }
   //Vice versa
   virtual void setAllocPostDeviceWeights(){
      originalConn->setAllocDeviceWeights();
   }
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

   virtual long * getPostToPreActivity(){
      return originalConn->postConn->getPostToPreActivity();
   }

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
#ifdef PV_USE_OPENCL
   virtual CLBuffer * getDeviceWData(){
#endif
#ifdef PV_USE_CUDA
   virtual PVCuda::CudaBuffer * getDeviceWData(){
#endif
      return originalConn->postConn->getDeviceWData();
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   virtual PVCuda::CudaBuffer * getCudnnWData(){
      return originalConn->postConn->getCudnnWData();
   }
#endif
#endif

protected:

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
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
#ifdef OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.
    virtual InitWeights * handleMissingInitWeights(PVParams * params);
#endif // OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.
    virtual int setInitialValues();
    virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvwdata_t ** dataStart);
    //int transpose(int arborId);
    virtual int calc_dW(int arborId){return PV_BREAK;};
    //virtual int reduceKernels(int arborID);
    virtual int constructWeights();
    virtual int allocatePostConn();

    // TransposeConn does not need to checkpoint; instead it gets its weights from the originalConn.
    virtual int checkpointWrite(const char * cpDir){return PV_SUCCESS;};
    virtual int checkpointRead(const char * cpDir, double *timef){return PV_SUCCESS;};

private:
    //int transposeSharedWeights(int arborId);
    //int transposeNonsharedWeights(int arborId);
    int deleteWeights();

    /**
     * Calculates the parameters of the the region that needs to be sent to adjoining processes using MPI.
     * Used only in the sharedWeights=false case, because in that case an individual weight's pre and post neurons can live in different processes.
     */
    //int mpiexchangesize(int neighbor, int * size, int * startx, int * stopx, int * starty, int * stopy, int * blocksize, size_t * buffersize);
// Member variables
protected:
    char * originalConnName;
    HyPerConn * originalConn;
}; // end class TransposeConn

BaseObject * createTransposeConn(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
