/*
 * TransposePoolingConn.cpp
 *
 *  Created on: March 25, 2015
 *      Author: slundquist
 */

#ifndef TRANSPOSEPOOLINGCONN_HPP_
#define TRANSPOSEPOOLINGCONN_HPP_

#include "HyPerConn.hpp"
#include "PoolingConn.hpp"
#include "../layers/PoolingIndexLayer.hpp"

namespace PV {

class TransposePoolingConn: public HyPerConn {
public:
   TransposePoolingConn();
   TransposePoolingConn(const char * name, HyPerCol * hc);
   virtual ~TransposePoolingConn();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   inline PoolingConn * getOriginalConn() {return originalConn;}

   virtual bool needUpdate(double timed, double dt);
   virtual int updateState(double time, double dt);
   virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime);
   //virtual int finalizeUpdate(double time, double dt);
   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID);
   virtual int deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int deliverPresynapticPerspectiveGPU(PVLayerCube const * activity, int arborID) {
      std::cout << "TransposePoolingConn not implemented on GPUs\n";
      exit(-1);
   }
   virtual int deliverPostsynapticPerspectiveGPU(PVLayerCube const * activity, int arborID) {
      std::cout << "TransposePoolingConn not implemented on GPUs\n";
      exit(-1);
   }
#endif
   virtual int checkpointRead(const char * cpDir, double * timeptr);
   virtual int checkpointWrite(const char * cpDir);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
    virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
    virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
    virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
    virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
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
    PoolingConn * originalConn;
}; // end class TransposePoolingConn

BaseObject * createTransposePoolingConn(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
