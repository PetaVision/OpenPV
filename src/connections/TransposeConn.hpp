/*
 * TransposeConn.hpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#ifndef TRANSPOSECONN_HPP_
#define TRANSPOSECONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class TransposeConn: public KernelConn {
public:
    TransposeConn();
    TransposeConn(const char * name, HyPerCol * hc);
    virtual ~TransposeConn();
    virtual int communicateInitInfo();
    virtual int allocateDataStructures();
    inline KernelConn * getOriginalConn() {return originalConn;}

    virtual int updateWeights(int axonId);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
    virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
    virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
    virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag);
    virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);
    virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag);
    virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
    virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);
    virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
    virtual void ioParam_useWindowPost(enum ParamsIOFlag ioFlag);
    virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
    virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
    virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
    virtual int setPatchSize();
    virtual int setNeededRNGSeeds() {return 0;}
    virtual InitWeights * handleMissingInitWeights(PVParams * params);
    virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvwdata_t ** dataStart,
                                          int numPatches);
    int transposeKernels(int arborId);
    virtual int calc_dW(int arborId){return PV_BREAK;};
    virtual int reduceKernels(int arborID);

// Member variables
protected:
    char * originalConnName;
    KernelConn * originalConn;
};

}  // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
