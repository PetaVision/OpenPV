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
    TransposeConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name, const char * originalConnName);
    virtual ~TransposeConn();
    virtual int communicateInitInfo();
    virtual int allocateDataStructures();
    inline KernelConn * getOriginalConn() {return originalConn;}

    virtual int updateWeights(int axonId);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name, const char * originalConnName);
    virtual void readNumAxonalArbors(PVParams * params);
    virtual int  readPatchSize(PVParams * params);
    virtual int  readNfp(PVParams * params);
    virtual void readPlasticityFlag(PVParams * params);
    virtual void readCombine_dW_with_W_flag(PVParams * params);
    virtual void read_dWMax(PVParams * params);
    virtual void readKeepKernelsSynchronized(PVParams * params);
    virtual void readWeightUpdatePeriod(PVParams * params);
    virtual void readInitialWeightUpdateTime(PVParams * params);
    virtual void readShrinkPatches(PVParams * params);
    virtual int setPatchSize();
    virtual int setNeededRNGSeeds() {return 0;}
    virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches, const char * filename);
    virtual InitWeights * handleMissingInitWeights(PVParams * params);
    int transposeKernels();
    virtual int calc_dW(int arborId){return PV_BREAK;};
    virtual int reduceKernels(int arborID);

// Member variables
protected:
    char * originalConnName;
    KernelConn * originalConn;
};

}  // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
