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
    TransposeConn(const char * name, HyPerCol * hc, HyPerLayer * preLayer, HyPerLayer * postLayer, KernelConn * auxConn);
    virtual ~TransposeConn();
    inline KernelConn * getOriginalConn() {return originalConn;}

    virtual int updateWeights(int axonId);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc, HyPerLayer * preLayer, HyPerLayer * postLayer, KernelConn * auxConn);
    virtual void readNumAxonalArborLists(PVParams * params);
    virtual int  readNxp(PVParams * params);
    virtual int  readNyp(PVParams * params);
    virtual int  readNfp(PVParams * params);
    virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches, const char * filename);
    virtual InitWeights * handleMissingInitWeights(PVParams * params);
    int setPatchSize(const char * filename);
    int transposeKernels();
    virtual int calc_dW(int arborId){return PV_BREAK;};
    KernelConn * originalConn;
};

}  // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
