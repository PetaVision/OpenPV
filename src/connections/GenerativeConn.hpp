/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class GenerativeConn : public KernelConn {
public:
    GenerativeConn();
    GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
    GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
            const char * filename);

    int initialize_base();
    int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
            const char * filename=NULL);
#ifdef OBSOLETE
    int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
#endif OBSOLETE
    inline float getRelaxation() { return relaxation; }
    virtual int updateWeights(int axonID);
    virtual int initNormalize();
    virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches);


protected:
    float relaxation;
    bool nonnegConstraintFlag;
    int normalizeMethod;
    float normalizeConstant;
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
