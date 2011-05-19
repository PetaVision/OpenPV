/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include "PeriodicUpdateConn.hpp"

namespace PV {

class GenerativeConn : public PeriodicUpdateConn {
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
            const char * filename);
    int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
    inline float getRelaxation() { return relaxation; }
    virtual int updateWeights(int axonID);
    virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches);


protected:
    float relaxation;
    bool nonnegConstraintFlag;
    int normalizeMethod;
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
