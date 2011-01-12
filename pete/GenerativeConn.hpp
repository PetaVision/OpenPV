/*
 * GenerativeConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef GENERATIVECONN_HPP_
#define GENERATIVECONN_HPP_

#include <assert.h>
#include "../PetaVision/src/connections/KernelConn.hpp"
#include "../PetaVision/src/columns/HyPerCol.hpp"

namespace PV {

class GenerativeConn : public KernelConn {
public:
    GenerativeConn();
    GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post);
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
    inline float getWeightUpdatePeriod() {return weightUpdatePeriod;}
    inline float getNextWeightUpdate() { return nextWeightUpdate;}
    inline float getRelaxation() { return relaxation; }
    virtual int updateState(float time, float dt);
    int updateWeights(int axonID);


protected:
    float weightUpdatePeriod;
    float nextWeightUpdate;
    float relaxation;
};

}  // end of block for namespace PV

#endif /* GENERATIVECONN_HPP_ */
