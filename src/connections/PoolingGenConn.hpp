/*
 * PoolingGenConn.hpp
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef POOLINGGENCONN_HPP_
#define POOLINGGENCONN_HPP_

#include "GenerativeConn.hpp"

namespace PV {

class PoolingGenConn : public GenerativeConn {
public:
    PoolingGenConn(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
            ChannelType channel);
    PoolingGenConn(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
            ChannelType channel, const char * filename);

    HyPerLayer * getPre2() { return pre2; }
    HyPerLayer * getPost2() { return post2; }

    int updateWeights(int axonID);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
            ChannelType channel, const char * filename);
    int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
            ChannelType channel);
    bool checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2);
    HyPerLayer * pre2;
    HyPerLayer * post2;
};  // end class PoolingGenConn

}  // end namespace PV

#endif /* GENPOOLCONN_HPP_ */
