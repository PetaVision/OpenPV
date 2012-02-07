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
           ChannelType channel, InitWeights *weightInit);
   PoolingGenConn(const char * name, HyPerCol * hc,
           HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
           ChannelType channel, const char * filename);
   PoolingGenConn(const char * name, HyPerCol * hc,
           HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
           ChannelType channel, const char * filename, InitWeights *weightInit);

    HyPerLayer * getPre2() { return pre2; }
    HyPerLayer * getPost2() { return post2; }

    int updateWeights(int axonID);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, HyPerLayer * pre2, HyPerLayer * post2,
            ChannelType channel, const char * filename=NULL, InitWeights *weightInit=NULL);
    bool checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2);
    int getSlownessLayer(HyPerLayer ** l, const char * paramname);
    HyPerLayer * pre2;
    HyPerLayer * post2;
    bool slownessFlag;
    HyPerLayer * slownessPre;
    HyPerLayer * slownessPost;
};  // end class PoolingGenConn

}  // end namespace PV

#endif /* GENPOOLCONN_HPP_ */
