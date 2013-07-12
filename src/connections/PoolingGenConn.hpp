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
         const char * pre_layer_name, const char * post_layer_name,
         const char * pre_layer_name2, const char * post_layer_name2,
         const char * filename=NULL, InitWeights *weightInit=NULL);

    HyPerLayer * getPre2() { return pre2; }
    HyPerLayer * getPost2() { return post2; }

    virtual int communicateInitInfo();
    int updateWeights(int axonID);

protected:
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc,
          const char * pre_layer_name, const char * post_layer_name,
          const char * pre_layer_name2, const char * post_layer_name2,
          const char * filename=NULL, InitWeights *weightInit=NULL);
    int setParams(PVParams * params);
    void readSlownessFlag(PVParams * params);
    int readSlownessPre(PVParams * params);
    int readSlownessPost(PVParams * params);
    bool checkLayersCompatible(HyPerLayer * layer1, HyPerLayer * layer2);
    int getSlownessLayer(HyPerLayer ** l, const char * paramname);

    char * preLayerName2;
    char * postLayerName2;
    HyPerLayer * pre2;
    HyPerLayer * post2;
    bool slownessFlag;
    HyPerLayer * slownessPre;
    HyPerLayer * slownessPost;
};  // end class PoolingGenConn

}  // end namespace PV

#endif /* GENPOOLCONN_HPP_ */
