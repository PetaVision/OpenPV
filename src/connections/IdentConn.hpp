/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "KernelConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class InitIdentWeights;

class IdentConn : public KernelConn {
public:
    IdentConn();
    IdentConn(const char * name, HyPerCol *hc,
            HyPerLayer * pre, HyPerLayer * post);

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, const char * filename);
   virtual int setParams(PVParams * inputParams);
   virtual int updateWeights(int axonID) {return PV_SUCCESS;}
   virtual int initShrinkPatches();

protected:
    int setPatchSize(const char * filename);
    virtual int initNormalize();
};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
