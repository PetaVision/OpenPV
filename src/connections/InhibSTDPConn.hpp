/*
 * InhibSTDPConn.h
 *
 *  Created on: Mar 21, 2013
 *      Author: dpaiton
 */

#ifndef INHSTDPCONN_H_
#define INHSTDPCONN_H_

#include "OjaSTDPConn.hpp"

//#define SPLIT_PRE_POST
#undef SPLIT_PRE_POST

namespace PV {

class InhibSTDPConn: public PV::OjaSTDPConn {
public:
   InhibSTDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            const char * filename=NULL, InitWeights *weightInit=NULL);

   virtual int updateWeights(int axonID);

protected:
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  const char * filename, InitWeights *weightInit);

   int setParams(PVParams * params);

   virtual void readTauOja(PVParams * params) {}
   virtual void readOjaFlag(PVParams * params) {}
   virtual void readWMax(PVParams * params) {}
};

}

#endif /* INHSTDPCONN_H_ */
