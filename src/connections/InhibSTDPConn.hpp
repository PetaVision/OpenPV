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
   InhibSTDPConn(const char * name, HyPerCol * hc);

   virtual int updateWeights(int axonID);

protected:
   int initialize(const char * name, HyPerCol * hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauO(enum ParamsIOFlag ioFlag) {}
   virtual void ioParam_ojaFlag(enum ParamsIOFlag ioFlag) {}
   virtual void ioParam_wMax(enum ParamsIOFlag ioFlag) {}
};

}

#endif /* INHSTDPCONN_H_ */
