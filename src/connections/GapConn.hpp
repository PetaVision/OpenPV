/*
 * GapConn.hpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#ifndef GAPCONN_HPP_
#define GAPCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class GapConn: public PV::KernelConn {
public:
   GapConn(const char * name, HyPerCol * hc);
   virtual ~GapConn();
   virtual int allocateDataStructures();
protected:
   GapConn();
   void ioParam_channelCode(enum ParamsIOFlag ioFlag); // No channel argument in params because GapConn must always use CHANNEL_GAP
   void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);

   int initialize(const char * name, HyPerCol * hc);

private:
   int initialize_base();
   bool initNormalizeFlag;

};

} /* namespace PV */
#endif /* GAPCONN_HPP_ */
