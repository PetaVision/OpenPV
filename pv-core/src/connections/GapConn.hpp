/*
 * GapConn.hpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#ifndef GAPCONN_HPP_
#define GAPCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class GapConn: public PV::HyPerConn {
public:
   GapConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~GapConn();
   virtual int allocateDataStructures();
protected:
   GapConn();
   void ioParam_channelCode(enum ParamsIOFlag ioFlag); // No channel argument in params because GapConn must always use CHANNEL_GAP
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);

   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);

private:
   int initialize_base();
   bool initNormalizeFlag;

}; // end class GapConn

BaseObject * createGapConn(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* GAPCONN_HPP_ */
