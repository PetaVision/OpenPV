/*
 * LineSegments.hpp
 *
 *  Created on: Jan 26, 2009
 *      Author: rasmussn
 */
#ifdef OBSOLETE // Use KernelConn or HyperConn and set the param "weightInitType" to "SubUnitWeight" in the params file

#ifndef LINESEGMENTS_HPP_
#define LINESEGMENTS_HPP_

#include "HyPerConn.hpp"

namespace PV {

class SubunitConn: public PV::HyPerConn {
public:
   SubunitConn(const char * name,
               HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);

   static int weights(PVPatch * wp);

protected:

   virtual int initializeWeights(const char * filename);

};

} // namespace PV

#endif /* LINESEGMENTS_HPP_ */
#endif
