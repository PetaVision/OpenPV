/*
 * ConvolveConn.hpp
 *
 *  Created on: Oct 5, 2009
 *      Author: rasmussn
 */

#ifndef CONVOLVECONN_HPP_
#define CONVOLVECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class ConvolveConn: public PV::HyPerConn {
public:
   ConvolveConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         ChannelType channel, InitWeights *weightInit=NULL);

   virtual int deliver(PVLayerCube * cube, int neighbor);
   void convolve(PVLayerCube * dst, PVLayerCube * src, PVPatch * patch);

protected:
   PVPatch patch;

   virtual int initialize(const char * filename);

};

} // namespace PV

#endif /* CONVOLVECONN_HPP_ */
