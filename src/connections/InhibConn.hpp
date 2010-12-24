/*
 * InhibConn.hpp
 *
 *  Created on: Feb 16, 2009
 *      Author: rasmussn
 */

#ifndef INHIBCONN_HPP_
#define INHIBCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class InhibConn: public PV::HyPerConn {
public:
   InhibConn(const char * name,
             HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);

protected:
   virtual int initializeWeights(const char * filename);
   virtual int inhibWeights(PVPatch * wp, int featureIndex, float strength);

private:
   float nfPre;    // number of orientations in pre-synaptic layer
};

} // namespace PV

#endif /* INHIBCONN_HPP_ */
