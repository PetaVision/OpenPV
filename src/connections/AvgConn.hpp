/*
 * AvConn.hpp
 *
 *  Created on: Oct 9, 2009
 *      Author: rasmussn
 */

#ifndef AVGCONN_HPP_
#define AVGCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class AvgConn: public PV::HyPerConn {
public:

   AvgConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
           int channel, HyPerConn * delegate);
   virtual ~AvgConn();

   virtual int createAxonalArbors();

   virtual int deliver(Publisher * pub, PVLayerCube * cube, int neighbor);
   virtual int write(const char * filename);

protected:

   int initialize();
   virtual PVPatch ** initializeWeights(PVPatch ** patches,
                                        int numPatches, const char * filename);

   PVLayerCube * avgActivity;
   HyPerConn   * delegate;

   float maxRate;  // maximum expected firing rate of pre-synaptic layer
};

} // namespace PV

#endif /* AVGCONN_HPP_ */
