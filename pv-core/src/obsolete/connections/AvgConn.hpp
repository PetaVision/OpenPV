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
           ChannelType channel, HyPerConn * delegate);
   virtual ~AvgConn();

   virtual int createAxonalArbors(int arborId);

   virtual int deliver(Publisher * pub, PVLayerCube * cube, int neighbor);
   virtual int write(const char * filename);

   inline  float getTimeWindow() {return timeWindow;};
   inline  int   getNumLevels() {return numLevels;};
   inline  int   getLastLevel() {return lastLevel;};

   int write_patch_activity(FILE * fp, PVPatch * patch,
                           const PVLayerLoc * loc, int kx0, int ky0, int kf0);

protected:

   int initialize();
   virtual PVPatch *** initializeWeights(PVPatch *** arbors,
                                        int numPatches, const char * filename);

   PVLayerCube * avgActivity;
   HyPerConn   * delegate;

   float      timeWindow;
   int        numLevels;
   int        lastLevel;


   float maxRate;  // maximum expected firing rate of pre-synaptic layer

};

} // namespace PV

#endif /* AVGCONN_HPP_ */
