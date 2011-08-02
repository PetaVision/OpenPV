/*
 * ODDConn.hpp
 *
 *  Created on: ?
 *      Author: Kenyon
 */

#ifndef ODDCONN_HPP_
#define ODDCONN_HPP_

#undef APPLY_ODD_WEIGHTS

#include "../connections/KernelConn.hpp"

namespace PV {

class ODDConn: public KernelConn {
private:

public:

   ODDConn();
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename);
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel);
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int arbor);
   virtual 	int writeWeights(float time, bool last);
   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches);

protected:
   PVPatch ** geislerPatches;   // list of kernels patches for accumulating pairwise stats
   pvdata_t * avePostActivity;
   pvdata_t * avePreActivity;
   int numUpdates;
   virtual int deleteWeights();
   virtual int initialize_base();
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);


};

}

#endif /* GeislerConn_HPP_ */
