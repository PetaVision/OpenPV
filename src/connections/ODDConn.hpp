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

   // ODDConn();  // default constructor not necessary?
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename);
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename, InitWeights *weightInit);
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel);
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, InitWeights *weightInit);
#ifdef OBSOLETE // marked obsolete Jul 21, 2011.  No routine calls it, and it doesn't make sense to define a connection without specifying a channel.
   ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
#endif
   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int arbor);
   virtual int writeWeights(float time, bool last);
   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

protected:
   PVPatch *** ODDPatches;   // list of kernels patches for accumulating pairwise stats
   pvdata_t * avePostActivity;
   pvdata_t * avePreActivity;
   int numUpdates;
   virtual int deleteWeights();
   virtual int initialize_base();
   virtual int createArbors();
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);


};  // class ODDConn

} // namespace PV

#endif /* GeislerConn_HPP_ */
