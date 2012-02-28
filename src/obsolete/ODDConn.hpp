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
   virtual ~ODDConn();

   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int arbor);
   virtual int writeWeights(float time, bool last);
   virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

protected:
   PVPatch *** ODDPatches;   // list of kernels patches for accumulating pairwise stats
   pvdata_t * avePostActivity;
   pvdata_t * avePreActivity;
   int numUpdates;
   // virtual int deleteWeights(); // Changed to a private method.  Should not be virtual since it's called from the destructor.
   virtual int initialize_base();
   virtual int createArbors();
   virtual pvdata_t * createWeights(PVPatch *** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);

private:
   int deleteWeights();

};  // class ODDConn

} // namespace PV

#endif /* GeislerConn_HPP_ */
