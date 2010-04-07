/*
 * CocircConn.h
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#ifndef GEISLERCONN_HPP_
#define GEISLERCONN_HPP_

#undef APPLY_GEISLER_WEIGHTS

#include "../PetaVision/src/connections/KernelConn.hpp"

namespace PV {

class GeislerConn: public KernelConn {
private:

public:

   GeislerConn();
   GeislerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         int channel, const char * filename);
   GeislerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
         int channel);
   GeislerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int arbor);
   virtual 	int writeWeights(float time, bool last);

protected:
   PVPatch ** geislerPatches;   // list of kernels patches for accumulating pairwise stats
   pvdata_t avePostActivity;
   pvdata_t avePreActivity;
   int numUpdates;
   virtual int deleteWeights();
   virtual int initialize_base();
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);


};

}

#endif /* GeislerConn_HPP_ */
