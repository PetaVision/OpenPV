/*
 * KernelConn.hpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#ifndef KERNELCONN_HPP_
#define KERNELCONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class KernelConn: public HyPerConn {

public:

   KernelConn();

   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              int channel, const char * filename);
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              int channel);
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);

   virtual int numDataPatches(int arbor);

   virtual int kernelIndexToPatchIndex(int kernelIndex);

   virtual int patchIndexToKernelIndex(int patchIndex);

   virtual int updateState(float time, float dt){ return 0;};

   virtual int updateWeights(int axonId){ return 0;};

   virtual float minWeight();
   virtual float maxWeight();

   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches);
   virtual PVPatch ** symmetrizeWeights(PVPatch ** patches, int numPatches);

   PVPatch * getKernelPatch(int kernelIndex)   {return kernelPatches[kernelIndex];}
   virtual int writeWeights(float time, bool last=false);

protected:
   PVPatch ** kernelPatches;   // list of kernel patches
   virtual int deleteWeights();
   virtual int initialize_base();
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);
   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);
   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
         const char * filename);
   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
                                     const char * filename);

};

}

#endif /* KERNELCONN_HPP_ */
