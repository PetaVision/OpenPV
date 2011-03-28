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
              ChannelType channel, const char * filename);
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              ChannelType channel);
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);

   virtual int numDataPatches(int arbor);

   virtual int updateState(float time, float dt){ return 0;};

   virtual int updateWeights(int axonId){ return 0;};

   virtual float minWeight();
   virtual float maxWeight();

   virtual int gauss2DCalcWeights(PVPatch * wp, int kPre, int noPost,
         int numFlanks, float shift, float rotate, float aspect, float sigma,
         float r2Max, float strength, float deltaThetaMax, float thetaMax,
         float bowtieFlag, float bowtieAngle);

   virtual int cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
         float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
         float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
         float aspect, float rotate, float sigma, float r2Max, float strength);

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
