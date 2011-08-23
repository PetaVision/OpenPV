/*
 * KernelConn.hpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#ifndef KERNELCONN_HPP_
#define KERNELCONN_HPP_

#include "HyPerConn.hpp"
#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

namespace PV {

class KernelConn: public HyPerConn {

public:

   KernelConn();
   virtual ~KernelConn();

   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              ChannelType channel, const char * filename = NULL, InitWeights *weightInit = NULL);
#ifdef OBSOLETE // marked obsolete Jul 25, 2011.  This case covered since other constructor's filename argumernt now has a default of null
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              ChannelType channel);
#endif // OBSOLETE
#ifdef OBSOLETE // marked obsolete Jul 25, 2011.  No routine calls it, and it doesn't make sense to define a connection without specifying a channel.
   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
#endif // OBSOLETE

   virtual int numDataPatches(int arbor);

   virtual float minWeight();
   virtual float maxWeight();

#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
   virtual int gauss2DCalcWeights(PVPatch * wp, int kPre, int noPost,
         int numFlanks, float shift, float rotate, float aspect, float sigma,
         float r2Max, float strength, float deltaThetaMax, float thetaMax,
         float bowtieFlag, float bowtieAngle);

   virtual int cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
         float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
         float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
         float aspect, float rotate, float sigma, float r2Max, float strength);
#endif

   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches);
   virtual PVPatch ** symmetrizeWeights(PVPatch ** patches, int numPatches);

   PVPatch * getKernelPatch(int kernelIndex)   {return kernelPatches[kernelIndex];}
   virtual int writeWeights(float time, bool last=false);

   bool getPlasticityFlag() {return plasticityFlag;}
   float getWeightUpdatePeriod() {return weightUpdatePeriod;}
   float getWeightUpdateTime() {return weightUpdateTime;}
   float getLastUpdateTime() {return lastUpdateTime;}

protected:
   PVPatch ** kernelPatches;   // list of kernel patches
   bool plasticityFlag;
   float weightUpdatePeriod;
   float weightUpdateTime;
   float lastUpdateTime;
   bool symmetrizeWeightsFlag;
#ifdef PV_USE_MPI
   pvdata_t * mpiReductionBuffer;
#endif // PV_USE_MPI

   virtual int deleteWeights();
   virtual int initialize_base();
   virtual int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
   virtual int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit);
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);
   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);
   virtual int initializeUpdateTime(PVParams * params);
   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
         const char * filename);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);
   virtual float computeNewWeightUpdateTime(float time, float currentUpdateTime);
#ifdef PV_USE_MPI
   virtual int reduceKernels(int axonID);
#endif // PV_USE_MPI
   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
                                     const char * filename);
};

}

#endif /* KERNELCONN_HPP_ */
