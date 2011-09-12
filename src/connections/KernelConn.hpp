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

   virtual int numDataPatches();

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

   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches, int arborId);
   virtual PVPatch ** symmetrizeWeights(PVPatch ** patches, int numPatches, int arborId);

   PVPatch * getKernelPatch(int axonId, int kernelIndex)   {return kernelPatches[axonId][kernelIndex];}
   PVPatch ** getKernelPatches(int axonId)   {return kernelPatches[axonId];}
   inline void setKernelPatches(PVPatch** newKernelPatch, int axonId) {kernelPatches[axonId]=newKernelPatch;}
   inline void setKernelPatch(PVPatch* newKernelPatch, int axonId, int kernelIndex) {kernelPatches[axonId][kernelIndex]=newKernelPatch;}
   virtual int writeWeights(float time, bool last=false);
   inline PVPatch *** getAllKernelPatches() {return kernelPatches;}

   bool getPlasticityFlag() {return plasticityFlag;}
   float getWeightUpdatePeriod() {return weightUpdatePeriod;}
   float getWeightUpdateTime() {return weightUpdateTime;}
   float getLastUpdateTime() {return lastUpdateTime;}

   virtual int correctPIndex(int patchIndex);

protected:
//   bool plasticityFlag;
   float weightUpdatePeriod;
   float weightUpdateTime;
   float lastUpdateTime;
   bool symmetrizeWeightsFlag;
   PVPatch ** tmpPatch;  // stores most recently allocated PVPatch **, set to NULL after assignment


private:
   //made private to control use and now 3D to allow different Kernel patches
   //for each axon:
   PVPatch *** kernelPatches;   // list of kernel patches


protected:
   PVPatch *** dKernelPatches;   // list of dKernel patches for storing changes in kernel strengths

#ifdef PV_USE_MPI
   pvdata_t * mpiReductionBuffer;
#endif // PV_USE_MPI

   virtual int deleteWeights();
   virtual int initialize_base();
   virtual int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
   virtual int initialize(const char * name, HyPerCol * hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit);
   virtual int createArbors();
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   virtual int initializeUpdateTime(PVParams * params);
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, int numPatches,
         const char * filename);
   virtual int calc_dW(int axonId);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);
   virtual float computeNewWeightUpdateTime(float time, float currentUpdateTime);
#ifdef PV_USE_MPI
   virtual int reduceKernels(int axonID);
#endif // PV_USE_MPI
   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
                                     const char * filename);
   virtual int setWPatches(PVPatch ** patches, int arborId);
   virtual int setdWPatches(PVPatch ** patches, int arborId);
};

}

#endif /* KERNELCONN_HPP_ */
