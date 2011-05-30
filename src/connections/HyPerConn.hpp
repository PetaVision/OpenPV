/*
 * HyPerConnection.hpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "../columns/InterColComm.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../utils/Timer.hpp"

#define PROTECTED_NUMBER 13
#define MAX_ARBOR_LIST (1+MAX_NEIGHBORS)

namespace PV {

class HyPerCol;
class HyPerLayer;
class ConnectionProbe;

/**
 * A PVConnection identifies a connection between two layers
 */
typedef struct {
   int delay; // current output delay in the associated f ring buffer (should equal fixed delay + varible delay for valid connection)
   int fixDelay; // fixed output delay. TODO: should be float
   int varDelayMin; // minimum variable conduction delay
   int varDelayMax; // maximum variable conduction delay
   int numDelay;
   int isGraded; //==1, release is stochastic with prob = (activity <= 1), default is 0 (no graded release)
   float vel;  // conduction velocity in position units (pixels) per time step--added by GTK
   float rmin; // minimum connection distance
   float rmax; // maximum connection distance
} PVConnParams;

class HyPerConn {

public:
   HyPerConn();
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, const char * filename);
   virtual ~HyPerConn();

   virtual int deliver(Publisher * pub, PVLayerCube * cube, int neighbor);

   virtual int insertProbe(ConnectionProbe * p);
   virtual int outputState(float time, bool last=false);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);

   inline  int numberOfAxonalArborLists()            {return numAxonalArborLists;}
   virtual int numWeightPatches(int arbor);
   virtual int numDataPatches(int arbor);
   virtual int writeWeights(float time, bool last=false);
   virtual int writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last);
   virtual int writeTextWeights(const char * filename, int k);
   virtual int writePostSynapticWeights(float time, bool last=false);

   int readWeights(const char * filename);
   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
                                  const char * filename);

   virtual PVPatch * getWeights(int kPre, int arbor);
#ifdef OBSOLETE_STDP
   virtual PVPatch * getPlasticityIncrement(int k, int arbor);

   inline PVLayerCube * getPlasticityDecrement()     {return pDecr;}
#endif

   inline PVPatch ** weights(int neighbor)           {return wPatches[neighbor];}

   inline const char * getName()                     {return name;}
   inline int          getDelay()                    {return params->delay;}

   virtual float minWeight()                          {return 0.0;}
   virtual float maxWeight()                          {return wMax;}

   inline int xPatchSize()                           {return nxp;}
   inline int yPatchSize()                           {return nyp;}
   inline int fPatchSize()                           {return nfp;}

   inline PVAxonalArbor * axonalArbor(int kPre, int neighbor)
      {return &axonalArborList[neighbor][kPre];}

   HyPerLayer * preSynapticLayer()     {return pre;}
   HyPerLayer * postSynapticLayer()    {return post;}

   int  getConnectionId()              {return connId;}
   void setConnectionId(int id)        {connId = id;}

   int setParams(PVParams * params, PVConnParams * p);

   PVPatch ** convertPreSynapticWeights(float time);

   int preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int * kxPre, int * kyPre);
   int postSynapticPatchHead(int kPre,
                             int * kxPostOut, int * kyPostOut, int * kfPostOut,
                             int * dxOut, int * dyOut, int * nxpOut, int * nypOut);

   virtual int gauss2DCalcWeights(PVPatch * wp, int kPre, int noPost,
                             int numFlanks, float shift, float rotate, float aspect, float sigma,
                             float r2Max, float strength, float deltaThetaMax, float thetaMax,
                             float bowtieFlag, float bowtieAngle);

   virtual int cocircCalcWeights(PVPatch * wp, int kPre, int noPre, int noPost,
         float sigma_cocirc, float sigma_kurve, float sigma_chord, float delta_theta_max,
         float cocirc_self, float delta_radius_curvature, int numFlanks, float shift,
         float aspect, float rotate, float sigma, float r2Max, float strength);

   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches);

   virtual int kernelIndexToPatchIndex(int kernelIndex, int * kxPatchIndex = NULL,
         int * kyPatchIndex = NULL, int * kfPatchIndex = NULL);

   virtual int patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex = NULL,
         int * kyKernelIndex = NULL, int * kfKernelIndex = NULL);

protected:
   HyPerLayer     * pre;
   HyPerLayer     * post;
   HyPerCol       * parent;
#ifdef OBSOLETE_STDP
   PVLayerCube    * pDecr;      // plasticity decrement variable (Mi) for post-synaptic layer
   PVPatch       ** pIncr;      // list of stdp patches Psij variable
#endif
   PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
   PVPatch       ** wPostPatches;  // post-synaptic linkage of weights
   PVAxonalArbor  * axonalArborList[MAX_ARBOR_LIST]; // list of axonal arbors for each neighbor

#ifdef OBSOLETE_STDP
   bool     localWmaxFlag;  // presence of rate dependent wMax;
   pvdata_t * Wmax;   // adaptive upper STDP weight boundary
#endif

   ChannelType channel;    // which channel of the post to update (e.g. inhibit)
   int connId;             // connection id

   char * name;
   int nxp, nyp, nfp;      // size of weight dimensions

   int numParams;
   PVConnParams * params;

   int numAxonalArborLists;  // number of axonal arbors (weight patches) for presynaptic layer

#ifdef OBSOLETE_STDP
   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float ampLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float dWMax;
#endif
   float wMax;
   float wMin;

   int numProbes;
   ConnectionProbe ** probes; // probes used to output data

#ifdef OBSOLETE_STDP
   bool stdpFlag;               // presence of spike timing dependent plasticity
#endif
   bool ioAppend;               // controls opening of binary files
   float wPostTime;             // time of last conversion to wPostPatches
   float writeTime;             // time of next output
   float writeStep;             // output time interval

   Timer * update_timer;

protected:
   virtual int setPatchSize(const char * filename);
   int patchSizeFromFile(const char * filename);
   virtual int checkPatchSize(int patchSize, int scalePre, int scalePost, char dim);

   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
#ifdef OBSOLETE
   int initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
#endif
   int initialize_base();
   int initialize(const char * filename);
#ifdef OBSOLETE_STDP
   int initializeSTDP();
#endif
   virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
         const char * filename);
   // PVPatch ** initializeRandomWeights(PVPatch ** patches, int numPatches, int seed);
   PVPatch ** initializeRandomWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeSmartWeights(PVPatch ** patches, int numPatches);
   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeGaussian2DWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeCocircWeights(PVPatch ** patches, int numPatches);
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);
   PVPatch ** createWeights(PVPatch ** patches);
   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch);
   PVPatch ** allocWeights(PVPatch ** patches);

   int uniformWeights(PVPatch * wp, float wMin, float wMax);
   int gaussianWeights(PVPatch * wp, float mean, float stdev);
   // int uniformWeights(PVPatch * wp, float wMin, float wMax, int * seed);
   // int gaussianWeights(PVPatch * wp, float mean, float stdev, int * seed);

   int smartWeights(PVPatch * wp, int k);

   virtual int checkPVPFileHeader(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);
   virtual int checkWeightsHeader(const char * filename, int wgtParams[]);

   virtual int deleteWeights();

   virtual int createAxonalArbors();

   // static member functions

public:
   static PVPatch ** createPatches(int numBundles, int nx, int ny, int nf)
   {
      PVPatch ** patches = (PVPatch**) malloc(numBundles*sizeof(PVPatch*));

      for (int i = 0; i < numBundles; i++) {
         patches[i] = pvpatch_inplace_new(nx, ny, nf);
      }

      return patches;
   }

   static int deletePatches(int numBundles, PVPatch ** patches)
   {
      for (int i = 0; i < numBundles; i++) {
         pvpatch_inplace_delete(patches[i]);
      }
      free(patches);

      return 0;
   }

};

} // namespace PV

#endif /* HYPERCONN_HPP_ */
