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
#include "InitWeights.hpp"
#include <stdlib.h>

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLKernel.hpp"
#include "../arch/opencl/CLBuffer.hpp"
#endif

#define PROTECTED_NUMBER 13
#define MAX_ARBOR_LIST (1+MAX_NEIGHBORS)

namespace PV {

class HyPerCol;
class HyPerLayer;
class InitWeights;
class InitUniformRandomWeights;
class InitGaussianRandomWeights;
class InitSmartWeights;
class InitCocircWeights;
class ConnectionProbe;

/**
 * A PVConnection identifies a connection between two layers
 */
//typedef struct {
//   int delay; // current output delay in the associated f ring buffer (should equal fixed delay + varible delay for valid connection)
//// Commenting out unused parameters.  Is PVConnParams still necessary?
////   int fixDelay; // fixed output delay. TODO: should be float
////   int varDelayMin; // minimum variable conduction delay
////   int varDelayMax; // maximum variable conduction delay
////   int numDelay;
////   int isGraded; //==1, release is stochastic with prob = (activity <= 1), default is 0 (no graded release)
////   float vel;  // conduction velocity in position units (pixels) per time step--added by GTK
////   float rmin; // minimum connection distance
////   float rmax; // maximum connection distance
//} PVConnParams;

class HyPerConn {

public:
   HyPerConn();
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, const char * filename);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, const char * filename, InitWeights *weightInit);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, InitWeights *weightInit);
   virtual ~HyPerConn();

   virtual int deliver(Publisher * pub, const PVLayerCube * cube, int neighbor);

   virtual int insertProbe(ConnectionProbe * p);
   virtual int outputState(float time, bool last=false);
   virtual int updateState(float time, float dt);
   virtual int calc_dW(int axonId);
   virtual int updateWeights(int axonId);

   virtual int writeWeights(float time, bool last=false);
   virtual int writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last, int arborId);
   virtual int writeTextWeights(const char * filename, int k);
   virtual int writeTextWeightsExtra(FILE * fd, int k, int arborID)  {return 0;}

   virtual int writePostSynapticWeights(float time, bool last=false);
   virtual int writePostSynapticWeights(float time, bool last, int axonID);

   int readWeights(const char * filename);

   virtual int correctPIndex(int patchIndex);

#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
                                  const char * filename);
#endif

   virtual PVLayerCube * getPlasticityDecrement()               {return NULL;}


   inline const char * getName()                     {return name;}
   inline HyPerCol * getParent()                     {return parent;}
   inline HyPerLayer * getPre()                        {return pre;}
   inline HyPerLayer * getPost()                       {return post;}
   inline ChannelType getChannel()                 {return channel;}
   inline InitWeights * getWeightInitializer()    {return weightInitializer;}
   //inline int          getDelay()                    {return params->delay;}
   inline int getDelay(int axonId)     {assert(axonId<numAxonalArborLists); return axonalArbor(axonId,0)->delay;}

   virtual float minWeight()                         {return 0.0;}
   virtual float maxWeight()                         {return wMax;}

   inline int xPatchSize()                           {return nxp;}
   inline int yPatchSize()                           {return nyp;}
   inline int fPatchSize()                           {return nfp;}

   //arbor and weight patch related get/set methods:
   inline PVPatch ** weights(int arborId)           {return wPatches[arborId];}
   virtual PVPatch * getWeights(int kPre, int arbor);
   inline PVAxonalArbor * axonalArbor(int kPre, int arborId)
      {return &axonalArborList[arborId][kPre];}
   virtual int numWeightPatches();
   virtual int numDataPatches();
   inline  int numberOfAxonalArborLists()            {return numAxonalArborLists;}

   HyPerLayer * preSynapticLayer()     {return pre;}
   HyPerLayer * postSynapticLayer()    {return post;}

   int  getConnectionId()              {return connId;}
   void setConnectionId(int id)        {connId = id;}

   int setParams(PVParams * params /*, PVConnParams * p*/);

   PVPatch *** convertPreSynapticWeights(float time);

   int preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int * kxPre, int * kyPre);
   int postSynapticPatchHead(int kPre,
                             int * kxPostOut, int * kyPostOut, int * kfPostOut,
                             int * dxOut, int * dyOut, int * nxpOut, int * nypOut);



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
   virtual int initNormalize();
   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

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
   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
   //PVAxonalArbor  * axonalArborList[MAX_ARBOR_LIST]; // list of axonal arbors for each neighbor
private:
   PVPatch       *** wPatches; // list of weight patches, one set per arbor
   PVAxonalArbor ** axonalArborList; // list of axonal arbors for each presynaptic cell in extended layer
   int numAxonalArborLists;  // number of axonal arbors (weight patches) for presynaptic layer
protected:
   PVPatch       *** wPostPatches;  // post-synaptic linkage of weights
   PVPatch       *** pIncr;      // list of weight patches for storing changes to weights

#ifdef OBSOLETE_STDP
   bool     localWmaxFlag;  // presence of rate dependent wMax;
   pvdata_t * Wmax;   // adaptive upper STDP weight boundary
#endif

   ChannelType channel;    // which channel of the post to update (e.g. inhibit)
   int connId;             // connection id

   char * name;
   int nxp, nyp, nfp;      // size of weight dimensions

   int numParams;
   //PVConnParams * params;


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

   bool writeCompressedWeights; // true=write weights with 8-bit precision;
                                // false=write weights with float precision

   Timer * update_timer;

   bool plasticityFlag;

   bool normalize_flag;
   float normalize_strength;
   float normalize_max;
   float normalize_zero_offset;
   float normalize_cutoff;

   //This object handles calculating weights.  All the initialize weights methods for all connection classes
   //are being moved into subclasses of this object.  The default root InitWeights class will create
   //2D Gaussian weights.  If weight initialization type isn't created in a way supported by Buildandrun,
   //this class will try to read the weights from a file or will do a 2D Gaussian.
   InitWeights *weightInitializer;

protected:
   virtual int setPatchSize(const char * filename);
   virtual int checkPatchSize(int patchSize, int scalePre, int scalePost, char dim);
   int calcPatchSize(int n, int kex,
                     int * kl, int * offset,
                     int * nxPatch, int * nyPatch,
                     int * dx, int * dy);

   int patchSizeFromFile(const char * filename);

   virtual int initialize_base();
   virtual int createArbors();
   int constructWeights(const char * filename);
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
         HyPerLayer * post, ChannelType channel, const char * filename);
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  ChannelType channel, const char * filename,
                  InitWeights *weightInit);
   virtual int initPlasticityPatches();
#ifdef OBSOLETE_STDP
   int initializeSTDP();
#endif
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, int numPatches,
         const char * filename);
#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
   // PVPatch ** initializeRandomWeights(PVPatch ** patches, int numPatches, int seed);
   PVPatch ** initializeRandomWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeSmartWeights(PVPatch ** patches, int numPatches);
   virtual PVPatch ** initializeDefaultWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeGaussian2DWeights(PVPatch ** patches, int numPatches);
   PVPatch ** initializeCocircWeights(PVPatch ** patches, int numPatches);
#endif
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   PVPatch ** createWeights(PVPatch ** patches, int axonId);
   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   //PVPatch ** allocWeights(PVPatch ** patches);

#ifdef OBSOLETE //The following methods have been added to the new InitWeights classes.  Please
                //use the param "weightInitType" to choose an initialization type
   int uniformWeights(PVPatch * wp, float minwgt, float maxwgt);
   int gaussianWeights(PVPatch * wp, float mean, float stdev);

   int smartWeights(PVPatch * wp, int k);
#endif
   virtual int checkPVPFileHeader(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);
   virtual int checkWeightsHeader(const char * filename, int wgtParams[]);

   virtual int deleteWeights();

   virtual int createAxonalArbors(int arborId);

   // following is overridden by KernelConn to set kernelPatches
   //inline void setWPatches(PVPatch ** patches, int arborId) {wPatches[arborId]=patches;}
   virtual int setWPatches(PVPatch ** patches, int arborId) {wPatches[arborId]=patches; return 0;}
   virtual int setdWPatches(PVPatch ** patches, int arborId) {pIncr[arborId]=patches; return 0;}
   inline void setArbor(PVAxonalArbor* arbor, int arborId) {axonalArborList[arborId]=arbor;}

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);

   CLKernel * krRecvSyn;        // CL kernel for layer recvSynapticInput call

   // OpenCL buffers
   //
   CLBuffer * clGSyn;
   CLBuffer * clActivity;
   CLBuffer * clWeights;
#endif

public:

   // static member functions
   //

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
