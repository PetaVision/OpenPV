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
class PVParams;

/**
 * A HyPerConn identifies a connection between two layers
 */

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
#ifdef PV_USE_OPENCL
   virtual int deliverOpenCL(Publisher * pub);
#endif

   virtual int insertProbe(ConnectionProbe * p);
   virtual int outputState(float time, bool last=false);
   virtual int updateState(float time, float dt);
   virtual int calc_dW(int axonId = 0);
   virtual int updateWeights(int axonId = 0);

   virtual int writeWeights(float time, bool last=false);
   virtual int writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last, int arborId);
   virtual int writeTextWeights(const char * filename, int k);
   virtual int writeTextWeightsExtra(FILE * fd, int k, int arborID)
                                                    {return PV_SUCCESS;}

   virtual int writePostSynapticWeights(float time, bool last);
   virtual int writePostSynapticWeights(float time, bool last, int axonID);

   int readWeights(const char * filename);

   virtual int correctPIndex(int patchIndex);

   bool stochasticReleaseFlag;
   int (*accumulateFunctionPointer)(int nk, float* RESTRICT v, float a, float* RESTRICT w);
   // TODO make a get-method to return this.

   virtual PVLayerCube * getPlasticityDecrement()    {return NULL;}


   inline const char * getName()                     {return name;}
   inline HyPerCol * getParent()                     {return parent;}
   inline HyPerLayer * getPre()                      {return pre;}
   inline HyPerLayer * getPost()                     {return post;}
   inline ChannelType getChannel()                   {return channel;}
   inline InitWeights * getWeightInitializer()       {return weightInitializer;}
   void setDelay(int axonId, int delay);
   inline int getDelay(int arborId = 0)               {assert(arborId>=0 && arborId<numAxonalArborLists); return delays[arborId];}

   virtual float minWeight(int arborId = 0)          {return 0.0;}
   virtual float maxWeight(int arborId = 0)          {return wMax;}

   inline int xPatchSize()                           {return nxp;}
   inline int yPatchSize()                           {return nyp;}
   inline int fPatchSize()                           {return nfp;}

   //arbor and weight patch related get/set methods:
   inline PVPatch ** weights(int arborId = 0)        {return wPatches[arborId];}
   virtual PVPatch * getWeights(int kPre, int arborId);
   inline PVAxonalArbor * axonalArbor(int kPre, int arborId)
                                                     {return &axonalArborList[arborId][kPre];}
   virtual int numWeightPatches();
   virtual int numDataPatches();
   inline  int numberOfAxonalArborLists()            {return numAxonalArborLists;}

   inline pvdata_t * get_dWData(int kPre, int arborId) {PVPatch * dW = axonalArbor(kPre,arborId)->plasticIncr; assert(dW); return dW->data;}
   inline size_t getGSynOffset(int kPre, int arborId) {return gSynOffset[arborId][kPre];} // {return axonalArbor(kPre,arborId)->offset;}
   int getAPostOffset(int kPre, int arborId);

   HyPerLayer * preSynapticLayer()                   {return pre;}
   HyPerLayer * postSynapticLayer()                  {return post;}

   int  getConnectionId()                            {return connId;}
   void setConnectionId(int id)                      {connId = id;}

   virtual int setParams(PVParams * params /*, PVConnParams * p*/);

   PVPatch *** convertPreSynapticWeights(float time);

   int preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int * kxPre, int * kyPre);
   int postSynapticPatchHead(int kPre,
                             int * kxPostOut, int * kyPostOut, int * kfPostOut,
                             int * dxOut, int * dyOut, int * nxpOut, int * nypOut);



   virtual int initShrinkPatches();

   virtual int shrinkPatches(int arborId);
   int shrinkPatch(int kExt, int arborId);
   bool getShrinkPatches_flag() {return shrinkPatches_flag;}

   virtual int initNormalize();
   int sumWeights(PVPatch * wp, double * sum, double * sum2, pvdata_t * maxVal);
   int scaleWeights(PVPatch * wp, pvdata_t sum, pvdata_t sum2, pvdata_t maxVal);
   virtual int checkNormalizeWeights(PVPatch * wp, float sum, float sigma2, float maxVal);
   virtual int checkNormalizeArbor(PVPatch ** patches, int numPatches, int arborId);
   virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

   virtual int kernelIndexToPatchIndex(int kernelIndex, int * kxPatchIndex = NULL,
         int * kyPatchIndex = NULL, int * kfPatchIndex = NULL);

   virtual int patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex = NULL,
         int * kyKernelIndex = NULL, int * kfKernelIndex = NULL);

protected:
   HyPerLayer     * pre;
   HyPerLayer     * post;
   HyPerCol       * parent;
   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
   //PVAxonalArbor  * axonalArborList[MAX_ARBOR_LIST]; // list of axonal arbors for each neighbor
private:
   PVPatch       *** wPatches; // list of weight patches, one set per arbor
   PVAxonalArbor ** axonalArborList; // list of axonal arbors for each presynaptic cell in extended layer
   size_t        ** gSynOffset; // gSynOffset[arborId][kExt] is the index of the start of a patch into a non-extended postsynaptic layer
   int           *  delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th arbor
protected:
   PVPatch       *** wPostPatches;  // post-synaptic linkage of weights
   PVPatch       *** pIncr;      // list of weight patches for storing changes to weights
   int numAxonalArborLists;  // number of axonal arbors (weight patches) for presynaptic layer

   ChannelType channel;    // which channel of the post to update (e.g. inhibit)
   int connId;             // connection id

   char * name;
   int nxp, nyp, nfp;      // size of weight dimensions

   int numParams;
   //PVConnParams * params;

   float wMax;
   float wMin;

   int numProbes;
   ConnectionProbe ** probes; // probes used to output data
   bool ioAppend;               // controls opening of binary files
   float wPostTime;             // time of last conversion to wPostPatches
   float writeTime;             // time of next output
   float writeStep;             // output time interval

   bool writeCompressedWeights; // true=write weights with 8-bit precision;
                                // false=write weights with float precision

   int fileType;                // type ID for file written by PV::writeWeights

   Timer * update_timer;

   bool plasticityFlag;

   bool normalize_flag;
   float normalize_strength;
   bool normalize_arbors_individually;  // if true, each arbor is normalized individually, otherwise, arbors normalized together
   bool normalize_max;
   bool normalize_zero_offset;
   float normalize_cutoff;
   bool shrinkPatches_flag;

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

   int initialize_base();
   virtual int createArbors();
   int constructWeights(const char * filename);
#ifdef OBSOLETE // Marked obsolete Oct 1, 2011.  Made redundant by adding default value to weightInit argument of other initialize method
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
         HyPerLayer * post, ChannelType channel, const char * filename);
#endif // OBSOLETE
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  ChannelType channel, const char * filename,
                  InitWeights *weightInit=NULL);
   virtual int initPlasticityPatches();
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, int numPatches,
         const char * filename);
   virtual InitWeights * handleMissingInitWeights(PVParams * params);
   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   PVPatch ** createWeights(PVPatch ** patches, int axonId);
   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   //PVPatch ** allocWeights(PVPatch ** patches);

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
   cl_event   evRecvSyn;

   // OpenCL buffers
   //
   CLBuffer *  clGSyn;
   CLBuffer *  clActivity;
   CLBuffer ** clWeights;

   // ids of OpenCL arguments that change
   //
   int clArgIdOffset;
   int clArgIdWeights;

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
