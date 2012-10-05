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
#include "../io/BaseConnectionProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../utils/Timer.hpp"
#include "../weightinit/InitWeights.hpp"
#include <stdlib.h>

#ifdef PV_USE_OPENCL
#undef DEBUG_OPENCL  //this is used with some debug code
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
class BaseConnectionProbe;
class PVParams;

/**
 * A HyPerConn identifies a connection between two layers
 */

class HyPerConn {

public:
   HyPerConn();
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             const char * filename);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             const char * filename, InitWeights *weightInit);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             InitWeights *weightInit);
   virtual ~HyPerConn();

   virtual int deliver(Publisher * pub, const PVLayerCube * cube, int neighbor);
   virtual int checkpointRead(const char * cpDir, float* timef);
   virtual int checkpointWrite(const char * cpDir);
   virtual int insertProbe(BaseConnectionProbe* p);
   virtual int outputState(float time, bool last = false);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId = 0);
   virtual int writeWeights(float time, bool last = false);
   virtual int writeWeights(const char* filename);
   virtual int writeWeights(PVPatch*** patches, float** dataStart,
         int numPatches, const char* filename, float timef, bool last);
   virtual int writeTextWeights(const char* filename, int k);

   virtual int writeTextWeightsExtra(FILE* fd, int k, int arborID) {
      return PV_SUCCESS;
   }

   virtual int writePostSynapticWeights(float time, bool last);
   int readWeights(const char* filename);
   bool stochasticReleaseFlag;
   int (*accumulateFunctionPointer)(int nk, float* v, float a, float* w);
   inline bool preSynapticActivityIsNotRate() {return preActivityIsNotRate;}

   // TODO make a get-method to return this.
   virtual PVLayerCube* getPlasticityDecrement() {
      return NULL;
   }

   inline const char* getName() {
      return name;
   }

   inline HyPerCol* getParent() {
      return parent;
   }

   inline HyPerLayer* getPre() {
      return pre;
   }

   inline HyPerLayer* getPost() {
      return post;
   }

   inline ChannelType getChannel() {
      return channel;
   }

   inline InitWeights* getWeightInitializer() {
      return weightInitializer;
   }

   void setDelay(int axonId, float delay);

   inline int getDelay(int arborId = 0) {
      assert(arborId >= 0 && arborId < numAxonalArborLists);
      return delays[arborId];
   }

   inline bool getSelfFlag() {
      return selfFlag;
   }

   ;
   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);

   inline int xPatchSize() {
      return nxp;
   }

   inline int yPatchSize() {
      return nyp;
   }

   inline int fPatchSize() {
      return nfp;
   }

   inline int xPatchStride() {
      return sxp;
   }

   inline int yPatchStride() {
      return syp;
   }

   inline int fPatchStride() {
      return sfp;
   }

   inline int xPostPatchSize() {
      return nxpPost;
   }

   inline int yPostPatchSize() {
      return nypPost;
   }

   inline int fPostPatchSize() {
      return nfpPost;
   }

   //arbor and weight patch related get/set methods:
   inline PVPatch** weights(int arborId = 0) {
      return wPatches[arborId];
   }

   virtual PVPatch* getWeights(int kPre, int arborId);

   // inline PVPatch * getPlasticIncr(int kPre, int arborId) {return plasticityFlag ? dwPatches[arborId][kPre] : NULL;}
   inline float* getPlasticIncr(int kPre, int arborId) {
      return
            plasticityFlag ?
                  &dwDataStart[arborId][kPre * nxp * nyp * nfp
                        + wPatches[arborId][kPre]->offset] :
                  NULL;
   }

   inline const PVPatchStrides* getPostExtStrides() {
      return &postExtStrides;
   }

   inline const PVPatchStrides* getPostNonextStrides() {
      return &postNonextStrides;
   }

   inline float* get_wDataStart(int arborId) {
      return wDataStart[arborId];
   }

   // inline void set_wDataStart(int arborId, pvdata_t * pDataStart) {wDataStart[arborId]=pDataStart;} // Should be protected
   inline float* get_wDataHead(int arborId, int dataIndex) {
      return &wDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   inline float* get_wData(int arborId, int patchIndex) {
      return &wDataStart[arborId][patchToDataLUT(patchIndex) * nxp * nyp * nfp
            + wPatches[arborId][patchIndex]->offset];
   }

   inline float* get_dwDataStart(int arborId) {
      return dwDataStart[arborId];
   }

   // inline void set_dwDataStart(int arborId, pvdata_t * pIncrStart) {dwDataStart[arborId]=pIncrStart;} // Should be protected
   inline float* get_dwDataHead(int arborId, int dataIndex) {
      return &dwDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   inline float* get_dwData(int arborId, int patchIndex) {
      return &dwDataStart[arborId][patchToDataLUT(patchIndex) * nxp * nyp * nfp
            + wPatches[arborId][patchIndex]->offset];
   }

   inline PVPatch* getWPostPatches(int arbor, int patchIndex) {
      return wPostPatches[arbor][patchIndex];
   }

   inline float* getWPostData(int arbor, int patchIndex) {
      return &wPostDataStart[arbor][patchIndex * nxpPost * nypPost * nfpPost]
            + wPostPatches[arbor][patchIndex]->offset;
   }

   inline float* getWPostData(int arbor) {
      return wPostDataStart[arbor];
   }

   int getNumWeightPatches() {
      return numWeightPatches;
   }

   int getNumDataPatches() {
      return numDataPatches;
   }

   inline int numberOfAxonalArborLists() {
      return numAxonalArborLists;
   }

   inline float* getGSynPatchStart(int kPre, int arborId) {
      return gSynPatchStart[arborId][kPre];
   }

   inline size_t getAPostOffset(int kPre, int arborId) {
      return aPostOffset[arborId][kPre];
   }

   HyPerLayer* preSynapticLayer() {
      return pre;
   }

   HyPerLayer* postSynapticLayer() {
      return post;
   }

   int getConnectionId() {
      return connId;
   }

   void setConnectionId(int id) {
      connId = id;
   }

   virtual int setParams(PVParams* params);
   PVPatch*** convertPreSynapticWeights(float time);
   PVPatch**** point2PreSynapticWeights(float time);
   int preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int* kxPre,
         int* kyPre);
   int postSynapticPatchHead(int kPre, int* kxPostOut, int* kyPostOut,
         int* kfPostOut, int* dxOut, int* dyOut, int* nxpOut, int* nypOut);
   virtual int initShrinkPatches();
   virtual int shrinkPatches(int arborId);
   int shrinkPatch(int kExt, int arborId);

   bool getShrinkPatches_flag() {
      return shrinkPatches_flag;
   }

   int sumWeights(int nx, int ny, int offset, float* dataStart, double* sum,
         double* sum2, float* maxVal);
   int scaleWeights(int nx, int ny, int offset, float* dataStart, float sum,
         float sum2, float maxVal);
   virtual int checkNormalizeWeights(float sum, float sigma2, float maxVal);
   virtual int checkNormalizeArbor(PVPatch** patches, float** dataStart,
         int numPatches, int arborId);
   virtual int normalizeWeights(PVPatch** patches, float** dataStart,
         int numPatches, int arborId);
   // patchIndexToKernelIndex() is deprecated.  Use patchIndexToDataIndex() or dataIndexToUnitCellIndex() instead
   /*
    virtual int patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex = NULL,
    int * kyKernelIndex = NULL, int * kfKernelIndex = NULL);
    */
   virtual int patchToDataLUT(int patchIndex);
   virtual int patchIndexToDataIndex(int patchIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   static int decodeChannel(int channel_code, ChannelType * channel_type);

#ifdef USE_SHMGET
    virtual bool getShmgetOwner(){
      return true;
   };
    virtual bool getShmgetFlag(){
      return false;
   };
#endif

protected:
   HyPerLayer* pre;
   HyPerLayer* post;
   HyPerCol* parent;
   int numWeightPatches; // Number of PVPatch structures in buffer pointed to by wPatches[arbor]
   int numDataPatches; // Number of blocks of pvdata_t's in buffer pointed to by wDataStart[arbor]

   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
#ifdef USE_SHMGET
   bool shmget_owner;
   int *shmget_id;
   bool shmget_flag;
#endif
private:
   PVPatch*** wPatches; // list of weight patches, one set per arbor
   float*** gSynPatchStart; //  gSynPatchStart[arborId][kExt] is a pointer to the start of the patch in the post-synaptic GSyn buffer
   float** gSynPatchStartBuffer;
   size_t** aPostOffset; // aPostOffset[arborId][kExt] is the index of the start of a patch into an extended post-synaptic layer
   size_t* aPostOffsetBuffer;
   int* delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th arbor
   PVPatchStrides postExtStrides; // sx,sy,sf for a patch mapping into an extended post-synaptic layer
   PVPatchStrides postNonextStrides; // sx,sy,sf for a patch mapping into a non-extended post-synaptic layer
   float** wDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   float** dwDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   bool combine_dW_with_W_flag; // indicates that dwDataStart should be set equal to wDataStart, useful for saving memory when weights are not being learned but not used
   int defaultDelay; //added to save params file defined delay...

protected:
   char* name;
   int nxp, nyp, nfp; // size of weight dimensions
   int sxp, syp, sfp; // stride in x,y,features
   ChannelType channel; // which channel of the post to update (e.g. inhibit)
   int connId; // connection id
   // PVPatch       *** dwPatches;      // list of weight patches for storing changes to weights
   int numAxonalArborLists; // number of axonal arbors (weight patches) for presynaptic layer
   PVPatch*** wPostPatches; // post-synaptic linkage of weights // This is being deprecated in favor of TransposeConn
   float** wPostDataStart;

   PVPatch**** wPostPatchesp; // Pointer to wPatches, but from the postsynaptic perspective
   float*** wPostDataStartp; // Pointer to wDataStart, but from the postsynaptic perspective

   int nxpPost, nypPost, nfpPost;
   int numParams;
   //PVConnParams * params;
   float wMax;
   float wMin;
   int numProbes;
   BaseConnectionProbe** probes; // probes used to output data
   bool ioAppend; // controls opening of binary files
   float wPostTime; // time of last conversion to wPostPatches
   float writeTime; // time of next output, initialized in params file parameter initialWriteTime
   float writeStep; // output time interval
   bool writeCompressedWeights; // true=write weights with 8-bit precision;
   // false=write weights with float precision
   int fileType; // type ID for file written by PV::writeWeights
   Timer* update_timer;
   bool plasticityFlag;
   bool selfFlag; // indicates that connection is from a layer to itself (even though pre and post may be separately instantiated)
   bool normalize_flag;
   float normalize_strength;
   bool normalizeArborsIndividually; // if true, each arbor is normalized individually, otherwise, arbors normalized together
   bool normalize_max;
   bool normalize_zero_offset;
   float normalize_cutoff;
   bool shrinkPatches_flag;
   //This object handles calculating weights.  All the initialize weights methods for all connection classes
   //are being moved into subclasses of this object.  The default root InitWeights class will create
   //2D Gaussian weights.  If weight initialization type isn't created in a way supported by Buildandrun,
   //this class will try to read the weights from a file or will do a 2D Gaussian.
   InitWeights* weightInitializer;
   bool preActivityIsNotRate; // TODO Rename this member variable
   bool normalizeTotalToPost; // if false, normalize the sum of weights from each presynaptic neuron.  If true, normalize the sum of weights into a postsynaptic neuron.

protected:
   virtual int initNumWeightPatches();
   virtual int initNumDataPatches();

   inline PVPatch*** get_wPatches() {
      return wPatches;
   } // protected so derived classes can use; public methods are weights(arbor) and getWeights(patchindex,arbor)

   inline void set_wPatches(PVPatch*** patches) {
      wPatches = patches;
   }

   inline float*** getGSynPatchStart() {
      return gSynPatchStart;
   }

   inline void setGSynPatchStart(float*** patchstart) {
      gSynPatchStart = patchstart;
   }

   inline size_t** getAPostOffset() {
      return aPostOffset;
   }

   inline void setAPostOffset(size_t** postoffset) {
      aPostOffset = postoffset;
   }

   inline float** get_wDataStart() {
      return wDataStart;
   }

   inline void set_wDataStart(float** datastart) {
      wDataStart = datastart;
   }

   inline void set_wDataStart(int arborId, float* pDataStart) {
      wDataStart[arborId] = pDataStart;
   }

   inline float** get_dwDataStart() {
      return dwDataStart;
   }

   inline void set_dwDataStart(float** datastart) {
      dwDataStart = datastart;
   }

   inline void set_dwDataStart(int arborId, float* pIncrStart) {
      dwDataStart[arborId] = pIncrStart;
   }

   inline int* getDelays() {
      return delays;
   }

   inline void setDelays(int* delayptr) {
      delays = delayptr;
   }

   virtual ChannelType readChannelCode(PVParams * params);

   int calcUnitCellIndex(int patchIndex, int* kxUnitCellIndex = NULL,
         int* kyUnitCellIndex = NULL, int* kfUnitCellIndex = NULL);
   virtual int setPatchSize(const char* filename);
   virtual int setPatchStrides();
   virtual int checkPatchSize(int patchSize, int scalePre, int scalePost,
         char dim);
   int calcPatchSize(int n, int kex, int* kl, int* offset, int* nxPatch,
         int* nyPatch, int* dx, int* dy);
   int patchSizeFromFile(const char* filename);
   int initialize_base();
   virtual int createArbors();
   void createArborsOutOfMemory();
   virtual int constructWeights(const char* filename);
   int initialize(const char* name, HyPerCol* hc, HyPerLayer* pre,
         HyPerLayer* post, const char* filename,
         InitWeights* weightInit = NULL);
   virtual int initPlasticityPatches();
   virtual PVPatch*** initializeWeights(PVPatch*** arbors, float** dataStart,
         int numPatches, const char* filename);
   virtual InitWeights* getDefaultInitWeightsMethod(const char* keyword);
   virtual InitWeights* handleMissingInitWeights(PVParams* params);
   virtual float* createWeights(PVPatch*** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   float* createWeights(PVPatch*** patches, int axonId);
   virtual float* allocWeights(PVPatch*** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   //PVPatch ** allocWeights(PVPatch ** patches);
   int clearWeights(float** dataStart, int numPatches, int nx, int ny, int nf);
   virtual int initNormalize();
   virtual int checkPVPFileHeader(Communicator* comm, const PVLayerLoc* loc,
         int params[], int numParams);
   virtual int checkWeightsHeader(const char* filename, int wgtParams[]);
   // virtual int deleteWeights(); // Changed to a private method.  Should not be virtual since it's called from the destructor.
   virtual int adjustAxonalArbors(int arborId);
   int checkpointFilename(char * cpFilename, int size, const char * cpDir);
   int writeScalarFloat(const char * cp_dir, const char * val_name, float val);

   virtual int calc_dW(int axonId = 0);
   void connOutOfMemory(const char* funcname);

   //this method setups up GPU stuff...
   // CL kernel for layer recvSynapticInput call
   //number of receive synaptic runs to wait for (=numarbors)
   //cl_event   evCopyDataStore;
   // OpenCL buffers
   // ids of OpenCL arguments that change
   //

private:
   int clearWeights(float* arborDataStart, int numPatches, int nx, int ny,
         int nf);
   int deleteWeights();

public:
   // static member functions
   //
   static PVPatch** createPatches(int nPatches, int nx, int ny) {
      PVPatch** patchpointers = (PVPatch**) (calloc(nPatches, sizeof(PVPatch*)));
      PVPatch* patcharray = (PVPatch*) (calloc(nPatches, sizeof(PVPatch)));

      PVPatch * curpatch = patcharray;
      for (int i = 0; i < nPatches; i++) {
         pvpatch_init(curpatch, nx, ny);
         patchpointers[i] = curpatch;
         curpatch++;
      }

      return patchpointers;
   }

   static int deletePatches(PVPatch ** patchpointers)
   {
      if (patchpointers != NULL && *patchpointers != NULL){
         free(*patchpointers);
         *patchpointers = NULL;
      }
      free(patchpointers);
      patchpointers = NULL;
//      for (int i = 0; i < numBundles; i++) {
//         pvpatch_inplace_delete(patches[i]);
//      }
      //free(patches);

      return 0;
   }

};

} // namespace PV

#endif /* HYPERCONN_HPP_ */
