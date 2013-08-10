/*
 * HyPerConnection.hpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "../columns/InterColComm.hpp"
#include "../columns/HyPerCol.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../io/BaseConnectionProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../utils/Timer.hpp"
#include "../weightinit/InitWeights.hpp"
#include <stdlib.h>

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLKernel.hpp"
#include "../arch/opencl/CLBuffer.hpp"
#endif

#define PROTECTED_NUMBER 13
#define MAX_ARBOR_LIST (1+MAX_NEIGHBORS)

namespace PV {


//class HyPerCol;
//class HyPerLayer;
class InitWeights;
class BaseConnectionProbe;
class PVParams;
class NormalizeBase;

/**
 * A HyPerConn identifies a connection between two layers
 */

class HyPerConn {

public:
	friend class CloneKernelConn;
   HyPerConn();
   HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name);
   HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name,
             const char * filename);
   HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name,
             const char * filename, InitWeights *weightInit);
   HyPerConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name,
             InitWeights *weightInit);
   virtual ~HyPerConn();
#ifdef PV_USE_OPENCL
   virtual int deliverOpenCL(Publisher * pub, const PVLayerCube * cube);
#endif

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

#ifdef OBSOLETE // Marked obsolete July 25, 2013.  recvSynapticInput is now called by recvAllSynapticInput, called by HyPerCol, so deliver andtriggerReceive aren't needed.
   virtual int deliver(Publisher * pub, const PVLayerCube * cube, int neighbor);
#endif // OBSOLETE
   virtual int checkpointRead(const char * cpDir, double* timef);
   virtual int checkpointWrite(const char * cpDir);
   virtual int insertProbe(BaseConnectionProbe* p);
   virtual int outputState(double time, bool last = false);
   virtual int updateState(double time, double dt);
   virtual int updateWeights(int arborId = 0);
   virtual int writeWeights(double time, bool last = false);
   virtual int writeWeights(const char* filename);
   virtual int writeWeights(PVPatch*** patches, float** dataStart,
         int numPatches, const char* filename, double timef, bool compressWeights, bool last);
   virtual int writeTextWeights(const char* filename, int k);

   virtual int writeTextWeightsExtra(PV_Stream * pvstream, int k, int arborID) {
      return PV_SUCCESS;
   }

   virtual int writePostSynapticWeights(double time, bool last);
   int readWeights(const char* filename);
    
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

   inline const char * getPreLayerName() {
      return preLayerName;
   }

   inline HyPerLayer* getPre() {
      return pre;
   }

   inline const char * getPostLayerName() {
      return postLayerName;
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

   void setDelay(int arborId, float delay);

   inline int getDelay(int arborId = 0) {
      assert(arborId >= 0 && arborId < numAxonalArborLists);
      return delays[arborId];
   }

   inline bool getSelfFlag() {
      return selfFlag;
   }

   inline bool getPlasticityFlag() {
      return plasticityFlag;
   };

   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);

   inline int getNxpShrunken() {
	   return nxpShrunken;
   }

   inline int getNypShrunken() {
	   return nypShrunken;
   }

   inline int getOffsetShrunken() {
	   return offsetShrunken;
   }

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

//   inline float* getGSynPatchStart(int kPre, int arborId) {
//      return gSynPatchStart[arborId][kPre];
//   }

   inline size_t getGSynPatchStart(int kPre, int arborId) {
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

   NormalizeBase * getNormalizer() {
      return normalizer;
   }

   PVPatch*** convertPreSynapticWeights(double time);
   PVPatch**** point2PreSynapticWeights();
   //PVPatch**** point2PreSynapticWeights2();
   int preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int* kxPre,
         int* kyPre);
   int postSynapticPatchHead(int kPre, int* kxPostOut, int* kyPostOut,
         int* kfPostOut, int* dxOut, int* dyOut, int* nxpOut, int* nypOut);
   virtual int shrinkPatches(int arborId);
   int shrinkPatch(int kExt, int arborId);

   bool getShrinkPatches_flag() {
      return shrinkPatches_flag;
   }

   bool getUseWindowPost(){return useWindowPost;};
   bool getUpdateGSynFromPostPerspective(){return updateGSynFromPostPerspective;};

   int sumWeights(int nx, int ny, int offset, float* dataStart, double* sum,
         double* sum2, float* maxVal);
   int scaleWeights(int nx, int ny, int offset, float* dataStart, float sum,
         float sum2, float maxVal);
   virtual int checkNormalizeWeights(float sum, float sum2, float sigma2, float maxVal);
   virtual int checkNormalizeArbor(PVPatch** patches, float** dataStart,
         int numPatches, int arborId);
   virtual int normalizeWeights();
//   virtual int normalizeWeights(PVPatch** patches, float** dataStart,
//         int numPatches, int arborId);
//


#ifdef PV_USE_OPENCL
   virtual int * getLUTpointer() {return NULL;}
#endif // PV_USE_OPENCL
   virtual void initPatchToDataLUT(){};
   virtual int patchToDataLUT(int patchIndex);
   virtual int patchIndexToDataIndex(int patchIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   static int decodeChannel(int channel_code, ChannelType * channel_type);
   static int getPreAndPostLayerNames(const char * name, PVParams * params, char ** preLayerNamePtr, char ** postLayerNamePtr);

#ifdef USE_SHMGET
   virtual bool getShmgetFlag(){
     return shmget_flag;
  };
   virtual bool getShmgetOwner(int arbor_ID = 0){
     return (shmget_owner == NULL) ? false : shmget_owner[arbor_ID];
  };
    virtual bool * getShmgetOwnerHead(){
    	return  shmget_owner;
    }
#endif

protected:
   char * preLayerName;
   char * postLayerName;
   HyPerLayer* pre;
   HyPerLayer* post;
   HyPerCol* parent;
   char * filename; // Filename if loading weights from a file
   int fileparams[NUM_WGT_PARAMS]; // The header of the file named by the filename member variable
   int numWeightPatches; // Number of PVPatch structures in buffer pointed to by wPatches[arbor]
   int numDataPatches; // Number of blocks of pvdata_t's in buffer pointed to by wDataStart[arbor]

   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
#ifdef USE_SHMGET
   bool *shmget_owner;
   int *shmget_id;
   bool shmget_flag;
#endif
private:
   PVPatch*** wPatches; // list of weight patches, one set per arbor
   // GTK:: gSynPatchStart redefined as offset from start of associated gSynBuffer
   //float*** gSynPatchStart; //  gSynPatchStart[arborId][kExt] is a pointer to the start of the patch in the post-synaptic GSyn buffer
   size_t** gSynPatchStart;  // gSynPatchStart[arborId][kExt] is the offset to the start of the patch from the beginning of the post-synaptic GSyn buffer for corresponding channel
   //float** gSynPatchStartBuffer;
   size_t * gSynPatchStartBuffer;
   size_t** aPostOffset; // aPostOffset[arborId][kExt] is the index of the start of a patch into an extended post-synaptic layer
   size_t* aPostOffsetBuffer;
   int* delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th arbor
   PVPatchStrides postExtStrides; // sx,sy,sf for a patch mapping into an extended post-synaptic layer
   PVPatchStrides postNonextStrides; // sx,sy,sf for a patch mapping into a non-extended post-synaptic layer
   float** wDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   float** dwDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   int defaultDelay; //added to save params file defined delay...
   const float* fDelayArray;
   int delayArraySize;

protected:
   bool useWindowPost;
   char* name;
   int nxp, nyp, nfp; // size of weight dimensions
   bool warnDefaultNfp; // Whether to print a warning if the default nfp is used.
   int nxpShrunken, nypShrunken, offsetShrunken; // if user requires a smaller patch than is required by PetaVision
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
   double wPostTime; // time of last conversion to wPostPatches
   double writeTime; // time of next output, initialized in params file parameter initialWriteTime
   double writeStep; // output time interval
   bool writeCompressedWeights; // if true, outputState writes weights with 8-bit precision; if false, write weights with float precision
   bool writeCompressedCheckpoints; // similar to writeCompressedWeights, but for checkpointWrite instead of outputState
   int fileType; // type ID for file written by PV::writeWeights
   Timer* update_timer;
   bool plasticityFlag;
   bool combine_dW_with_W_flag; // indicates that dwDataStart should be set equal to wDataStart, useful for saving memory when weights are not being learned but not used
   bool selfFlag; // indicates that connection is from a layer to itself (even though pre and post may be separately instantiated)
   const char * normalizeMethod;
   NormalizeBase * normalizer;
   // bool normalize_flag; // replaced by testing whether normalizer!=NULL
   float normalize_strength;
   bool normalizeArborsIndividually; // if true, each arbor is normalized individually, otherwise, arbors normalized together
   bool normalize_max;
   bool normalize_zero_offset;
   bool normalize_RMS_amp;
   float normalize_cutoff;
   bool shrinkPatches_flag;
   //This object handles calculating weights.  All the initialize weights methods for all connection classes
   //are being moved into subclasses of this object.  The default root InitWeights class will create
   //2D Gaussian weights.  If weight initialization type isn't created in a way supported by Buildandrun,
   //this class will try to read the weights from a file or will do a 2D Gaussian.
   InitWeights* weightInitializer;
   const char* pvpatchAccumulateType;
   bool preActivityIsNotRate; // TODO Rename this member variable
   bool normalizeTotalToPost; // if false, normalize the sum of weights from each presynaptic neuron.  If true, normalize the sum of weights into a postsynaptic neuron.
   float dWMax;  // dW scale factor
   bool useListOfArborFiles;
   bool combineWeightFiles;
   bool updateGSynFromPostPerspective;

   int neededRNGSeeds;  // The number of independent random number generators used by the layer, summed over all MPI processes.
   unsigned long rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use by this layer

#ifdef PV_USE_OPENCL
   bool gpuAccelerateFlag; // Whether to accelerate the connection on a GPU
   bool ignoreGPUflag;     // Don't use GPU (overrides gpuAccelerateFlag)
#endif // PV_USE_OPENCL
protected:
   virtual int initNumWeightPatches();
   virtual int initNumDataPatches();

   inline PVPatch*** get_wPatches() {
      return wPatches;
   } // protected so derived classes can use; public methods are weights(arbor) and getWeights(patchindex,arbor)

   inline void set_wPatches(PVPatch*** patches) {
      wPatches = patches;
   }

//   inline float*** getGSynPatchStart() {
//      return gSynPatchStart;
//   }

   inline size_t** getGSynPatchStart() {
      return gSynPatchStart;
   }

//   inline void setGSynPatchStart(float*** patchstart) {
//      gSynPatchStart = patchstart;
//   }

   inline void setGSynPatchStart(size_t** patchstart) {
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

   int calcUnitCellIndex(int patchIndex, int* kxUnitCellIndex = NULL,
         int* kyUnitCellIndex = NULL, int* kfUnitCellIndex = NULL);
   // virtual int setPatchSize();
   virtual int setPatchStrides();
   int checkPatchDimensions();
   virtual int checkPatchSize(int patchSize, int scalePre, int scalePost,
         char dim);
   int calcPatchSize(int n, int kex, int* kl, int* offset, int* nxPatch,
         int* nyPatch, int* dx, int* dy);
   int patchSizeFromFile(const char* filename);
   int initialize_base();
   virtual int createArbors();
   void createArborsOutOfMemory();
   int initializeDelays(const float * fDelayArray, int size);
   virtual int constructWeights(const char* filename);
   int initialize(const char* name, HyPerCol* hc, const char * pre_layer_name,
         const char * post_layer_name, const char* filename,
         InitWeights* weightInit = NULL);
   int setParent(HyPerCol * hc);
   int setName(const char * name);
   int setPreLayerName(const char * pre_name);
   int setPostLayerName(const char * post_name);
   int setFilename(const char * filename);
   int setWeightInitializer(InitWeights * weightInit);
   virtual int initPlasticityPatches();
   virtual int setPatchSize(); // Sets nxp, nyp, nfp if weights are loaded from file.  Subclasses override if they have specialized ways of setting patch size that needs to go in the communicate stage.
                               // (e.g. BIDSConn uses pre and post layer size to set nxp,nyp, but pre and post aren't set until communicateInitInfo().
   virtual PVPatch*** initializeWeights(PVPatch*** arbors, float** dataStart,
         int numPatches, const char* filename);
   virtual InitWeights* getDefaultInitWeightsMethod(const char* keyword);
   virtual InitWeights* handleMissingInitWeights(PVParams* params);
   virtual int createWeights(PVPatch*** patches, int nWeightPatches, int nDataPatches, int nxPatch,
         int nyPatch, int nfPatch, int arborId);
   int createWeights(PVPatch*** patches, int arborId);
   virtual pvdata_t * allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch);
   //virtual pvdata_t * allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
   //      int nyPatch, int nfPatch, int arborId);
   //PVPatch ** allocWeights(PVPatch ** patches);
   int clearWeights(float** dataStart, int numPatches, int nx, int ny, int nf);
   virtual int initNormalize();
   virtual int checkPVPFileHeader(Communicator* comm, const PVLayerLoc* loc,
         int params[], int numParams);
   virtual int checkWeightsHeader(const char* filename, const int wgtParams[]);
   // virtual int deleteWeights(); // Changed to a private method.  Should not be virtual since it's called from the destructor.
   virtual int adjustAxonalArbors(int arborId);
   int checkpointFilename(char * cpFilename, int size, const char * cpDir);
   // int writeScalarFloat(const char * cp_dir, const char * val_name, double val); // Replaced by HyPerCol::writeScalarToFile

   virtual int calc_dW(int arborId = 0);
   void connOutOfMemory(const char* funcname);

   virtual int setParams(PVParams* params);
   virtual void readChannelCode(PVParams * params);
   virtual void readNumAxonalArbors(PVParams * params);
   virtual void readPlasticityFlag(PVParams * params);
   virtual void readPvpatchAccumulateType(PVParams * params);
   virtual void readPreActivityIsNotRate(PVParams * params);
   virtual void readWriteCompressedWeights(PVParams * params);
   virtual void readWriteCompressedCheckpoints(PVParams * params);
   virtual void readSelfFlag(PVParams * params);
   virtual void readCombine_dW_with_W_flag(PVParams * params);
   virtual void read_dWMax(PVParams * params);
   virtual void readShrinkPatches(PVParams * params);
   virtual void readWriteStep(PVParams * params);
   virtual void readInitialWriteTime(PVParams * params);
   virtual void readDelay(PVParams * params);
   virtual int readPatchSize(PVParams * params);
   virtual int readPatchSizeFromFile(const char * filename);
   virtual int checkShrunkenSize();
   virtual int readNfp(PVParams * params);
   virtual void readUseListOfArborFiles(PVParams * params);
   virtual void readCombineWeightFiles(PVParams * params);
   virtual void readUpdateGSynFromPostPerspective(PVParams * params);

   virtual int setNeededRNGSeeds();

#ifdef PV_USE_OPENCL
   virtual void initIgnoreGPUFlag(); // sets the ignoreGPUFlag parameter.  virtual so that a class can make it always false or always true
   int initializeGPU(); //this method sets up GPU stuff...
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);

   CLKernel * krRecvSyn;        // CL kernel for layer recvSynapticInput call
   cl_event   evRecvSyn;
   cl_event * evRecvSynWaitList;
   int numWait;  //number of receive synaptic runs to wait for (=numarbors)
   //cl_event   evCopyDataStore;

   size_t nxl;
   size_t nyl;

   // OpenCL buffers
   CLBuffer *  clGSyn;
   CLBuffer *  clPatch2DataLookUpTable;
   CLBuffer *  clActivity;
   CLBuffer ** clWeights;

   // ids of OpenCL arguments that change
   //
   int clArgIdOffset;
   int clArgIdWeights;
   int clArgIdDataStore;
#endif // PV_USE_OPENCL

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

protected:
   static inline int computeMargin(int prescale, int postscale, int patchsize) {
   // 2^prescale is the distance between adjacent neurons in pre-layer, thus a smaller prescale means a layer with more neurons
      int margin = 0;
      if (prescale<postscale) { // Density of pre is greater than density of pre
         assert(patchsize%2==1);
         int densityratio = (int) powf(2.0f,(float)(postscale-prescale));
         margin = ((patchsize-1)/2) * densityratio;
      }
      else
      {
         int densityratio = (int) powf(2.0f,(float)(prescale-postscale));
         int numcells = patchsize/densityratio;
         assert(numcells*densityratio==patchsize && numcells%2==1);
         margin = (numcells-1)/2;
      }
      return margin;
   }

};

} // namespace PV

#endif /* HYPERCONN_HPP_ */
