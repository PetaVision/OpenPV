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
#include "../columns/Random.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../io/BaseConnectionProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../utils/Timer.hpp"
#include "../weightinit/InitWeights.hpp"
#include <stdlib.h>
#include <vector>

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLKernel.hpp"
#endif
#ifdef PV_USE_CUDA
#include "../cudakernels/CudaRecvPost.hpp"
#include "../cudakernels/CudaRecvPre.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
#endif

#define PROTECTED_NUMBER 13
#define MAX_ARBOR_LIST (1+MAX_NEIGHBORS)

namespace PV {


//class HyPerCol;
//class HyPerLayer;
class InitWeights;
class BaseConnectionProbe;
class PVParams;
class CloneConn;
class PlasticCloneConn;
class NormalizeBase;
class Random;

/**
 * A HyPerConn identifies a connection between two layers
 */

class HyPerConn {

public:
   friend class CloneConn;
   HyPerConn(const char * name, HyPerCol * hc);
   virtual ~HyPerConn();
//#ifdef PV_USE_OPENCL
//   virtual int deliverOpenCL(Publisher * pub, const PVLayerCube * cube);
//#endif

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   int ioParams(enum ParamsIOFlag ioFlag);

   // TODO The two routines below shouldn't be public, but HyPerCol needs to call them, so for now they are.
   void setInitInfoCommunicatedFlag() {initInfoCommunicatedFlag = true;}
   void setDataStructuresAllocatedFlag() {dataStructuresAllocatedFlag = true;}
   void setInitialValuesSetFlag() {initialValuesSetFlag = true;}

   bool getInitInfoCommunicatedFlag() {return initInfoCommunicatedFlag;}
   bool getDataStructuresAllocatedFlag() {return dataStructuresAllocatedFlag;}
   bool getInitialValuesSetFlag() {return initialValuesSetFlag;}

   int initializeState(); // Not virtual since all connections should respond to initializeFromCheckpointFlag in the same way.
                          // It calls either readStateFromCheckpoint or initializeWeights, both of which are virtual.
   virtual int checkpointRead(const char * cpDir, double* timef);
   virtual int checkpointWrite(const char * cpDir);
   virtual int writeTimers(FILE* stream);
   virtual int insertProbe(BaseConnectionProbe* p);
   int outputProbeParams();
   virtual int outputState(double time, bool last = false);
   int updateStateWrapper(double time, double dt);
   virtual int updateState(double time, double dt);
   virtual bool needUpdate(double time, double dt);
   virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime);
   virtual int writeWeights(double timed, bool last = false);
   virtual int writeWeights(const char* filename);
   virtual int writeWeights(PVPatch*** patches, pvwdata_t** dataStart,
         int numPatches, const char* filename, double timef, bool compressWeights, bool last);
   virtual int writeTextWeights(const char* filename, int k);

   virtual int writeTextWeightsExtra(PV_Stream * pvstream, int k, int arborID) {
      return PV_SUCCESS;
   }

   virtual int writePostSynapticWeights(double time, bool last);
   int readWeights(const char* filename);
    
   GSynAccumulateType getPvpatchAccumulateType() { return pvpatchAccumulateType; }
   int (*accumulateFunctionPointer)(int nk, float* v, float a, pvwdata_t* w, void* auxPtr);
   int (*accumulateFunctionFromPostPointer)(int nk, float* v, float* a, pvwdata_t* w, float dt_factor, void* auxPtr);
   inline bool preSynapticActivityIsNotRate() {return preActivityIsNotRate;}

   double getWeightUpdatePeriod() {return weightUpdatePeriod;}
   double getWeightUpdateTime() {return weightUpdateTime;}
   double getLastUpdateTime() {return lastUpdateTime;}

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

   inline bool usingSharedWeights() {
      return sharedWeights;
   }

   inline bool getPlasticityFlag() {
      return plasticityFlag;
   };

   inline bool getKeepKernelsSynchronized() { 
      return keepKernelsSynchronized_flag;
   }

   /** Actual mininum weight value */
   virtual float minWeight(int arborId = 0);

   /** Actual maximum weight value */
   virtual float maxWeight(int arborId = 0);

   /** Minimum allowed weight value */
   inline float getWMin() {
      return wMin;
   };

   /** Maximum allowed weight value */
   inline float getWMax() {
      return wMax;
   };

   inline float getDWMax(){
      return dWMax;
   }

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
   inline pvwdata_t* getPlasticIncr(int kPre, int arborId) {
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

   inline pvwdata_t* get_wDataStart(int arborId) {
      return wDataStart[arborId];
   }

   inline pvwdata_t* get_wDataHead(int arborId, int dataIndex) {
      return &wDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   inline pvwdata_t* get_wData(int arborId, int patchIndex) {
      return &wDataStart[arborId][patchToDataLUT(patchIndex) * nxp * nyp * nfp
            + wPatches[arborId][patchIndex]->offset];
   }

   inline pvwdata_t* get_dwDataStart(int arborId) {
      return dwDataStart[arborId];
   }

   inline pvwdata_t* get_dwDataHead(int arborId, int dataIndex) {
      return &dwDataStart[arborId][dataIndex * nxp * nyp * nfp];
   }

   inline pvwdata_t* get_dwData(int arborId, int patchIndex) {
      return &dwDataStart[arborId][patchToDataLUT(patchIndex) * nxp * nyp * nfp
            + wPatches[arborId][patchIndex]->offset];
   }

   inline PVPatch* getWPostPatches(int arbor, int patchIndex) {
      return wPostPatches[arbor][patchIndex];
   }

   inline pvwdata_t* getWPostData(int arbor, int patchIndex) {
      return &wPostDataStart[arbor][patchIndex * nxpPost * nypPost * nfpPost]
            + wPostPatches[arbor][patchIndex]->offset;
   }

   inline pvwdata_t* getWPostData(int arbor) {
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

   inline size_t getGSynPatchStart(int kPre, int arborId) {
      return gSynPatchStart[arborId][kPre];
   }

   inline size_t getAPostOffset(int kPre, int arborId) {
      return aPostOffset[arborId][kPre];
   }

   const char * preSynapticLayerName() {return preLayerName;}
   const char * postSynapticLayerName() {return postLayerName;}

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

   bool getUseWindowPost(){return useWindowPost;}
   bool getUpdateGSynFromPostPerspective(){return updateGSynFromPostPerspective;}

   // uint4 * getRnd_state(int index) { return pvpatchAccumulateType==ACCUMULATE_STOCHASTIC ? &rnd_state[index] : NULL; }
   uint4 * getRandState(int index);

   int sumWeights(int nx, int ny, int offset, pvwdata_t* dataStart, double* sum,
         double* sum2, float* maxVal);
   int scaleWeights(int nx, int ny, int offset, pvwdata_t* dataStart, float sum,
         float sum2, float maxVal);
   virtual int checkNormalizeWeights(float sum, float sum2, float sigma2, float maxVal);
   virtual int checkNormalizeArbor(PVPatch** patches, pvwdata_t** dataStart,
         int numPatches, int arborId);
   virtual int normalizeWeights();

   virtual void addClone(PlasticCloneConn* conn);

   virtual void reduceNumKernelActivations();

   virtual double getNumKernelActivations(int kernelIndex){
      assert(kernelIndex < getNumDataPatches());
      return numKernelActivations[kernelIndex];
   }

   virtual long* getPostToPreActivity(){
      return postToPreActivity;
   }

   virtual void initPatchToDataLUT();
   virtual int patchToDataLUT(int patchIndex);
   virtual int* getPatchToDataLUT(){return patch2datalookuptable;}
   virtual int patchIndexToDataIndex(int patchIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   static int decodeChannel(int channel_code, ChannelType * channel_type);
   static int getPreAndPostLayerNames(const char * name, PVParams * params, char ** preLayerNamePtr, char ** postLayerNamePtr);
   static int inferPreAndPostFromConnName(const char * name, PVParams * params, char ** preLayerNamePtr, char ** postLayerNamePtr);
   virtual int handleMissingPreAndPostLayerNames();
   virtual void ioParam_strength(enum ParamsIOFlag, float * strength, bool warnIfAbsent=true);

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
   // char * filename; // Filename if loading weights from a file
   int fileparams[NUM_WGT_PARAMS]; // The header of the file named by the filename member variable
   int numWeightPatches; // Number of PVPatch structures in buffer pointed to by wPatches[arbor]
   int numDataPatches;   // Number of blocks of pvwdata_t's in buffer pointed to by wDataStart[arbor]

   std::vector <PlasticCloneConn*> clones; //A vector of plastic clones that are cloning from this connection

   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
#ifdef USE_SHMGET
   bool shmget_flag;
   bool *shmget_owner;
   int *shmget_id;
#endif
private:
   PVPatch*** wPatches; // list of weight patches, one set per arbor
   // GTK:: gSynPatchStart redefined as offset from start of associated gSynBuffer
   //pvwdata_t*** gSynPatchStart; //  gSynPatchStart[arborId][kExt] is a pointer to the start of the patch in the post-synaptic GSyn buffer
   size_t** gSynPatchStart;  // gSynPatchStart[arborId][kExt] is the offset to the start of the patch from the beginning of the post-synaptic GSyn buffer for corresponding channel
   size_t** aPostOffset; // aPostOffset[arborId][kExt] is the index of the start of a patch into an extended post-synaptic layer
   int* delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th arbor
   PVPatchStrides postExtStrides;    // sx,sy,sf for a patch mapping into an extended post-synaptic layer
   PVPatchStrides postNonextStrides; // sx,sy,sf for a patch mapping into a non-extended post-synaptic layer
   pvwdata_t** wDataStart;  //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   pvwdata_t** dwDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   int defaultDelay; //added to save params file defined delay...
   float * fDelayArray;
   int delayArraySize;
   char* triggerLayerName;
   double triggerOffset;
   HyPerLayer* triggerLayer;
   bool strengthParamHasBeenWritten;
   int * patch2datalookuptable;
   
   long * postToPreActivity;

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
   pvwdata_t** wPostDataStart;

   PVPatch**** wPostPatchesp; // Pointer to wPatches, but from the postsynaptic perspective
   pvwdata_t*** wPostDataStartp; // Pointer to wDataStart, but from the postsynaptic perspective

   int nxpPost, nypPost, nfpPost;
   int numParams;
   //PVConnParams * params;
   float wMax;
   float wMin;
   int numProbes;
   BaseConnectionProbe** probes; // probes used to output data
   bool ioAppend; // controls opening of binary files
   double wPostTime; // time of last conversion to wPostPatches
   double initialWriteTime;
   double writeTime; // time of next output, initialized in params file parameter initialWriteTime
   double writeStep; // output time interval
   bool writeCompressedWeights; // if true, outputState writes weights with 8-bit precision; if false, write weights with float precision
   bool writeCompressedCheckpoints; // similar to writeCompressedWeights, but for checkpointWrite instead of outputState
   int fileType; // type ID for file written by PV::writeWeights

   Timer * io_timer;
   Timer * update_timer;

   bool sharedWeights; // Set to true for the old KernelConn behavior
   bool plasticityFlag;
   bool triggerFlag;
   bool combine_dW_with_W_flag; // indicates that dwDataStart should be set equal to wDataStart, useful for saving memory when weights are not being learned but not used
   bool selfFlag; // indicates that connection is from a layer to itself (even though pre and post may be separately instantiated)
   char * normalizeMethod;
   NormalizeBase * normalizer;
   // bool normalize_flag; // replaced by testing whether normalizer!=NULL
   float normalize_strength;
   bool normalizeArborsIndividually; // if true, each arbor is normalized individually, otherwise, arbors normalized together
   bool normalize_max;
   bool normalize_zero_offset;
   bool normalize_RMS_amp;
   float normalize_cutoff;
   bool shrinkPatches_flag;
   float shrinkPatchesThresh;
   //This object handles calculating weights.  All the initialize weights methods for all connection classes
   //are being moved into subclasses of this object.  The default root InitWeights class will create
   //2D Gaussian weights.  If weight initialization type isn't created in a way supported by Buildandrun,
   //this class will try to read the weights from a file or will do a 2D Gaussian.
   char * weightInitTypeString;
   InitWeights* weightInitializer;
   char * pvpatchAccumulateTypeString;
   GSynAccumulateType pvpatchAccumulateType;
   bool preActivityIsNotRate; // TODO Rename this member variable
   bool normalizeTotalToPost; // if false, normalize the sum of weights from each presynaptic neuron.  If true, normalize the sum of weights into a postsynaptic neuron.
   float dWMax;  // dW scale factor
   bool useListOfArborFiles;
   bool combineWeightFiles;
   bool updateGSynFromPostPerspective;

   double weightUpdatePeriod;
   double weightUpdateTime;
   double initialWeightUpdateTime;
   double lastUpdateTime;

   bool symmetrizeWeightsFlag;
   int* numKernelActivations;
   bool keepKernelsSynchronized_flag;

   // unsigned int rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use by this layer
   // uint4 * rnd_state; // An array of RNGs.
   Random * randState;

   bool initInfoCommunicatedFlag;
   bool dataStructuresAllocatedFlag;
   bool initialValuesSetFlag;
   bool initializeFromCheckpointFlag;

protected:
   HyPerConn();
   virtual int initNumWeightPatches();
   virtual int initNumDataPatches();

   inline PVPatch*** get_wPatches() {
      return wPatches;
   } // protected so derived classes can use; public methods are weights(arbor) and getWeights(patchindex,arbor)

   inline void set_wPatches(PVPatch*** patches) {
      wPatches = patches;
   }

//   inline pvwdata_t*** getGSynPatchStart() {
//      return gSynPatchStart;
//   }

public:
   inline size_t** getGSynPatchStart() {
      return gSynPatchStart;
   }
protected:

//   inline void setGSynPatchStart(pvwdata_t*** patchstart) {
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

   inline pvwdata_t** get_wDataStart() {
      return wDataStart;
   }

   inline void set_wDataStart(pvwdata_t** datastart) {
      wDataStart = datastart;
   }

   inline void set_wDataStart(int arborId, pvwdata_t* pDataStart) {
      wDataStart[arborId] = pDataStart;
   }

   inline pvwdata_t** get_dwDataStart() {
      return dwDataStart;
   }

   inline void set_dwDataStart(pvwdata_t** datastart) {
      dwDataStart = datastart;
   }

   inline void set_dwDataStart(int arborId, pvwdata_t* pIncrStart) {
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
   int initialize_base();
   virtual int createArbors();
   void createArborsOutOfMemory();
   int initializeDelays(const float * fDelayArray, int size);
   virtual int constructWeights();
   int initialize(const char * name, HyPerCol * hc);
   virtual int setPreAndPostLayerNames();
   virtual int setWeightInitializer();
   virtual InitWeights * createInitWeightsObject(const char * weightInitTypeStr);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_preLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_postLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);
   // virtual void ioParam_initWeightsFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag);
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   virtual void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_preActivityIsNotRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);
   virtual void ioParam_selfFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_delay(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shrinkPatchesThresh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shmget_flag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
   virtual void ioParam_useWindowPost(enum ParamsIOFlag ioFlag);

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag);
   virtual void ioParam_preDataLocal(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numXLocal(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numYLocal(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numFLocal(enum ParamsIOFlag ioFlag);
   virtual void ioParam_postGroupXSize(enum ParamsIOFlag ioFlag);
   virtual void ioParam_postGroupYSize(enum ParamsIOFlag ioFlag);
#endif

   int setParent(HyPerCol * hc);
   int setName(const char * name);
   int setPreLayerName(const char * pre_name);
   int setPostLayerName(const char * post_name);
   virtual int initPlasticityPatches();
   virtual int setPatchSize(); // Sets nxp, nyp, nfp if weights are loaded from file.  Subclasses override if they have specialized ways of setting patch size that needs to go in the communicate stage.
                               // (e.g. BIDSConn uses pre and post layer size to set nxp,nyp, but pre and post aren't set until communicateInitInfo().
   virtual void handleDefaultSelfFlag(); // If selfFlag was not set in params, set it in this function.
   virtual PVPatch*** initializeWeights(PVPatch*** arbors, pvwdata_t** dataStart);
   virtual InitWeights* getDefaultInitWeightsMethod(const char* keyword);
   virtual InitWeights* handleMissingInitWeights(PVParams* params);
   virtual int createWeights(PVPatch*** patches, int nWeightPatches, int nDataPatches, int nxPatch,
         int nyPatch, int nfPatch, int arborId);
   int createWeights(PVPatch*** patches, int arborId);
   virtual pvwdata_t * allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch);
   int clearWeights(pvwdata_t** dataStart, int numPatches, int nx, int ny, int nf);
   virtual int adjustAllPatches(int nxPre, int nyPre, int nfPre, const PVHalo * haloPre, int nxPost, int nyPost, int nfPost, const PVHalo * haloPost, PVPatch*** inWPatches, size_t** inGSynPatchStart, size_t** inAPostOffset, int arborId);
   virtual int adjustAxonalArbors(int arborId);
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readWeightsFromCheckpoint(const char * cpDir, double * timeptr);
   int checkpointFilename(char * cpFilename, int size, const char * cpDir);
   virtual int setInitialValues(); // returns PV_SUCCESS if successful, or PV_POSTPONE if it needs to wait on other objects (e.g. TransposeConn has to wait for original conn)
   virtual int clear_dW();
   virtual int calc_dW(int arborId = 0);
   virtual int update_dW(int arborId);
   virtual int defaultUpdate_dW(int arborId);
   virtual int defaultUpdateInd_dW(int arbor_ID, int kExt);
   virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);
   virtual int updateWeights(int arborId = 0);
   virtual int normalize_dW(int arbor_ID);
   virtual bool skipPre(pvdata_t preact){return preact == 0.0f;};
#ifdef PV_USE_MPI
   virtual int reduceKernels(int arborID);
// #else
//    virtual int reduceKernels(int arborID){return PV_SUCCESS;};
#endif // PV_USE_MPI

   void connOutOfMemory(const char* funcname);

//GPU variables
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
public:
   //TODO check if kernel name is ever needed
   void setAllocDeviceWeights(){
      allocDeviceWeights = true;
   }
#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceWData(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceWData(){
#endif
      return d_WData;
   }
#ifdef PV_USE_OPENCL
   CLBuffer * getDevicePatches(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDevicePatches(){
#endif
      return d_Patches;
   }
#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceGSynPatchStart(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceGSynPatchStart(){
#endif
      return d_GSynPatchStart;
   }
#ifdef PV_USE_OPENCL
   void setDeviceWData(CLBuffer* inBuf){
#endif
#ifdef PV_USE_CUDA
   void setDeviceWData(PVCuda::CudaBuffer* inBuf){
#endif
      d_WData = inBuf;
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   PVCuda::CudaBuffer * getCudnnWData(){
      return cudnn_WData;
   }
   void setCudnnWData(PVCuda::CudaBuffer* inBuf){
      cudnn_WData = inBuf;
   }
#endif

#ifdef PV_USE_OPENCL
   void clFinishW(){ 
      if(allocDeviceWeights && d_WData){
         d_WData->finish();
      }
   }
#endif

   bool getReceiveGpu(){return receiveGpu;}
   bool getPreDataLocal(){return preDataLocal;}

#ifdef PV_USE_OPENCL
   CLKernel * getKrRecvPost(){return krRecvPost;}
   CLKernel * getKrRecvPre(){return krRecvPre;}
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaRecvPost * getKrRecvPost(){return krRecvPost;}
   PVCuda::CudaRecvPre * getKrRecvPre(){return krRecvPre;}
#endif

   virtual int getNumXLocal(){return numXLocal;}
   virtual int getNumYLocal(){return numYLocal;}
   virtual int getNumFLocal(){return numFLocal;}
   virtual int getPostGroupXSize(){return postGroupXSize;}
   virtual int getPostGroupYSize(){return postGroupYSize;}
   virtual int getNumPostGroupX(){return numPostGroupX;}
   virtual int getNumPostGroupY(){return numPostGroupY;}

   
   
   bool getUpdatedDeviceWFlag(){
      return updatedDeviceWeights;
   }
   void setUpdatedDeviceWFlag(bool in){
      updatedDeviceWeights = in;
   }
   
protected:
   virtual int allocateDeviceBuffers();
   virtual int allocateReceivePostKernel();
   virtual int allocateReceivePreKernel();

   bool allocDeviceWeights;
   bool updatedDeviceWeights;

#ifdef PV_USE_OPENCL
   CLBuffer * d_WData;
   CLBuffer * d_Patches;
   CLBuffer * d_GSynPatchStart;
   CLBuffer * d_PostToPreActivity;
   CLBuffer * d_Patch2DataLookupTable;
   CLKernel * krRecvPost;        // CL kernel for update state call
   CLKernel * krRecvPre;        // CL kernel for update state call
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * d_WData;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer * cudnn_WData;
#endif
   PVCuda::CudaBuffer * d_Patches;
   PVCuda::CudaBuffer * d_GSynPatchStart;
   PVCuda::CudaBuffer * d_PostToPreActivity;
   PVCuda::CudaBuffer * d_Patch2DataLookupTable;
   PVCuda::CudaRecvPost* krRecvPost;        // Cuda kernel for update state call
   PVCuda::CudaRecvPre* krRecvPre;        // Cuda kernel for update state call
#endif

   PVPatch ** localWPatches;
   size_t * localGSynPatchStart;

   bool receiveGpu;
   bool preDataLocal;

   int numXLocal;
   int numYLocal;
   int numFLocal;
   int localPreX;
   int localPreY;
   int postGroupXSize;
   int postGroupYSize;
   int numPostGroupX;
   int numPostGroupY;




//   bool gpuAccelerateFlag; // Whether to accelerate the connection on a GPU
//   bool ignoreGPUflag;     // Don't use GPU (overrides gpuAccelerateFlag)
//   virtual void initIgnoreGPUFlag(); // sets the ignoreGPUFlag parameter.  virtual so that a class can make it always false or always true
//   int initializeGPU(); //this method sets up GPU stuff...
//   virtual int initializeThreadKernels(const char * kernelName);
//   virtual int initializeThreadBuffers(const char * kernelName);
//
//   CLKernel * krRecvSyn;        // CL kernel for layer recvSynapticInput call
//   cl_event   evRecvSyn;
//   cl_event * evRecvSynWaitList;
//   int numWait;  //number of receive synaptic runs to wait for (=numarbors)
//   //cl_event   evCopyDataStore;
//
//   size_t nxl;
//   size_t nyl;
//
//   // OpenCL buffers
//   CLBuffer *  clGSyn;
//   CLBuffer *  clPatch2DataLookUpTable;
//   CLBuffer *  clActivity;
//   CLBuffer ** clWeights;

//
//   // ids of OpenCL arguments that change
//   //
//   int clArgIdOffset;
//   int clArgIdWeights;
//   int clArgIdDataStore;
#endif // PV_USE_OPENCL

private:
   int clearWeights(pvwdata_t* arborDataStart, int numPatches, int nx, int ny,
         int nf);
   int deleteWeights();

// static member functions
//
public:
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

   static inline
   void pvpatch_init(PVPatch * p, int nx, int ny)
   {
      p->nx = nx;
      p->ny = ny;
      p->offset = 0;
   }

   static inline
   void pvpatch_adjust(PVPatch * p, int sx, int sy, int nxNew, int nyNew, int dx, int dy)
   {
      p->nx = nxNew;
      p->ny = nyNew;
      p->offset += dx * sx + dy * sy;
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
