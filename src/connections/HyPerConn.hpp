/*
 * HyPerConnection.hpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "BaseConnection.hpp"
#include "../columns/InterColComm.hpp"
#include "../columns/HyPerCol.hpp"
#include "../columns/Random.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../io/BaseConnectionProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../utils/Timer.hpp"
#include <stdlib.h>
#include <vector>
#include <stdlib.h>
#include <set>
#include <map>

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

struct SparseWeightInfo {
   unsigned long size;
   pvwdata_t thresholdWeight;
   float percentile;
};

//class HyPerCol;
//class HyPerLayer;
class InitWeights;
class BaseConnectionProbe;
class PVParams;
class CloneConn;
class PlasticCloneConn;
class NormalizeBase;
class Random;
class TransposeConn;
class privateTransposeConn;

/**
 * A HyPerConn identifies a connection between two layers
 */

class HyPerConn : public BaseConnection {

public:
   friend class CloneConn;
   friend class PlasticCloneConn;
   friend class TransposeConn;
   friend class privateTransposeConn;
   friend class TransposePoolingConn;
   
   HyPerConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~HyPerConn();
//#ifdef PV_USE_OPENCL
//   virtual int deliverOpenCL(Publisher * pub, const PVLayerCube * cube);
//#endif

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   virtual int checkpointRead(const char * cpDir, double* timef);
   virtual int checkpointWrite(const char * cpDir);
   virtual int writeTimers(FILE* stream);
   virtual int insertProbe(BaseConnectionProbe* p);
   int outputProbeParams();
   virtual int outputState(double time, bool last = false);
   int updateState(double time, double dt); 
   virtual int finalizeUpdate(double timed, double dt);
   virtual bool needUpdate(double time, double dt);
   virtual int updateInd_dW(int arbor_ID, int batch_ID, int kExt);
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

   /**
    * Uses presynaptic layer's activity to modify the postsynaptic GSyn or thread_gSyn
    */
   virtual int deliver();
   virtual void deliverOnePreNeuronActivity(int patchIndex, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr);
   virtual void deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, taus_uint4 * rngPtr);
    
   GSynAccumulateType getPvpatchAccumulateType() { return pvpatchAccumulateType; }
   int (*accumulateFunctionPointer)(int kPreRes, int nk, float* v, float a, pvwdata_t* w, void* auxPtr, int sf);
   int (*accumulateFunctionFromPostPointer)(int kPreRes, int nk, float* v, float* a, pvwdata_t* w, float dt_factor, void* auxPtr, int sf);

   double getWeightUpdatePeriod() {return weightUpdatePeriod;}
   double getWeightUpdateTime() {return weightUpdateTime;}
   double getLastUpdateTime() {return lastUpdateTime;}

   // TODO make a get-method to return this.
   virtual PVLayerCube* getPlasticityDecrement() {
      return NULL;
   }

   inline InitWeights* getWeightInitializer() {
      return weightInitializer;
   }

   inline bool getSelfFlag() {
      return selfFlag;
   }

   inline bool usingSharedWeights() {
      return sharedWeights;
   }

   //plasticityFlag moved to parent class BaseConnection on Jan 26, 2015.
   //inline bool getPlasticityFlag() {
   //   return plasticityFlag;
   //};

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
                  &dwDataStart[arborId][patchStartIndex(kPre)
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
      return &wDataStart[arborId][patchStartIndex(dataIndex)];
   }

   inline pvwdata_t* get_wData(int arborId, int patchIndex) {
      return &wDataStart[arborId][patchStartIndex(patchToDataLUT(patchIndex))
            + wPatches[arborId][patchIndex]->offset];
   }

   inline pvwdata_t* get_dwDataStart(int arborId) {
      return dwDataStart[arborId];
   }

   inline long* get_activations(int arborId){
      return numKernelActivations[arborId];
   }

   inline pvwdata_t* get_dwDataHead(int arborId, int dataIndex) {
      return &dwDataStart[arborId][patchStartIndex(dataIndex)];
   }

   inline long* get_activationsHead(int arborId, int dataIndex){
      return &numKernelActivations[arborId][patchStartIndex(dataIndex)];
   }

   inline pvwdata_t* get_dwData(int arborId, int patchIndex) {
      return &dwDataStart[arborId][patchStartIndex(patchToDataLUT(patchIndex))
            + wPatches[arborId][patchIndex]->offset];
   }

   inline long* get_activations(int arborId, int patchIndex){
      return &numKernelActivations[arborId][patchStartIndex(patchToDataLUT(patchIndex))
            + wPatches[arborId][patchIndex]->offset];
   }

   inline PVPatch* getWPostPatches(int arbor, int patchIndex) {
      return wPostPatches[arbor][patchIndex];
   }

   inline pvwdata_t* getWPostData(int arbor, int patchIndex) {
      return &wPostDataStart[arbor][postPatchStartIndex(patchIndex)]
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

   inline size_t getGSynPatchStart(int kPre, int arborId) {
      return gSynPatchStart[arborId][kPre];
   }

   inline size_t getAPostOffset(int kPre, int arborId) {
      return aPostOffset[arborId][kPre];
   }

   NormalizeBase * getNormalizer() {
      return normalizer;
   }

   bool getNormalizeDwFlag() {
      return normalizeDwFlag;
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

   bool getUpdateGSynFromPostPerspective(){return updateGSynFromPostPerspective;}

   taus_uint4 * getRandState(int index);

   int sumWeights(int nx, int ny, int offset, pvwdata_t* dataStart, double* sum,
         double* sum2, float* maxVal);

   virtual void addClone(PlasticCloneConn* conn);

   //virtual int reduceActivations(int arborID);

   virtual long * getPostToPreActivity(){
      return postToPreActivity;
   }

   virtual void initPatchToDataLUT();
   virtual int patchToDataLUT(int patchIndex);
   virtual int* getPatchToDataLUT(){return patch2datalookuptable;}
   virtual int patchIndexToDataIndex(int patchIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int* kx = NULL, int* ky =
         NULL, int* kf = NULL);

   void setNeedPost(bool inBool){needPost = inBool;}
   void setNeedAllocPostWeights (bool inBool){needAllocPostWeights = inBool;}

protected:
   // char * filename; // Filename if loading weights from a file
   int fileparams[NUM_WGT_PARAMS]; // The header of the file named by the filename member variable
   int numWeightPatches; // Number of PVPatch structures in buffer pointed to by wPatches[arbor]
   int numDataPatches;   // Number of blocks of pvwdata_t's in buffer pointed to by wDataStart[arbor]
   bool needAllocPostWeights;

   std::vector <PlasticCloneConn*> clones; //A vector of plastic clones that are cloning from this connection

   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
private:
   PVPatch*** wPatches; // list of weight patches, one set per arbor
   // GTK:: gSynPatchStart redefined as offset from start of associated gSynBuffer
   //pvwdata_t*** gSynPatchStart; //  gSynPatchStart[arborId][kExt] is a pointer to the start of the patch in the post-synaptic GSyn buffer
   size_t** gSynPatchStart;  // gSynPatchStart[arborId][kExt] is the offset to the start of the patch from the beginning of the post-synaptic GSyn buffer for corresponding channel
   size_t** aPostOffset; // aPostOffset[arborId][kExt] is the index of the start of a patch into an extended post-synaptic layer
   PVPatchStrides postExtStrides;    // sx,sy,sf for a patch mapping into an extended post-synaptic layer
   PVPatchStrides postNonextStrides; // sx,sy,sf for a patch mapping into a non-extended post-synaptic layer
   pvwdata_t** wDataStart;  //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   pvwdata_t** dwDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   //int defaultDelay; //added to save params file defined delay...
   bool strengthParamHasBeenWritten;
   int * patch2datalookuptable;
   
   long * postToPreActivity;

   bool needPost;

   // All weights that are above the threshold
   typedef pvwdata_t WeightType;
   typedef std::vector<WeightType> WeightListType;
   typedef std::vector<int> IndexListType;

   // Percentage of weights that are ignored. Weight values must be above this threshold
   // to be included in the calculation. Valid values are 0.0 - 1.0. But there's no
   // point in setting this to 1.0.
   float _weightSparsity;
   // The output offset into the post layer for a weight
   std::vector<int> _sparsePost;
   // Start of sparse weight data in the _sparseWeight array, indexed by data patch
   std::vector<int> _patchSparseWeightIndex;
   // Number of sparse weights for a patch, indexed by data patch
   std::vector<int> _patchSparseWeightCount;
   // Have sparse weights been allocated for each arbor?
   std::vector<bool> _sparseWeightsAllocated;

   typedef std::map<const WeightType * const, const WeightListType> WeightPtrMapType;
   typedef std::map<const WeightType * const, const IndexListType>  WeightIndexMapType;

   // Map nk -> weight ptr -> sparse weights
   typedef std::map<int, WeightPtrMapType> WeightMapType;
   // Map nk -> weight ptr -> output index
   typedef std::map<int, WeightIndexMapType> IndexMapType;

   WeightMapType _sparseWeightValues;
   IndexMapType _sparseWeightIndexes;
   SparseWeightInfo _sparseWeightInfo;

   std::set<int> _kPreExtWeightSparsified;

   unsigned long _numDeliverCalls; // Number of times deliver has been called
   unsigned long _allocateSparseWeightsFrequency; // Number of _numDeliverCalls that need to happen before the pre list needs to be rebuilt

   // Allocate sparse weights when performing presynaptic delivery
   void allocateSparseWeightsPre(PVLayerCube const *activity, int arbor);
   // Allocate sparse weights when performing postsynaptic delivers
   void allocateSparseWeightsPost(PVLayerCube const *activity, int arbor);
   // Calculates the sparse weight threshold
   SparseWeightInfo calculateSparseWeightInfo() const;
   SparseWeightInfo findPercentileThreshold(float percentile, pvwdata_t **wDataStart, size_t numAxonalArborLists, size_t numPatches, size_t patchSize) const;
   void deliverOnePreNeuronActivitySparseWeights(int kPreExt, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr);
   void deliverOnePostNeuronActivitySparseWeights(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, taus_uint4 * rngPtr);


protected:
   HyPerConn* postConn;
   bool needFinalize;
   bool useMask;
   char* maskLayerName;
   int maskFeatureIdx;
   HyPerLayer* mask;
   bool* batchSkip;

   bool normalizeDwFlag;

   int nxp, nyp, nfp; // size of weight dimensions
   bool warnDefaultNfp; // Whether to print a warning if the default nfp is used.
   int sxp, syp, sfp; // stride in x,y,features
   // PVPatch       *** dwPatches;      // list of weight patches for storing changes to weights
   PVPatch*** wPostPatches; // post-synaptic linkage of weights // This is being deprecated in favor of TransposeConn
   pvwdata_t** wPostDataStart;

   PVPatch**** wPostPatchesp; // Pointer to wPatches, but from the postsynaptic perspective
   pvwdata_t*** wPostDataStartp; // Pointer to wDataStart, but from the postsynaptic perspective

   int nxpPost, nypPost, nfpPost;
   int numParams;
   //PVConnParams * params;
   float wMax;
   float wMin;
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
   // bool plasticityFlag; // Moved to base class BaseConnection on Jan 26, 2015
   bool triggerFlag;
   char* triggerLayerName;
   double triggerOffset;
   HyPerLayer* triggerLayer;
   bool combine_dW_with_W_flag; // indicates that dwDataStart should be set equal to wDataStart, useful for saving memory when weights are not being learned but not used
   bool selfFlag; // indicates that connection is from a layer to itself (even though pre and post may be separately instantiated)
   char * normalizeMethod;
   NormalizeBase * normalizer;
   char * normalizeGroupName; // If normalizeMethod is normalizeGroup, holds the name of the group.  Otherwise null
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
   bool normalizeTotalToPost; // if false, normalize the sum of weights from each presynaptic neuron.  If true, normalize the sum of weights into a postsynaptic neuron.
   float dWMax;  // dW scale factor
   bool useListOfArborFiles;
   bool combineWeightFiles;
   bool updateGSynFromPostPerspective;

   pvdata_t ** thread_gSyn; //Accumulate buffer for each thread, only used if numThreads > 1 // Move back to HyPerLayer?

   double weightUpdatePeriod;
   double weightUpdateTime;
   double initialWeightUpdateTime;
   double lastUpdateTime;

   bool symmetrizeWeightsFlag;
   long ** numKernelActivations;
   bool keepKernelsSynchronized_flag;

   // unsigned int rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use by this layer
   // taus_uint4 * rnd_state; // An array of RNGs.
   Random * randState;



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

public:
   inline size_t** getGSynPatchStart() {
      return gSynPatchStart;
   }
protected:

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

   inline long** get_activations(){
      return numKernelActivations;
   }

   inline void set_dwDataStart(pvwdata_t** datastart) {
      dwDataStart = datastart;
   }

   inline void set_dwDataStart(int arborId, pvwdata_t* pIncrStart) {
      dwDataStart[arborId] = pIncrStart;
   }

   inline size_t patchStartIndex(int patchIndex) {
      return (size_t) patchIndex * (size_t) (nxp*nyp*nfp);
   }

   inline size_t postPatchStartIndex(int patchIndex) {
      return (size_t) patchIndex * (size_t) (nxpPost*nypPost*nfpPost);
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
   virtual int constructWeights();
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   virtual int setWeightInitializer(); // Deprecated Feb 9, 2015; the preferred way of specifying the weight initialization is by passing an InitWeights argument to the constructor.
   virtual InitWeights * createInitWeightsObject(const char * weightInitTypeStr); // Deprecated Feb 9, 2015.
   int setWeightNormalizer(); // Included Feb 17, 2015 for backward compatibility only.  The preferred way of specifying the normalizer is by passing a NormalizeBase argument to the constructor.
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /** 
    * List of parameters needed from the HyPerConn class
    * @name HyPerConn Parameters
    * @{
    */

   /**
    * @brief channelCode: Specifies which channel in the post layer this connection is attached to
    * @details Channels can be -1 for no update, or >= 0 for channel number. <br />
    * 0 is excitatory, 1 is inhibitory
    */
   virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);

   // virtual void ioParam_initWeightsFile(enum ParamsIOFlag ioFlag);
   /**
    * @brief sharedWeights: Defines if the HyPerConn uses shared weights (kernelConn)
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);

   /**
    * @brief weightInitType: Specifies the initialization method of weights
    * @details Possible choices are
    * - @link InitGauss2DWeightsParams Gauss2DWeight@endlink: 
    *   Initializes weights with a gaussian distribution in x and y over each f
    *
    * - @link InitCocircWeightsParams CoCircWeight@endlink:
    *   Initializes cocircular weights
    *
    * - @link InitUniformWeightsParams UniformWeight@endlink:
    *   Initializes weights with a single uniform weight
    *
    * - @link InitSmartWeights SmartWeight@endlink:
    *   TODO
    *
    * - @link InitUniformRandomWeightsParams UniformRandomWeight@endlink:
    *   Initializes weights with a uniform distribution
    *
    * - @link InitGaussianRandomWeightsParams GaussianRandomWeight@endlink:
    *   Initializes individual weights with a gaussian distribution
    *
    * - @link InitIdentWeightsParams IdentWeight@endlink:
    *   Initializes weights for ident conn (one to one with a strength to 1)
    *
    * - @link InitOneToOneWeightsParams OneToOneWeight@endlink:
    *   Initializes weights as a multiple of the identity matrix
    *
    * - @link InitOneToOneWeightsWithDelaysParams OneToOneWeightsWithDelays@endlink:
    *   Initializes weights as a multiple of the identity matrix with delays
    *
    * - @link InitSpreadOverArborsWeightsParams SpreadOverArborsWeight@endlink:
    *   Initializes weights where different part of the weights over different arbors
    *
    * - @link InitWeightsParams FileWeight@endlink:
    *   Initializes weights from a specified pvp file.
    *
    * Further parameters are needed depending on initialization type
    */

   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag);

   // plasticityFlag was moved to base class BaseConnection on Jan 26, 2015.
   // /**
   //  * @brief plasticityFlag: Specifies if the weights will be updated
   //  */
   // virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief weightUpdatePeriod: If plasticity flag is set, specifies the update period of weights
    */
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);

   /**
    * @brief initialWeightUpdateTime: If plasticity flag is set, specifies the inital weight update time; ignored if triggerFlag = true
    */
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerFlag: If plasticity flag is set, allows weight updates to be triggered off layers
    */
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: If trigger flag is set, specifies the layer to trigger off of
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerOffset: If trigger flag is set, triggers <triggerOffset> timesteps before target trigger
    * @details Defaults to 0.
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);

   /**
    * @brief pvpatchAccumulateType: Specifies the method to accumulate synaptic input
    * @details Possible choices are
    * - convolve: Accumulates through convolution
    * - stochastic: Accumulates through stochastic release
    * - maxpooling: Accumulates through max pooling
    * - sumpooling: Accumulates through sum pooling
    *
    * Defaults to convolve.
    */
   virtual void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeStep: Specifies the write period of the connection.
    * @details Defaults to every timestep. -1 to not write at all.
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /**
    * @brief initialWriteTime: If writeStep is >= 0, sets the initial write time of the connection.
    */
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeCompressedWeights: If writeStep >= 0, weights written out are bytes as opposed to floats.
    */
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeCompressedCheckpoints: Checkpoint weights are written compressed.
    * @details The parent HyPerCol must be writing checkpoints for this flag to be used
    */
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);

   /**
    * @brief selfFlag: Indicates if pre and post is the same layer.
    * @details The default value for selfFlag should be pre==post, but at the time ioParams(PARAMS_IO_READ) is called,
    * pre and post have not been set.  So we read the value with no warning if it's present;
    * if it's absent, set the value to pre==post in the communicateInitInfo stage and issue
    * the using-default-value warning then.
    */
   virtual void ioParam_selfFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief combine_dW_with_W_flag: If plasticity flag is set, specifies if dW buffer is allocated
    * @details dW buffer, if not allocated, will point to weight buffer and accumulate weights as
    * it gets them
    */
   virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag);

   /**
    * @brief nxp: Specifies the x patch size
    * @details If one pre to many post, nxp restricted to many * an odd number
    * If many pre to one post or one pre to one post, nxp restricted to an odd number
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);

   /**
    * @brief nyp: Specifies the y patch size
    * @details If one pre to many post, nyp restricted to many * an odd number
    * If many pre to one post or one pre to one post, nyp restricted to an odd number
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);

   /**
    * @brief nxpShrunken: Specifies a shrunken patch size (deprecated)
    */
   virtual void ioParam_nxpShrunken(enum ParamsIOFlag ioFlag); // Deprecated Feb 2, 2015

   /**
    * @brief nypShrunken: Specifies a shrunken patch size (deprecated)
    */
   virtual void ioParam_nypShrunken(enum ParamsIOFlag ioFlag); // Deprecated Feb 2, 2015

   /**
    * @brief nfp: Specifies the post feature patch size
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);

   /**
    * @brief shrinkPatches: Optimization for shrinking a patch to it's non-zero values 
    */
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief shrinkPatchesThresh: If shrinkPatches flag is set, specifies threshold to consider weight as zero
    */
   virtual void ioParam_shrinkPatchesThresh(enum ParamsIOFlag ioFlag);

   /**
    * @brief updateGSynFromPostPerspective: Specifies if the connection should push from pre or pull from post.
    */
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag);

   /**
    * @brief dWMax: If plasticity flag is set, specifies the learning rate of the weight updates.
    */
   virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief normalizeMethod: Specifies the normalization method for weights
    * @details Weights will be normalized after initialization and after each weight update.
    * Possible choices are:
    * - @link NormalizeSum normalizeSum@endlink: 
    *   Normalization where sum of weights add up to strength
    * - @link NormalizeL2 normalizeL2@endlink: 
    *   Normaliztion method where L2 of weights add up to strength
    * - @link NormalizeMax normalizeMax@endlink: 
    *   Normaliztion method where Max is clamped at strength
    * - @link NormalizeContrastZeroMean normalizeContrastZeroMean@endlink: 
    *   Normalization method for a weight with specified mean and std
    * - @link NormalizeScale normalizeScale@endlink: 
    *   TODO
    * - none: Do not normalize 
    *
    * Further parameters are needed depending on initialization type.
    */
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief groupNormalizerName: If normalizeMethod is set to "normalizeGroup", sets the name of the normalization group for this connection.
    */
   virtual void ioParam_normalizeGroupName(enum ParamsIOFlag ioFlag);

   /**
    * @brief keepKernelsSynchronized: If using sharedWeights and plasticityFlag, sets if kernels should be synchronized during the run.
    */
   virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief normalizeDw: Specifies if this connection is averaging gradients (true) or summing them (false)
    */
   virtual void ioParam_normalizeDw(enum ParamsIOFlag ioFlag);

   /**
    * @brief useMask: Specifies if this connection is using a post mask for learning
    */
   virtual void ioParam_useMask(enum ParamsIOFlag ioFlag);

   /**
    * @brief maskLayerName: If using mask, specifies the layer to use as a binary mask layer
    */
   virtual void ioParam_maskLayerName(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief maskLayerName: If using mask, specifies which feature dim to use for the mask
    * @details Defaults to -1, which means point wise mask
    */
   virtual void ioParam_maskFeatureIdx(enum ParamsIOFlag ioFlag);

   /**
    * @brief gpuGroupIdx: All connections in the same group uses the same GPU memory for weights
    * @details Specify a group index. An index of -1 means no group (default).
    * This parameter is ignored if PetaVision was compiled without GPU acceleration.
    */
   virtual void ioParam_gpuGroupIdx(enum ParamsIOFlag ioFlag);

   /**
    * @brief preDataLocal: If not using CUDNN, specifies if preData should be in local memory.
    * This parameter is ignored if PetaVision was compiled without GPU acceleration.
    */
   virtual void ioParam_preDataLocal(enum ParamsIOFlag ioFlag);
   /**
    * @brief numXLocal: Specifies number of local threads to run in x direction
    * @details Only set if receiving from gpu. Not used if using receive CUDNN from post. 
    * This parameter is ignored if PetaVision was compiled without GPU acceleration.
    */
   virtual void ioParam_numXLocal(enum ParamsIOFlag ioFlag);
   /**
    * @brief numYLocal: Specifies number of local threads to run in xydirection
    * @details Only set if receiving from gpu. Not used if using receive CUDNN from post. 
    * Must be divisible by post layer x size. numXLocal * numYLocal * numFLocal must be less
    * than the amount of local threads specified by the hardware.
    * This parameter is ignored if PetaVision was compiled without GPU acceleration.
    */
   virtual void ioParam_numYLocal(enum ParamsIOFlag ioFlag);
   /**
    * @brief numYLocal: Specifies number of local threads to run in xydirection
    * @details Only set if receiving from gpu. Not used if using receive CUDNN from post. 
    * If using preDataLocal and recv form post, must be set to 1. numXLocal * numYLocal * numFLocal
    * must be less than the amount of local threads specified by the hardware. Must be set to 1.
    * This parameter is ignored if PetaVision was compiled without GPU acceleration.
    */
   virtual void ioParam_numFLocal(enum ParamsIOFlag ioFlag);

   /**
    * @brief weightSparsity: Specifies what percentage of weights will be ignored. Default is 0.0
    */
   virtual void ioParam_weightSparsity(enum ParamsIOFlag ioFlag);

   /** @} */

   int setPreLayerName(const char * pre_name);
   int setPostLayerName(const char * post_name);
   virtual int initPlasticityPatches();
   virtual int setPatchSize(); // Sets nxp, nyp, nfp if weights are loaded from file.  Subclasses override if they have specialized ways of setting patch size that needs to go in the communicate stage.
   virtual int setPostPatchSize(); // Sets nxp, nyp, nfp if weights are loaded from file.  Subclasses override if they have specialized ways of setting patch size that needs to go in the communicate stage.
                               // (e.g. BIDSConn uses pre and post layer size to set nxp,nyp, but pre and post aren't set until communicateInitInfo().
   virtual void handleDefaultSelfFlag(); // If selfFlag was not set in params, set it in this function.
   virtual PVPatch*** initializeWeights(PVPatch*** arbors, pvwdata_t** dataStart);
   virtual InitWeights* getDefaultInitWeightsMethod(const char* keyword);
#ifdef OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.
   virtual InitWeights* handleMissingInitWeights(PVParams* params);
#endif // OBSOLETE // Marked obsolete Mar 20, 2015.  Not used since creating the InitWeights object was taken out of HyPerConn.
   virtual int createWeights(PVPatch*** patches, int nWeightPatches, int nDataPatches, int nxPatch,
         int nyPatch, int nfPatch, int arborId);
   int createWeights(PVPatch*** patches, int arborId);
   virtual pvwdata_t * allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch);
   virtual int allocatePostToPreBuffer();
   virtual int allocatePostConn();

   int clearWeights(pvwdata_t** dataStart, int numPatches, int nx, int ny, int nf);
   virtual int adjustAllPatches(int nxPre, int nyPre, int nfPre, const PVHalo * haloPre, int nxPost, int nyPost, int nfPost, const PVHalo * haloPost, PVPatch*** inWPatches, size_t** inGSynPatchStart, size_t** inAPostOffset, int arborId);
   virtual int adjustAxonalArbors(int arborId);
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readWeightsFromCheckpoint(const char * cpDir, double * timeptr);
   int checkpointFilename(char * cpFilename, int size, const char * cpDir);
   virtual int setInitialValues(); // returns PV_SUCCESS if successful, or PV_POSTPONE if it needs to wait on other objects (e.g. TransposeConn has to wait for original conn)

   /**
    * calc_dW is a function that calls initialze_dW, update_dW, reduce_dW, and normalize_dW
    */
   virtual int calc_dW();

   /**
    * Initializes dW. Default behaviour is to clear dW.
    */
   virtual int initialize_dW(int arborId);
   virtual int clear_dW(int arborId);
   virtual int clear_numActivations(int arborId);
   /**
    * Updates the dW buffer
    */
   virtual int update_dW(int arborId);
   virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);
   virtual int updateWeights(int arborId = 0);
   virtual bool skipPre(pvdata_t preact){return preact == 0.0f;};
   /**
    * Reduces all dW and activations across MPI
    */
   virtual int reduce_dW(int arborId);
   virtual int reduceKernels(int arborID);
   virtual int reduceActivations(int arborID);

   /**
    * Normalizes all dW by dividing dW by activations
    */
   virtual int normalize_dW(int arbor_ID);

   void connOutOfMemory(const char* funcname);

   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID);
   virtual int deliverPostsynapticPerspective(PVLayerCube const * activity, int arborId){
      return deliverPostsynapticPerspective(activity, arborId, NULL, NULL);
   }
   virtual int deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID, int* numActive, int** activeList);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int deliverPresynapticPerspectiveGPU(PVLayerCube const * activity, int arborID);
   virtual int deliverPostsynapticPerspectiveGPU(PVLayerCube const * activity, int arborID);
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

   float getConvertToRateDeltaTimeFactor();

//GPU variables
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
public:
   bool getAllocDeviceWeights(){
      return allocDeviceWeights;
   }
   bool getAllocPostDeviceWeights(){
      return allocPostDeviceWeights;
   }

   virtual void setAllocDeviceWeights(){
      allocDeviceWeights= true;
   }
   virtual void setAllocPostDeviceWeights(){
      allocPostDeviceWeights = true;
   }
#ifdef PV_USE_OPENCL
   virtual CLBuffer * getDeviceWData(){
#endif
#ifdef PV_USE_CUDA
   virtual PVCuda::CudaBuffer * getDeviceWData(){
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
   virtual PVCuda::CudaBuffer * getCudnnWData(){
      return cudnn_WData;
   }
   void setCudnnWData(PVCuda::CudaBuffer* inBuf){
      cudnn_WData = inBuf;
   }
#endif

#ifdef PV_USE_OPENCL
   void clFinishW(){ 
      if(allocDeviceWeights&& d_WData){
         d_WData->finish();
      }
   }
#endif

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
   
protected:
   virtual int allocatePostDeviceWeights();
   virtual int allocateDeviceWeights();
   virtual int allocateDeviceBuffers();
   virtual int allocateReceivePostKernel();
   virtual int allocateReceivePreKernel();
   virtual void updateDeviceWeights();

   bool allocDeviceWeights;
   bool allocPostDeviceWeights;
   //bool updatedDeviceWeights;
   

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
   int gpuGroupIdx;
   bool preDataLocal;
   int numXLocal;
   int numYLocal;
   int numFLocal;



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
   void unsetAccumulateType();

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
      if (prescale==postscale) {
         assert(patchsize%2==1);
         margin = (patchsize-1)/2;
      }
      else if (prescale<postscale) { // Density of pre is greater than density of pre: many-to-one
         // any patchsize is permissible
         int densityratio = (int) powf(2.0f,(float)(postscale-prescale));
         assert(densityratio % 2 == 0);
         margin = (patchsize-1) * densityratio/2;
      }
      else
      {
         assert(prescale>postscale); // one-to-many
         int densityratio = (int) powf(2.0f,(float)(prescale-postscale));
         int numcells = patchsize/densityratio;
         assert(numcells*densityratio==patchsize); // For one-to-many, patchsize must be a multiple of "many".
         margin = numcells/2; // integer division is correct, no matter whether numcells is even or odd
      }
      return margin;
   }

   static inline int adjustedPatchDimension(int zPre, int preNeuronsPerPostNeuron, int postNeuronsPerPreNeuron, int nPost, int patchDim, int * postStartPtr, int * patchStartPtr, int * adjustedDim) {
      float preInPostCoords; // The location, in postsynaptic restricted coordinates, of the presynaptic cell of this patch
      if (postNeuronsPerPreNeuron > 1) {
         preInPostCoords = zPre * postNeuronsPerPreNeuron + 0.5f*(postNeuronsPerPreNeuron-1);
      }
      else if (preNeuronsPerPostNeuron > 1) {
         preInPostCoords = ((float) (2*zPre-(preNeuronsPerPostNeuron-1)))/((float) 2*preNeuronsPerPostNeuron);
      }
      else {
         preInPostCoords = (float) zPre;
      }
      float postStartf = preInPostCoords - 0.5f*patchDim; // The location, in postsynaptic restricted coordinates of the start of an interval of length nxp and center xPreInPostCoords
      float postStopf = preInPostCoords + 0.5f*patchDim; // The location of the end of the interval starting at xPostStartf.
      // Everything between xPostStartf and xPostStopf, inclusive, is in the patch.
      int postStart = (int) ceil(postStartf);
      int postStop = (int) floor(postStopf) + 1;
      assert(postStop-postStart==patchDim);
      int patchStart = 0;
      int patchStop = patchDim;
      if (postStop<0) {
         postStop=0;
         postStart=0;
         patchStart=0;
         patchStop=0;
      }
      if (postStart<0) {
         patchStart += -postStart;
         postStart = 0;
      }
      if (postStart>nPost) {
         postStart=nPost;
         postStop=nPost;
         patchStart=0;
         patchStop=0;
      }
      if (postStop>nPost) {
         patchStop-=(postStop-nPost);
         postStop = nPost;
      }
      assert(postStop-postStart==patchStop-patchStart);
      // calculate width of the edge-adjusted patch and perform sanity checks.
      int width = patchStop - patchStart;
      assert(width>=0 && width<=patchDim && patchStart>=0 && patchStart+width<=patchDim);
      *postStartPtr = postStart;
      *patchStartPtr = patchStart;
      *adjustedDim = width;
      return PV_SUCCESS;
   }

}; // class HyPerConn

BaseObject * createHyPerConn(char const * name, HyPerCol * hc);
InitWeights * getWeightInitializer(char const * name, HyPerCol * hc);
NormalizeBase * getWeightNormalizer(char const * name, HyPerCol * hc);

} // namespace PV

#endif /* HYPERCONN_HPP_ */
