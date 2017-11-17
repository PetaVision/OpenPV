/*
 * HyPerConnection.hpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "BaseConnection.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/Random.hpp"
#include "components/HyPerDeliveryFacade.hpp"
#include "components/Weights.hpp"
#include "connections/accumulate_functions.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "layers/HyPerLayer.hpp"
#include "probes/BaseConnectionProbe.hpp"
#include "utils/Timer.hpp"
#include <map>
#include <set>
#include <stdlib.h>
#include <string>
#include <vector>

#ifdef PV_USE_CUDA
#include "arch/cuda/CudaBuffer.hpp"
#include "cudakernels/CudaRecvPost.hpp"
#include "cudakernels/CudaRecvPre.hpp"
#endif

#define PROTECTED_NUMBER 13
#define MAX_ARBOR_LIST (1 + MAX_NEIGHBORS)

namespace PV {

struct SparseWeightInfo {
   unsigned long size;
   float thresholdWeight;
   float percentile;
};

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

   enum AccumulateType { UNDEFINED, CONVOLVE, STOCHASTIC };
   // Subclasses that need different accumulate types should define their own enums

   HyPerConn(const char *name, HyPerCol *hc);

   virtual ~HyPerConn();

   virtual int insertProbe(BaseConnectionProbe *p) override;

   // preLayerData and postLayerData point to the data for pre and post over all batch elements
   // (batchID argument is used to navigate to the correct part of the buffers)
   int updateInd_dW(
         int arborID,
         int batchID,
         float const *preLayerData,
         float const *postLayerData,
         int kExt);

   virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime);
   void
   writeWeights(double timestamp, Weights *weights, bool compressWeights, FileStream *fileStream);
   void writeWeights(
         double timestamp,
         Weights *weights,
         bool compressWeights,
         std::string const &path,
         bool appendFlag,
         bool verifyWrites);
   virtual int writeTextWeights(const char *filename, bool verifyWrites, int k);

   virtual int writeTextWeightsExtra(PrintStream *pvstream, int k, int arborID) {
      return PV_SUCCESS;
   }

   // writePostSynapticWeights was removed Apr 28, 2017. Create a TransposeConn if needed.

   /**
    * Uses presynaptic layer's activity to modify the postsynaptic GSyn or thread_gSyn
    */
   virtual int deliver() override;
   virtual void deliverOnePreNeuronActivity(
         int patchIndex,
         int arbor,
         float a,
         float *postBufferStart,
         void *auxPtr);
   virtual void deliverOnePostNeuronActivity(
         int arborID,
         int kTargetExt,
         int inSy,
         float *activityStartBuf,
         float *gSynPatchPos,
         float dt_factor,
         taus_uint4 *rngPtr);

   AccumulateType getPvpatchAccumulateType() { return pvpatchAccumulateType; }
   int (*accumulateFunctionPointer)(
         int kPreRes,
         int nk,
         float *v,
         float a,
         float *w,
         void *auxPtr,
         int sf);
   int (*accumulateFunctionFromPostPointer)(
         int kPreRes,
         int nk,
         float *v,
         float *a,
         float *w,
         float dt_factor,
         void *auxPtr,
         int sf);

   double getWeightUpdatePeriod() { return weightUpdatePeriod; }
   double getWeightUpdateTime() { return weightUpdateTime; }

   /**
    * Returns the last time that weights were updated.
    */
   double getLastUpdateTime() { return lastUpdateTime; }

   /**
    * Returns the last time that the connection's updateState function was called.
    * Provided so that connections that depend on other connections (e.g. CopyConn)
    * can postpone their update until the connection they depend has processed its updateState call.
    */
   double getLastTimeUpdateCalled() { return lastTimeUpdateCalled; }

   inline InitWeights *getWeightInitializer() { return weightInitializer; }

   inline bool getSelfFlag() { return selfFlag; }

   inline bool usingSharedWeights() { return sharedWeights; }

   /** Actual mininum weight value */
   virtual float minWeight(int arborId = 0);

   /** Actual maximum weight value */
   virtual float maxWeight(int arborId = 0);

   /** Minimum allowed weight value */
   inline float getWMin() { return wMin; };

   /** Maximum allowed weight value */
   inline float getWMax() { return wMax; };

   inline float getDWMax() { return dWMax; }

   inline int xPatchSize() { return nxp; }

   inline int yPatchSize() { return nyp; }

   inline int fPatchSize() { return nfp; }

   inline int xPatchStride() { return sxp; }

   inline int yPatchStride() { return syp; }

   inline int fPatchStride() { return sfp; }

   inline int xPostPatchSize() { return nxpPost; }

   inline int yPostPatchSize() { return nypPost; }

   inline int fPostPatchSize() { return nfpPost; }

   Patch const *getPatch(int kPre) { return &mWeights->getPatch(kPre); }

   inline const PVPatchStrides *getPostExtStrides() { return &postExtStrides; }

   inline const PVPatchStrides *getPostNonextStrides() { return &postNonextStrides; }

   inline float *getWeightsDataStart(int arborId) { return mWeights->getData(arborId); }

   inline float *getWeightsDataHead(int arborId, int dataIndex) {
      return mWeights->getDataFromDataIndex(arborId, dataIndex);
   }

   inline float *getWeightsData(int arborId, int patchIndex) {
      return mWeights->getDataFromPatchIndex(arborId, patchIndex)
             + mWeights->getPatch(patchIndex).offset;
   }

   inline float *getDeltaWeightsDataStart(int arborId) { return mDeltaWeights->getData(arborId); }

   inline float *getDeltaWeightsDataHead(int arborId, int dataIndex) {
      return mDeltaWeights->getDataFromDataIndex(arborId, dataIndex);
   }

   inline float *getDeltaWeightsData(int arborId, int patchIndex) {
      return mDeltaWeights->getDataFromPatchIndex(arborId, patchIndex)
             + mDeltaWeights->getPatch(patchIndex).offset;
   }

   inline long *getActivations(int arborId) { return numKernelActivations[arborId]; }

   inline long *getActivations(int arborId, int patchIndex) {
      return &numKernelActivations[arborId][patchStartIndex(patchToDataLUT(patchIndex))
                                            + mWeights->getPatch(patchIndex).offset];
   }

   inline long *getActivationsHead(int arborId, int dataIndex) {
      return &numKernelActivations[arborId][patchStartIndex(dataIndex)];
   }

   int getNumGeometryPatches() { return mWeights->getGeometry()->getNumPatches(); }

   int getNumDataPatchesX() { return mWeights->getNumDataPatchesX(); }

   int getNumDataPatchesY() { return mWeights->getNumDataPatchesY(); }

   int getNumDataPatchesF() { return mWeights->getNumDataPatchesF(); }

   int getNumDataPatches() { return mWeights->getNumDataPatches(); }

   size_t const *getGSynPatchStart() const {
      return mWeights->getGeometry()->getGSynPatchStart().data();
   }

   size_t getGSynPatchStart(int kPre) { return mWeights->getGeometry()->getGSynPatchStart(kPre); }

   size_t const *getAPostOffset() const { return mWeights->getGeometry()->getAPostOffset().data(); }

   size_t getAPostOffset(int kPre) { return mWeights->getGeometry()->getAPostOffset(kPre); }

   NormalizeBase *getNormalizer() { return normalizer; }

   bool getNormalizeDwFlag() { return normalizeDwFlag; }

   inline bool getConvertRateToSpikeCount() const { return convertRateToSpikeCount; }

   // convertPreSynapticWeights was marked obsolete Jul 27, 2017.
   Patch ***convertPreSynapticWeights(double time);

   bool getUpdateGSynFromPostPerspective() { return updateGSynFromPostPerspective; }

   taus_uint4 *getRandState(int index);

   int sumWeights(
         int nx,
         int ny,
         int offset,
         float *dataStart,
         double *sum,
         double *sum2,
         float *maxVal);

   virtual void addClone(PlasticCloneConn *conn);
   virtual long *getPostToPreActivity() { return postToPreActivity; }

   virtual int patchToDataLUT(int patchIndex);
   virtual int *getPatchToDataLUT() { return patch2datalookuptable; }
   virtual int
   patchIndexToDataIndex(int patchIndex, int *kx = NULL, int *ky = NULL, int *kf = NULL);
   virtual int
   dataIndexToUnitCellIndex(int dataIndex, int *kx = NULL, int *ky = NULL, int *kf = NULL);

   /**
    * Sets the flag indicating that the postsynaptic perspective is needed.
    */
   void setNeedPost() { needPost = true; }

  private:
   Weights *mWeights      = nullptr; // Contains the connectivity graph and the weight values
   Weights *mDeltaWeights = nullptr; // Used by plastic weights to hold the weight updates

   PVPatchStrides
         postExtStrides; // sx,sy,sf for a patch mapping into an extended post-synaptic layer
   PVPatchStrides
         postNonextStrides; // sx,sy,sf for a patch mapping into a non-extended post-synaptic layer
   bool strengthParamHasBeenWritten;
   int *patch2datalookuptable;

   long *postToPreActivity;

   bool needPost; // needPost is set during the communicate stage.  During the allocate stage, the
   // value is used to decide whether to create postConn.

   // All weights that are above the threshold
   typedef float WeightType;
   typedef std::vector<WeightType> WeightListType;
   typedef std::vector<int> IndexListType;

   // Percentage of weights that are ignored. Weight values must be above this threshold
   // to be included in the calculation. Valid values are 0.0 - 1.0. But there's no
   // point in setting this to 1.0.
   float mWeightSparsity;

   // Commented out 11/3/16, seem to be unused?
   // The output offset into the post layer for a weight
   // std::vector<int> mSparsePost;
   // Start of sparse weight data in the _sparseWeight array, indexed by data patch
   // std::vector<int> mPatchSparseWeightIndex;
   // Number of sparse weights for a patch, indexed by data patch
   // std::vector<int> mPatchSparseWeightCount;
   // Have sparse weights been allocated for each arbor?
   std::vector<bool> mSparseWeightsAllocated;

   typedef std::map<const WeightType *const, const WeightListType> WeightPtrMapType;
   typedef std::map<const WeightType *const, const IndexListType> WeightIndexMapType;

   // Map nk -> weight ptr -> sparse weights
   typedef std::map<int, WeightPtrMapType> WeightMapType;
   // Map nk -> weight ptr -> output index
   typedef std::map<int, WeightIndexMapType> IndexMapType;

   WeightMapType mSparseWeightValues;
   IndexMapType mSparseWeightIndices;
   SparseWeightInfo mSparseWeightInfo;

   std::set<int> mKPreExtWeightSparsified;

   // unsigned long mNumDeliverCalls; // Number of times deliver has been called
   // unsigned long mAllocateSparseWeightsFrequency; // Number of mNumDeliverCalls that need to
   // happen
   // before the pre list needs to be rebuilt

   // Allocate sparse weights when performing presynaptic delivery
   void allocateSparseWeightsPre(PVLayerCube const *activity, int arbor);
   // Allocate sparse weights when performing postsynaptic delivers
   void allocateSparseWeightsPost(PVLayerCube const *activity, int arbor);
   // Calculates the sparse weight threshold
   SparseWeightInfo calculateSparseWeightInfo() const;
   SparseWeightInfo findPercentileThreshold(float percentile, Weights *weights) const;
   void deliverOnePreNeuronActivitySparseWeights(
         int kPreExt,
         int arbor,
         float a,
         float *postBufferStart,
         void *auxPtr);
   void deliverOnePostNeuronActivitySparseWeights(
         int arborID,
         int kTargetExt,
         int inSy,
         float *activityStartBuf,
         float *gSynPatchPos,
         float dt_factor,
         taus_uint4 *rngPtr);

  protected:
   HyPerConn *postConn;
   bool needFinalize;

   // Whether to check if pre-layer is spiking and, if it is not,
   // scale activity by dt to convert it to a spike count
   bool convertRateToSpikeCount;

   bool useMask;
   char *maskLayerName;
   int maskFeatureIdx;
   HyPerLayer *mask;
   bool *batchSkip;

   bool normalizeDwFlag;

   int nxp, nyp, nfp; // size of weight dimensions
   bool warnDefaultNfp; // Whether to print a warning if the default nfp is used.
   int sxp, syp, sfp; // stride in x,y,features

   int nxpPost, nypPost, nfpPost;
   int numParams;
   float wMax;
   float wMin;
   double initialWriteTime;
   double writeTime; // time of next output, initialized in params file parameter initialWriteTime
   double writeStep; // output time interval
   bool writeCompressedWeights; // if true, outputState writes weights with 8-bit precision; if
   // false, write weights with float precision
   bool writeCompressedCheckpoints; // similar to writeCompressedWeights, but for checkpointWrite
   // instead of outputState
   int fileType; // type ID for file written by PV::writeWeights

   Timer *io_timer;
   Timer *update_timer;

   bool sharedWeights;
   bool triggerFlag;
   char *triggerLayerName;
   double triggerOffset;
   HyPerLayer *triggerLayer;
   bool combine_dW_with_W_flag; // indicates that dwDataStart should be set equal to wDataStart,
   // useful for saving memory when weights are not being learned but
   // not used
   bool selfFlag; // indicates that connection is from a layer to itself (even though pre and post
   // may be separately instantiated)
   char *normalizeMethod;
   NormalizeBase *normalizer;

   // This object handles calculating weights.  All the initialize weights methods for all
   // connection classes
   // are being moved into subclasses of this object.  The default root InitWeights class will
   // create
   // 2D Gaussian weights.  If weight initialization type isn't created in a way supported by
   // Buildandrun,
   // this class will try to read the weights from a file or will do a 2D Gaussian.
   char *weightInitTypeString;
   InitWeights *weightInitializer;
   char *pvpatchAccumulateTypeString;
   AccumulateType pvpatchAccumulateType;
   bool normalizeTotalToPost; // if false, normalize the sum of weights from each presynaptic
   // neuron.  If true, normalize the sum of weights into a postsynaptic
   // neuron.
   float dWMax; // dW scale factor
   bool useListOfArborFiles;
   bool combineWeightFiles;
   bool updateGSynFromPostPerspective;

   float **thread_gSyn; // Accumulate buffer for each thread, only used if numThreads > 1 // Move
   // back to HyPerLayer?

   double weightUpdatePeriod;
   double weightUpdateTime;
   double initialWeightUpdateTime;
   double lastUpdateTime;
   double lastTimeUpdateCalled;
   bool mImmediateWeightUpdate = true;

   long **numKernelActivations;
   std::vector<MPI_Request> m_dWReduceRequests;
   bool mReductionPending = false;
   // mReductionPending is set by reduce_dW() and cleared by
   // blockingNormalize_dW(). We don't use the nonemptiness of
   // m_dWReduceRequests as the signal to blockingNormalize_dW because the
   // requests are not created if there is only a single MPI processes.

   Random *randState;

   int mDWMaxDecayInterval = 0; // How many weight updates between each dWMax modification
   int mDWMaxDecayTimer    = 0; // Number of updates left before next dWMax modification
   float mDWMaxDecayFactor = 0.0f; // Each modification is dWMax = dWMax * (1.0 - decayFactor);

   CheckpointableFileStream *mOutputStateStream = nullptr; // weights file written by outputState

   // A vector of plastic clones that are cloning from this connection
   std::vector<PlasticCloneConn *> clones;

  protected:
   HyPerConn();

   // Protected method to return the entire weights object.
   // There are public methods to retrieve pointers to individual patches and weight values.
   Weights *getWeights() { return mWeights; }

   // Protected method to set the entire weights object.
   void setWeights(Weights *weights) { mWeights = weights; }
   void setDeltaWeights(Weights *deltaWeights) { mDeltaWeights = deltaWeights; }

   Weights *getDeltaWeights() { return mDeltaWeights; }

   inline long **get_activations() { return numKernelActivations; }

   inline size_t patchStartIndex(int patchIndex) {
      return (size_t)patchIndex * (size_t)(nxp * nyp * nfp);
   }

   inline size_t postPatchStartIndex(int patchIndex) {
      return (size_t)patchIndex * (size_t)(nxpPost * nypPost * nfpPost);
   }

   int calcUnitCellIndex(
         int patchIndex,
         int *kxUnitCellIndex = NULL,
         int *kyUnitCellIndex = NULL,
         int *kfUnitCellIndex = NULL);
   virtual int setPatchStrides();
   int checkPatchDimensions();
   virtual int checkPatchSize(int patchSize, int scalePre, int scalePost, char dim);

   virtual void allocateWeights();
   virtual void initPatchToDataLUT();

   /**
    * Initializes the connection.  This routine should be called by the initialize method of classes
    * derived from HyPerConn.
    * It is not called by the default HyPerConn constructor.
    */
   int initialize(char const *name, HyPerCol *hc);
   void setWeightInitializer();
   int setWeightNormalizer();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters needed from the HyPerConn class
    * @name HyPerConn Parameters
    * @{
    */

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

   /**
    * @brief weightUpdatePeriod: If plasticity flag is set and there is no trigger layer,
    * specifies the update period of weights. This parameter is required.
    * To specify updating every time period, make the weightUpdatePeriod no bigger than
    * the parent HyPerCol's dt parameter.
    */
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);

   /**
    * @brief initialWeightUpdateTime: If plasticity flag is set, specifies the inital weight update
    * time; ignored if triggerFlag = true
    */
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: Specifies the layer to trigger weight updates
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerOffset: If trigger flag is set, triggers \<triggerOffset\> timesteps before
    * target trigger
    * @details Defaults to 0.
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);

   /**
    * @brief immediateWeightUpdate: This flag is read for plastic connections
    * with shared weights. If set to true, the change in weights is applied to
    * the weights immediately at the end of the weight update period. If set
    * to false, the change in weights is applied at the end of the next weight
    * update, to allow for concurrent reduction in the shared weights.
    */
   virtual void ioParam_immediateWeightUpdate(enum ParamsIOFlag ioFlag);

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
    * @brief writeCompressedWeights: If writeStep >= 0, weights written out are bytes as opposed to
    * floats.
    */
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeCompressedCheckpoints: Checkpoint weights are written compressed.
    * @details The parent HyPerCol must be writing checkpoints for this flag to be used
    */
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);

   /**
    * @brief selfFlag: Indicates if pre and post is the same layer.
    * @details The default value for selfFlag should be pre==post, but at the time
    * ioParams(PARAMS_IO_READ) is called,
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
    * @brief nfp: Specifies the post feature patch size
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);

   /**
    * @brief shrinkPatches: This parameter is no longer used
    */
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag);

   /**
    * @brief updateGSynFromPostPerspective: Specifies if the connection should push from pre or pull
    * from post.
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
    * @brief normalizeDw: Specifies if this connection is averaging gradients (true) or summing them
    * (false)
    */
   virtual void ioParam_normalizeDw(enum ParamsIOFlag ioFlag);

   /**
    * @brief convertRateToSpikeCount: If true, presynaptic activity should be converted from a rate
    * to a count.
    * @details If this flag is true and the presynaptic layer is not spiking, the activity will be
    * interpreted
    * as a spike rate, and will be converted to a spike count when delivering activity to the
    * postsynaptic GSyn buffer.
    * If this flag is false, activity will not be converted.
    */
   virtual void ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag);

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

#ifdef PV_USE_CUDA
   /**
    * @brief gpuGroupIdx: All connections in the same group uses the same GPU memory for weights
    * @details Specify a group index. An index of -1 means no group (default).
    * This parameter is ignored if PetaVision was compiled without GPU acceleration.
    */
   virtual void ioParam_gpuGroupIdx(enum ParamsIOFlag ioFlag);
#endif // PV_USE_CUDA

   /**
    * @brief weightSparsity: Specifies what percentage of weights will be ignored. Default is 0.0
    */
   virtual void ioParam_weightSparsity(enum ParamsIOFlag ioFlag);

   virtual void ioParam_dWMaxDecayInterval(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWMaxDecayFactor(enum ParamsIOFlag ioFlag);
   /** @} */

   virtual void createDeliveryObject() override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   virtual int registerData(Checkpointer *checkpointer) override;

   /**
    * Called by registerData. If writeStep is nonnegative, opens the weights pvp file to be
    * used by outputState, and registers its file position with the checkpointer.
    */
   void openOutputStateFile(Checkpointer *checkpointer);
   void registerTimers(Checkpointer *checkpointer);
   virtual int finalizeUpdate(double timed, double dt) override;

   virtual int setPatchSize(); // Sets nxp, nyp, nfp if weights are loaded from file.  Subclasses
   // override if they have specialized ways of setting patch size that
   // needs to go in the communicate stage.
   virtual int setPostPatchSize(); // Sets nxp, nyp, nfp if weights are loaded from file.
   // Subclasses override if they have specialized ways of setting
   // patch size that needs to go in the communicate stage.
   // (e.g. BIDSConn uses pre and post layer size to set nxp,nyp, but pre and post aren't set until
   // communicateInitInfo().
   virtual void
   handleDefaultSelfFlag(); // If selfFlag was not set in params, set it in this function.
   virtual float *allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch);
   virtual int allocatePostToPreBuffer();
   virtual int allocatePostConn();

   int clearWeights(float **dataStart, int numPatches, int nx, int ny, int nf);
   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override;
   void checkpointWeightPvp(Checkpointer *checkpointer, char const *bufferName, Weights *weights);
   virtual int setInitialValues() override; // returns PV_SUCCESS if successful,
   // or PV_POSTPONE if it needs to wait on other objects
   // (e.g. TransposeConn has to wait for original conn)

   /**
    * Calls blockingNormalize_dW.
    */
   virtual int prepareCheckpointWrite() override;

   virtual void writeWeights(double timestamp);

   void updateWeightsImmediate(double simTime, double dt);
   void updateWeightsDelayed(double simTime, double dt);

   /**
    * updateLocal_dW computes the contribution of the current process to dW,
    * before MPI reduction and normalization. The routine calls initialize_dW
    * for each arbor, and then updateWeights for each arbor.
    */
   void updateLocal_dW();

   virtual int outputState(double time) override;
   virtual int updateState(double time, double dt) override;
   virtual bool needUpdate(double time, double dt) override;

   /**
    * Initializes dW. Default behaviour is to clear dW.
    */
   virtual int initialize_dW(int arborId);
   virtual int clear_dW(int arborId);
   virtual int clear_numActivations(int arborId);
   /**
    * Updates the dW buffer
    */
   virtual int update_dW(int arborID);
   virtual float updateRule_dW(float pre, float post);

   /**
    * Updates the weights in all arbors, by calling updateWeights(int) for each arbor index.
    */
   void updateArbors();

   /**
    * Updates the weights in one arbor. HyPerConn updates the weights using
    * W_new = W_old + dW.
    */
   virtual int updateWeights(int arborId = 0);

   /**
    * Decrements the counter for dWMaxDecayInterval, and if at the end of the interval,
    * decays the dWMax value.
    */
   void decay_dWMax();

   virtual bool skipPre(float preact) { return preact == 0.0f; };
   /**
    * Reduces dW and activations for all arbors across MPI
    */
   void reduce_dW();

   /**
    * Reduces dW and activations for one arbor across MPI
    */
   virtual int reduce_dW(int arborId);
   virtual int reduceKernels(int arborID);
   virtual int reduceActivations(int arborID);

   /**
    * Used when batch size is greater than one, plasticityFlag is on, and shared weights are off.
    * Sums the dW across all elements in the batch (both local and MPI).
    */
   void reduceAcrossBatch(int arborID);

   /**
    * If there are outstanding MPI requests for reducing dW, wait for them to complete and
    * then call normalize_dW(). This function is called in two situations: when writing a
    * checkpoint * (so that we don't have to separately checkpoint the number of activation
    * kernels) and in timesteps where the weights need to be updated (and now need the
    * calculation of dW to be complete).
    */
   void blockingNormalize_dW();

   /**
    * Normalizes dW for all arbors, by calling normalize_dW(int) for each arbor index.
    */
   void normalize_dW();

   /**
    * Normalizes dW for one arbor.
    * HyPerConn normalizes dW by dividing by the number of activations.
    */
   virtual int normalize_dW(int arbor_ID);

   virtual int deliverPresynapticPerspectiveConvolve(PVLayerCube const *activity, int arborID);
   virtual int deliverPresynapticPerspectiveStochastic(PVLayerCube const *activity, int arborID);
   virtual int deliverPresynapticPerspective(PVLayerCube const *activity, int arborId) {
      int status = PV_SUCCESS;
      switch (pvpatchAccumulateType) {
         case CONVOLVE: status = deliverPresynapticPerspectiveConvolve(activity, arborId); break;
         case STOCHASTIC:
            status = deliverPresynapticPerspectiveStochastic(activity, arborId);
            break;
         default:
            pvAssert(0);
            status = PV_FAILURE;
            break;
      }
      return status;
   }
   virtual int deliverPostsynapticPerspective(PVLayerCube const *activity, int arborID) {
      int status = PV_SUCCESS;
      switch (pvpatchAccumulateType) {
         case CONVOLVE:
            status = deliverPostsynapticPerspectiveConvolve(activity, arborID, NULL, NULL);
            break;
         case STOCHASTIC:
            status = deliverPostsynapticPerspectiveStochastic(activity, arborID, NULL, NULL);
            break;
         default:
            pvAssert(0);
            status = PV_FAILURE;
            break;
      }
      return status;
   }
   virtual int deliverPostsynapticPerspectiveConvolve(
         PVLayerCube const *activity,
         int arborID,
         int *numActive,
         int **activeList);
   virtual int deliverPostsynapticPerspectiveStochastic(
         PVLayerCube const *activity,
         int arborID,
         int *numActive,
         int **activeList);
#ifdef PV_USE_CUDA
   virtual int deliverPresynapticPerspectiveGPU(PVLayerCube const *activity, int arborID);
   virtual int deliverPostsynapticPerspectiveGPU(PVLayerCube const *activity, int arborID);
#endif // PV_USE_CUDA

   double getConvertToRateDeltaTimeFactor();

   virtual int cleanup() override;
   void wait_dWReduceRequests();

// GPU variables
#ifdef PV_USE_CUDA
  public:
   bool getAllocDeviceWeights() { return allocDeviceWeights; }
   bool getAllocPostDeviceWeights() { return allocPostDeviceWeights; }
   int getGpuGroupIdx() const { return mGpuGroupIdx; }
   HyPerConn *getGpuGroupHead() const { return mGpuGroupHead; }

   virtual void setAllocDeviceWeights() { allocDeviceWeights = true; }
   virtual void setAllocPostDeviceWeights() { allocPostDeviceWeights = true; }
   virtual PVCuda::CudaBuffer *getDeviceWData() { return d_WData; }
   PVCuda::CudaBuffer *getDevicePatches() { return d_Patches; }
   PVCuda::CudaBuffer *getDeviceGSynPatchStart() { return d_GSynPatchStart; }
   void setDeviceWData(PVCuda::CudaBuffer *inBuf) { d_WData = inBuf; }

#ifdef PV_USE_CUDNN
   virtual PVCuda::CudaBuffer *getCudnnWData() { return cudnn_WData; }
   void setCudnnWData(PVCuda::CudaBuffer *inBuf) { cudnn_WData = inBuf; }
#endif // PV_USE_CUDNN

   PVCuda::CudaRecvPost *getKrRecvPost() { return krRecvPost; }
   PVCuda::CudaRecvPre *getKrRecvPre() { return krRecvPre; }

  protected:
   virtual int allocatePostDeviceWeights();
   virtual int allocateDeviceWeights();
   virtual int allocateDeviceBuffers();
   virtual int initializeReceivePostKernelArgs();
   virtual int initializeReceivePreKernelArgs();
   virtual void updateDeviceWeights();

   bool allocDeviceWeights;
   bool allocPostDeviceWeights;

   PVCuda::CudaBuffer *d_WData;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *cudnn_WData;
#endif // PV_USE_CUDNN
   PVCuda::CudaBuffer *d_Patches;
   PVCuda::CudaBuffer *d_GSynPatchStart;
   PVCuda::CudaBuffer *d_PostToPreActivity;
   PVCuda::CudaBuffer *d_Patch2DataLookupTable;
   PVCuda::CudaRecvPost *krRecvPost; // Cuda kernel for update state call
   PVCuda::CudaRecvPre *krRecvPre; // Cuda kernel for update state call
   int mGpuGroupIdx         = -1;
   HyPerConn *mGpuGroupHead = nullptr;

#endif // PV_USE_CUDA

  private:
   int initialize_base();
   int clearWeights(float *arborDataStart, int numPatches, int nx, int ny, int nf);
   int deleteWeights();
   void unsetAccumulateType();

}; // class HyPerConn

} // namespace PV

#endif /* HYPERCONN_HPP_ */
