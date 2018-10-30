/*
 * HebbianUpdater.hpp
 *
 *  Created on: Nov 29, 2017
 *      Author: Pete Schultz
 */

#ifndef HEBBIANUPDATER_HPP_
#define HEBBIANUPDATER_HPP_

#include "components/Weights.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class HebbianUpdater : public BaseWeightUpdater {
  protected:
   /**
    * List of parameters needed from the HebbianUpdater class
    * @name HebbianUpdater Parameters
    * @{
    */

   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_immediateWeightUpdate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWMaxDecayInterval(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dWMaxDecayFactor(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeDw(enum ParamsIOFlag ioFlag);
   virtual void ioParam_useMask(enum ParamsIOFlag ioFlag);
   virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag);

   /** @} */ // end of HebbianUpdater parameters

  public:
   HebbianUpdater(char const *name, HyPerCol *hc);

   virtual ~HebbianUpdater();

   void addClone(ConnectionData *connectionData);

   float const *getDeltaWeightsDataStart(int arborId) const {
      return mDeltaWeights->getData(arborId);
   }

   float const *getDeltaWeightsDataHead(int arborId, int dataIndex) const {
      return mDeltaWeights->getDataFromDataIndex(arborId, dataIndex);
   }

  protected:
   HebbianUpdater() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   virtual Response::Status prepareCheckpointWrite() override;

   virtual void updateState(double timestamp, double dt) override;

   virtual bool needUpdate(double time, double dt);

   void updateWeightsImmediate(double simTime, double dt);
   void updateWeightsDelayed(double simTime, double dt);

   /**
    * updateLocal_dW computes the contribution of the current process to dW,
    * before MPI reduction and normalization. The routine calls initialize_dW
    * for each arbor, and then updateWeights for each arbor.
    */
   void updateLocal_dW();

   int initialize_dW(int arborId);

   int clear_dW(int arborId);

   int clearNumActivations(int arborId);

   int update_dW(int arborID);

   void updateInd_dW(
         int arborID,
         int batchID,
         float const *preLayerData,
         float const *postLayerData,
         int kExt);

   virtual float updateRule_dW(float pre, float post);

   void reduce_dW();

   virtual int reduce_dW(int arborId);

   virtual int reduceKernels(int arborID);

   virtual int reduceActivations(int arborID);

   void reduceAcrossBatch(int arborID);

   void blockingNormalize_dW();

   void wait_dWReduceRequests();

   virtual void normalize_dW();

   virtual int normalize_dW(int arbor_ID);

   void updateArbors();

   virtual int updateWeights(int arborId);

   /**
    * Decrements the counter for dWMaxDecayInterval, and if at the end of the interval,
    * decays the dWMax value.
    */
   void decay_dWMax();

   virtual void computeNewWeightUpdateTime(double time, double currentUpdateTime);

   virtual Response::Status cleanup() override;

  protected:
   char *mTriggerLayerName         = nullptr;
   double mTriggerOffset           = 0.0;
   double mWeightUpdatePeriod      = 0.0;
   double mInitialWeightUpdateTime = 0.0;
   bool mImmediateWeightUpdate     = true;

   // dWMax is required if plasticityFlag is true
   float mDWMax                       = std::numeric_limits<float>::quiet_NaN();
   int mDWMaxDecayFactor              = 0;
   float mDWMaxDecayInterval          = 0.0f;
   bool mNormalizeDw                  = true;
   bool mCombine_dWWithWFlag          = false;
   bool mWriteCompressedCheckpoints   = false;
   bool mInitializeFromCheckpointFlag = false;

   Weights *mWeights            = nullptr;
   Weights *mDeltaWeights       = nullptr;
   HyPerLayer *mTriggerLayer    = nullptr;
   bool mTriggerFlag            = false;
   double mWeightUpdateTime     = 0.0;
   double mLastUpdateTime       = 0.0;
   bool mNeedFinalize           = true;
   double mLastTimeUpdateCalled = 0.0;
   int mDWMaxDecayTimer         = 0;
   long **mNumKernelActivations = nullptr;
   std::vector<MPI_Request> mDeltaWeightsReduceRequests;
   bool mReductionPending = false;
   // mReductionPending is set by reduce_dW() and cleared by
   // blockingNormalize_dW(). We don't use the nonemptiness of
   // m_dWReduceRequests as the signal to blockingNormalize_dW because the
   // requests are not created if there is only a single MPI processes.
   std::vector<ConnectionData *> mClones;
};

} // namespace PV

#endif // HEBBIANUPDATER_HPP_
