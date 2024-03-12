#ifndef STATSPROBELOCAL_HPP_
#define STATSPROBELOCAL_HPP_

#include "StatsProbeTypes.hpp"

#include "include/PVLayerLoc.hpp"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "probes/BufferParamInterface.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"

#include <cfloat>
#include <cstdlib>
#include <memory>
#include <string>

namespace PV {

class StatsProbeLocal : public ProbeComponent {
  protected:
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag);

  public:
   StatsProbeLocal(char const *objName, PVParams *params);
   virtual ~StatsProbeLocal() {}

   void clearStoredValues();
   void initializeState(HyPerLayer *targetLayer);
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void storeValues(double simTime);

   StatsBufferType getBufferType() const { return mBufferParam->getBufferType(); }
   PVLayerLoc const *getLayerLoc() const { return mTargetLayer->getLayerLoc(); }
   float getNnzThreshold() const { return mNnzThreshold; }
   ProbeDataBuffer<LayerStats> const &getStoredValues() const { return mStoredValues; }

  protected:
   StatsProbeLocal() {}
   void initialize(char const *objName, PVParams *params);

   /**
    * Sets the BufferParam data member, based on the indicated typename.
    * The typename T must be a class derived from BufferParamInterface, and
    * have a constructor that takes (char const *objName, PVParams *params)
    * as arguments.
    */
   template <typename T>
   void setBufferParam(char const *objName, PVParams *params);

  private:
   template <StatsBufferType bufferType>
   float const *calculateBatchElementStart(int localBatchIndex) const;

   template <StatsBufferType bufferType>
   int calculateOffset(int k) const;

   void calculateStats(double simTime, ProbeData<LayerStats> &values) const;

   template <StatsBufferType bufferType>
   void calculateValues(LayerStats &stats, int localBatchIndex) const;

   float const *findDataBufferA() const;
   float const *findDataBufferV() const;

  private:
   std::shared_ptr<BufferParamInterface> mBufferParam = nullptr;
   float mNnzThreshold;
   ProbeDataBuffer<LayerStats> mStoredValues;
   HyPerLayer *mTargetLayer = nullptr;
};

template <StatsBufferType bufferType>
void StatsProbeLocal::calculateValues(LayerStats &stats, int localBatchIndex) const {
   PVLayerLoc const *loc = getLayerLoc();
   int numNeurons        = loc->nx * loc->ny * loc->nf;
   float const *data     = calculateBatchElementStart<bufferType>(localBatchIndex);
   double sum            = 0.0;
   double sumSquared     = 0.0;
   float min             = FLT_MAX;
   float max             = -FLT_MAX;
   int numNonzero        = 0;
   for (int k = 0; k < numNeurons; k++) {
      int kOffset       = calculateOffset<bufferType>(k);
      float const a     = data[kOffset];
      double const aDbl = static_cast<double>(a);
      sum += aDbl;
      sumSquared += aDbl * aDbl;
      numNonzero += (std::abs(a) > mNnzThreshold) ? 1 : 0;
      min = (a < min) ? a : min;
      max = (a > max) ? a : max;
   }
   stats.mSum        = sum;
   stats.mSumSquared = sumSquared;
   stats.mMin        = min;
   stats.mMax        = max;
   stats.mNumNeurons = numNeurons;
   stats.mNumNonzero = numNonzero;
}

template <typename T>
void StatsProbeLocal::setBufferParam(char const *objname, PVParams *params) {
   mBufferParam = std::make_shared<T>(objname, params);
}

} // namespace PV

#endif // STATSPROBELOCAL_HPP_
