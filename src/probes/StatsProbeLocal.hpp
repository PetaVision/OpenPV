#ifndef STATSPROBELOCAL_HPP_
#define STATSPROBELOCAL_HPP_

#include "StatsProbeTypes.hpp"

#include "structures/PVLayerLoc.hpp"
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
   void calculateStats(double simTime, ProbeData<LayerStats> &values) const;

   void calculateActivityValues(LayerStats &stats, int localBatchIndex) const;
   void calculateInternalStateValues(LayerStats &stats, int localBatchIndex) const;

   PVLayerCube findDataCube() const;
   float const *findVBuffer() const;

  private:
   std::shared_ptr<BufferParamInterface> mBufferParam = nullptr;
   float mNnzThreshold;
   ProbeDataBuffer<LayerStats> mStoredValues;
   HyPerLayer *mTargetLayer = nullptr;
};

template <typename T>
void StatsProbeLocal::setBufferParam(char const *objname, PVParams *params) {
   mBufferParam = std::make_shared<T>(objname, params);
}

} // namespace PV

#endif // STATSPROBELOCAL_HPP_
