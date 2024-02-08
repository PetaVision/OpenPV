#ifndef RESETSTATEONTRIGGERTESTPROBELOCAL_HPP_
#define RESETSTATEONTRIGGERTESTPROBELOCAL_HPP_

#include "include/PVLayerLoc.hpp"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"

using PV::HyPerLayer;
using PV::ParamsIOFlag;
using PV::ProbeComponent;
using PV::ProbeData;
using PV::ProbeDataBuffer;
using PV::PVParams;

class ResetStateOnTriggerTestProbeLocal : public ProbeComponent {
  public:
   ResetStateOnTriggerTestProbeLocal(char const *objName, PVParams *params);
   virtual ~ResetStateOnTriggerTestProbeLocal() {}

   void clearStoredValues();
   void initializeState(HyPerLayer *targetLayer);
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag) {}
   void storeValues(double simTime);

   PVLayerLoc const *getLayerLoc() const { return mTargetLayer->getLayerLoc(); }
   ProbeDataBuffer<int> const &getStoredValues() const { return mStoredValues; }

  protected:
   ResetStateOnTriggerTestProbeLocal() {}
   void initialize(char const *objName, PVParams *params);

  private:
   static int calcExtendedIndex(int k, PVLayerLoc const *loc);

   void countDiscrepancies(ProbeData<int> &values) const;

  private:
   ProbeDataBuffer<int> mDiscrepancies;
   ProbeDataBuffer<int> mStoredValues;
   HyPerLayer *mTargetLayer      = nullptr;
   float const *mTargetLayerData = nullptr;
   ;
};

#endif // RESETSTATEONTRIGGERTESTPROBELOCAL_HPP_
