#ifndef RESETSTATEONTRIGGERTESTPROBELOCAL_HPP_
#define RESETSTATEONTRIGGERTESTPROBELOCAL_HPP_

#include "structures/PVLayerLoc.hpp"
#include "params/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"

using namespace PV;

class ResetStateOnTriggerTestProbeLocal : public ProbeComponent {
  public:
   ResetStateOnTriggerTestProbeLocal(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~ResetStateOnTriggerTestProbeLocal() {}

   void clearStoredValues();
   void initializeState(HyPerLayer *targetLayer);
   void ioParamsFillGroup(ParamsIOSwitch ioSwitch) {}
   void storeValues(double simTime);

   PVLayerLoc const *getLayerLoc() const { return mTargetLayer->getLayerLoc(); }
   ProbeDataBuffer<int> const &getStoredValues() const { return mStoredValues; }

  protected:
   ResetStateOnTriggerTestProbeLocal() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO);

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
