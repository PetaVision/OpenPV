#include "ResetStateOnTriggerTestProbeLocal.hpp"
#include <components/BasePublisherComponent.hpp>
#include <structures/PVLayerLoc.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <utils/conversions.hpp>

#include <vector>

ResetStateOnTriggerTestProbeLocal::ResetStateOnTriggerTestProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

void ResetStateOnTriggerTestProbeLocal::countDiscrepancies(ProbeData<int> &values) const {
   auto *loc  = getLayerLoc();
   int nbatch = loc->nbatch;
   pvAssert(nbatch == static_cast<int>(values.size()));
   double timestamp = values.getTimestamp();
   if (values.getTimestamp() > 0.0) {
      long N           = (long)loc->nx * (long)loc->ny * (long)loc->nf;
      FatalIf(
            static_cast<long>(static_cast<int>(N)) != N,
            "Probe \"%s\" target layer \"%s\" is too large (%d-by-%d-by-%d)\n",
            getName().c_str(), mTargetLayer->getName(), loc->nx, loc->ny, loc->nf);
      long NGlobal     = (long)loc->nxGlobal * (long)loc->nyGlobal * (long)loc->nf;
      int nxExt        = loc->nx + loc->halo.lt + loc->halo.rt;
      int nyExt        = loc->nx + loc->halo.dn + loc->halo.up;
      long numExtended = (long)nxExt * (long)nyExt * (long)loc->nf;
      int inttime      = static_cast<int>(timestamp);
      for (int b = 0; b < nbatch; ++b) {
         int numDiscrepancies  = 0;
         float const *activity = mTargetLayerData + b * numExtended;
         for (int k = 0; k < (int)N; k++) {
            long kex          = calcExtendedIndex(k, loc);
            float a           = activity[kex];
            long kGlobal      = PV::globalIndexFromLocal(k, *loc);
            long correctValue = 4 * (kGlobal + 1) * ((inttime + 4) % 5 + 1)
                               + (kGlobal == ((((inttime - 1) / 5) * 5) + 1) % NGlobal);
            if (a != (float)correctValue) {
               ++numDiscrepancies;
            }
         }
         values.getValue(b) = numDiscrepancies;
      }
   }
   else {
      for (int b = 0; b < nbatch; ++b) {
         values.getValue(b) = 0;
      }
   }
}

void ResetStateOnTriggerTestProbeLocal::clearStoredValues() { mStoredValues.clear(); }

void ResetStateOnTriggerTestProbeLocal::initialize(char const *objName, PVParams *params) {
   ProbeComponent::initialize(objName, params);
}

void ResetStateOnTriggerTestProbeLocal::initializeState(BaseLayer *targetLayer) {
   mTargetLayer          = targetLayer;
   auto *targetPublisher = targetLayer->getComponentByType<PV::BasePublisherComponent>();
   FatalIf(
         targetPublisher == nullptr,
         "Probe %s could not find layer data for target layer \"%s\".\n",
         getName_c(),
         mTargetLayer->getName());
   mTargetLayerData = targetPublisher->getLayerData();
}

void ResetStateOnTriggerTestProbeLocal::storeValues(double simTime) {
   mStoredValues.store(ProbeData<int>(simTime, getLayerLoc()->nbatch));
   auto &discrepancies = mStoredValues.getBuffer().back();
   countDiscrepancies(discrepancies);
}

long ResetStateOnTriggerTestProbeLocal::calcExtendedIndex(long k, PVLayerLoc const *loc) {
   long kExt = PV::kIndexExtended(
         k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   return kExt;
}
