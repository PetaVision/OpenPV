/*
 * TestNotAlwaysAllZerosProbe.cpp
 *
 * A subclass of StatsProbe that verifies a layer takes a nonzero value at some point in time.
 * Once the target layer gets a nonzero value, it sets an internal flag to true.  The public
 * function member nonzeroValueHasOccurred() tells whether this has happened.
 * Typical use is to check this value after the run has completed but before the HyPerCol is
 * deleted. It is useful for preventing the test from mistakenly passing, because two layers that
 * should always be equal are only equal because a bug makes them each always zero.
 *
 *  Created on: Apr 2, 2015
 *      Author: pschultz
 */

#include "TestNotAlwaysAllZerosProbe.hpp"
#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>

#include <memory>

namespace PV {

TestNotAlwaysAllZerosProbe::TestNotAlwaysAllZerosProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void TestNotAlwaysAllZerosProbe::checkStats() {
   auto const &storedValues           = mProbeAggregator->getStoredValues();
   auto numTimestamps                 = storedValues.size();
   int lastTimestampIndex             = static_cast<int>(numTimestamps) - 1;
   ProbeData<LayerStats> const &stats = storedValues.getData(lastTimestampIndex);
   int nbatch                         = static_cast<int>(stats.size());
   for (int b = 0; b < nbatch; b++) {
      LayerStats const &statsElem = stats.getValue(b);
      if (statsElem.mNumNonzero != 0) {
         mNonzeroValueOccurred = true;
      }
   }
}

void TestNotAlwaysAllZerosProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(params, defaults);
}

void TestNotAlwaysAllZerosProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(params, defaults, comm);
}

}; // namespace PV
