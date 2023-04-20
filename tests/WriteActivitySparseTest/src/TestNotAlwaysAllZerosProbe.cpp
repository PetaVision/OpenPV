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
#include <io/PVParams.hpp>
#include <probes/ActivityBufferStatsProbeLocal.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeImmediate.hpp>
#include <probes/StatsProbeTypes.hpp>

#include <memory>

namespace PV {

TestNotAlwaysAllZerosProbe::TestNotAlwaysAllZerosProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
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

void TestNotAlwaysAllZerosProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<ActivityBufferStatsProbeLocal>(name, params);
}

void TestNotAlwaysAllZerosProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeImmediate::initialize(name, params, comm);
}

}; // namespace PV
