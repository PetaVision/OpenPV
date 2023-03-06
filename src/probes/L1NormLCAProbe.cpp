/*
 * L1NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L1NormLCAProbe.hpp"
#include "probes/VThreshEnergyProbeComponent.hpp"

namespace PV {

L1NormLCAProbe::L1NormLCAProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void L1NormLCAProbe::createEnergyProbeComponent(char const *name, PVParams *params) {
   mEnergyProbeComponent = std::make_shared<VThreshEnergyProbeComponent>(name, params);
}

void L1NormLCAProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   L1NormProbe::initialize(name, params, comm);
}

} /* namespace PV */
