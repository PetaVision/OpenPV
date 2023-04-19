/*
 * L0NormLCAProbe.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#include "L0NormLCAProbe.hpp"
#include "probes/L0NormLCAEnergyProbeComponent.hpp"

namespace PV {

L0NormLCAProbe::L0NormLCAProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void L0NormLCAProbe::createEnergyProbeComponent(char const *name, PVParams *params) {
   mEnergyProbeComponent = std::make_shared<L0NormLCAEnergyProbeComponent>(name, params);
}

void L0NormLCAProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   L0NormProbe::initialize(name, params, comm);
}

} /* namespace PV */
