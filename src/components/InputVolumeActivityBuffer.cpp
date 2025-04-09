/*
 * InputVolumeActivityBuffer.cpp
 */

#include "InputVolumeActivityBuffer.hpp"

namespace PV {

InputVolumeActivityBuffer::InputVolumeActivityBuffer(
      char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void InputVolumeActivityBuffer::initialize(
      char const *name, PVParams *params, Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void InputVolumeActivityBuffer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "displayPeriod", &mDisplayPeriod, mDisplayPeriod);
}

void InputVolumeActivityBuffer::setObjectType() { mObjectType = "InputVolumeActivityBuffer"; }

} // namespace PV
