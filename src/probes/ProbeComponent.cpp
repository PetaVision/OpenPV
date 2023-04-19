#include "ProbeComponent.hpp"

namespace PV {

ProbeComponent::ProbeComponent(char const *objName, PVParams *params) {
   initialize(objName, params);
}

ProbeComponent::ProbeComponent() {}

void ProbeComponent::initialize(char const *objName, PVParams *params) {
   mName   = objName;
   mParams = params;
}

} // namespace PV
