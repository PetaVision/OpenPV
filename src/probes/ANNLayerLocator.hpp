#ifndef ANNLAYERLOCATOR_HPP_
#define ANNLAYERLOCATOR_HPP_

#include "TargetLayerComponent.hpp"
#include "components/ANNActivityBuffer.hpp"

namespace PV {

/**
 * locateANNActivityBuffer(targetLayerComponent)
 * finds the ANNActivityBuffer of the layer pointed to by the targetLayerComponent,
 * and returns a const pointer to that buffer.
 * If any step of finding that buffer fails, nullptr is returned.
 */
ANNActivityBuffer const *
locateANNActivityBuffer(std::shared_ptr<TargetLayerComponent> targetLayerComponent);

} // namespace PV
#endif // ANNLAYERLOCATOR_HPP_
