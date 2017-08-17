/*
 * ImpliedWeights.cpp
 *
 *  Created on: Jul 28, 2017
 *      Author: Pete Schultz
 */

#include "ImpliedWeights.hpp"

namespace PV {

ImpliedWeights::ImpliedWeights(
      std::string const &name,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   auto geometry =
         std::make_shared<PatchGeometry>(name, patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc);
   initialize(name, geometry, numArbors, sharedWeights, timestamp);
}

ImpliedWeights::ImpliedWeights(
      std::string const &name,
      std::shared_ptr<PatchGeometry> geometry,
      int numArbors,
      bool sharedWeights,
      double timestamp) {
   initialize(name, geometry, numArbors, sharedWeights, timestamp);
}

ImpliedWeights::ImpliedWeights(std::string const &name, Weights const *baseWeights) {
   initialize(
         name,
         baseWeights->getGeometry(),
         baseWeights->getNumArbors(),
         baseWeights->getSharedFlag(),
         baseWeights->getTimestamp());
}

void ImpliedWeights::initNumDataPatches() { setNumDataPatches(0, 0, 0); }

} // end namespace PV
