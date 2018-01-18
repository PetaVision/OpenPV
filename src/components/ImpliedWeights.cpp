/*
 * ImpliedWeights.cpp
 *
 *  Created on: Jul 28, 2017
 *      Author: Pete Schultz
 */

#include "ImpliedWeights.hpp"

namespace PV {

ImpliedWeights::ImpliedWeights(std::string const &name) { setName(name); }

ImpliedWeights::ImpliedWeights(
      std::string const &name,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      double timestamp) {
   setName(name);
   int const numArbors      = 1;
   bool const sharedWeights = true;
   Weights::initialize(
         patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc, numArbors, sharedWeights, timestamp);
}

void ImpliedWeights::initNumDataPatches() { setNumDataPatches(0, 0, 0); }

} // end namespace PV
