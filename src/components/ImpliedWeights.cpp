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
   FatalIf(
         preLoc->bcast,
         "ImpliedWeights \"%s\" cannot have a broadcast pre-layer\n",
         name.c_str());
   FatalIf(
         postLoc->bcast,
         "ImpliedWeights \"%s\" cannot have a broadcast post-layer\n",
         name.c_str());
   setName(name);
   int const numArbors       = 1;
   bool const sharedWgtsFlag = true;
   Weights::initialize(
         patchSizeX, patchSizeY, patchSizeF, preLoc, postLoc, numArbors, sharedWgtsFlag, timestamp);
}

void ImpliedWeights::initNumDataPatches() { setNumDataPatches(0, 0, 0); }

} // end namespace PV
