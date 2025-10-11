#include "requiredConvolveMargin.hpp"

#include "utils/PVLog.hpp"

#include <cassert>
#include <cmath>

namespace PV {

int requiredConvolveMargin(int nPre, int nPost, int patchSize, char axis, char const *objectName) {
   int margin = 0;
   if (nPre == nPost) {
      FatalIf(
            patchSize % 2 != 1,
            "%s has one-to-one weights, which requires odd patch size, but n%cp = %d\n",
            objectName, axis, patchSize);
      margin = (patchSize - 1) / 2;
   }
   else if (nPre > nPost) { // many-to-one
      FatalIf(
            nPre % nPost != 0,
            "%s has many-to-one weights, which requires presynaptic dimension to be a "
            "power of two times the postsynaptic dimension (pre n%c = %d; post n%c = %d)\n",
            objectName, axis, nPre, axis, nPost);
      int densityRatio = nPre / nPost;
      double log2densityRatio = std::log2(densityRatio);
      FatalIf(
            log2densityRatio != std::round(log2densityRatio),
            "%s has many-to-one weights, which requires presynaptic dimension to be a "
            "power of two times the postsynaptic dimension (pre n%c = %d; post n%c = %d)\n",
            objectName, axis, nPre, axis, nPost);
      margin = (patchSize - 1) * densityRatio / 2;
      assert(2 * margin * nPost == (patchSize - 1) * nPre);
   }
   else {
      assert(nPre < nPost); // one-to-many
      FatalIf(
            nPost % nPre != 0,
            "%s has one-to-many weights, which requires postsynaptic dimension to be a "
            "power of two times the presynaptic dimension (pre n%c = %d; post n%c = %d)\n",
            objectName, axis, nPre, axis, nPost);
      int densityRatio = nPost / nPre;
      double log2densityRatio = std::log2(densityRatio);
      FatalIf(
            log2densityRatio != std::round(log2densityRatio),
            "%s has one-to-many weights, which requires postsynaptic dimension to be a "
            "power of two times the presynaptic dimension (pre n%c = %d; post n%c = %d)\n",
            objectName, axis, nPre, axis, nPost);
      FatalIf(
            patchSize % densityRatio != 0,
            "%s has one-to-many weights, which requires postsynaptic dimension to be a "
            "power of two times the presynaptic dimension (pre n%c = %d; post n%c = %d)\n",
            objectName, axis, nPre, axis, nPost);
      int numCells = patchSize / densityRatio;
      margin       = numCells / 2;
      // integer division is correct, no matter whether numCells is even or odd
   }
   return margin;
}

}
