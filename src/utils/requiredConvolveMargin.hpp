#include "utils/PVLog.hpp"

/**
 * For a convolution-based connection between two layers, computes the
 * margin width the presynaptic layer must have, given the size of the
 * presynaptic and postsynaptic layers, and the patch size (the number of
 * postsynaptic neurons each presynaptic neuron connects to).
 * If nPre and nPost are not the same, the larger must be an 2^k times the
 * smaller for some positive integer k.
 * If nPre == nPost, patchSize must be odd.
 * If nPre > nPost (many-to-one), any patchSize is permissible.
 * If nPost > nPre (one-to-many), patchSize must be a multiple of (nPost/nPre).
 */
inline int requiredConvolveMargin(int nPre, int nPost, int patchSize) {
   int margin = 0;
   if (nPre == nPost) {
      FatalIf(patchSize % 2 != 1, "one-to-one weights with even patch size");
      margin = (patchSize - 1) / 2;
   }
   else if (nPre > nPost) { // many-to-one
      FatalIf(nPre % nPost != 0, "many-to-one weights but pre is not a power of two times post\n");
      int densityRatio = nPre / nPost;
      double log2densityRatio = std::log2(densityRatio);
      FatalIf(log2densityRatio != std::round(log2densityRatio), "many-to-one weights but pre is not a power of two times post\n");
      margin = (patchSize - 1) * densityRatio / 2;
      assert(2 * margin * nPost == (patchSize - 1) * nPre);
   }
   else {
      assert(nPre < nPost); // one-to-many
      FatalIf(nPost % nPre != 0, "one-to-many weights but post is not a power of two times pre\n");
      int densityRatio = nPost / nPre;
      double log2densityRatio = std::log2(densityRatio);
      FatalIf(log2densityRatio != std::round(log2densityRatio), "one-to-many weights but post is not a power of two times pre\n");
      FatalIf(patchSize % densityRatio != 0,  "one-to-many weights but patch size is not a multiple of post/pre\n");
      int numCells = patchSize / densityRatio;
      margin       = numCells / 2;
      // integer division is correct, no matter whether numCells is even or odd
   }
   return margin;
}
