#include "BufferUtilsPvp.hpp"
#include "utils/conversions.h"

namespace PV {

namespace BufferUtils {

WeightHeader buildWeightHeader(
      bool shared,
      int preLayerNx,
      int preLayerNy,
      int preLayerNf,
      int preLayerNxExt,
      int preLayerNyExt,
      int numArbors,
      double timestamp,
      int nxp,
      int nyp,
      int nfp,
      bool compress,
      float minVal,
      float maxVal) {
   WeightHeader weightHeader;
   ActivityHeader &baseHeader = weightHeader.baseHeader;
   baseHeader.headerSize      = (int)sizeof(weightHeader);
   baseHeader.numParams       = baseHeader.headerSize / 4;
   pvAssert(baseHeader.numParams * 4 == baseHeader.headerSize);

   baseHeader.fileType   = shared ? PVP_KERNEL_FILE_TYPE : PVP_WGT_FILE_TYPE;
   baseHeader.nx         = preLayerNx;
   baseHeader.ny         = preLayerNy;
   baseHeader.nf         = preLayerNf;
   baseHeader.numRecords = numArbors;
   baseHeader.recordSize = 0;

   int numPatches    = preLayerNxExt * preLayerNyExt * preLayerNf;
   int numPatchItems = nxp * nyp * nfp;
   if (compress) {
      baseHeader.dataSize = (int)sizeof(unsigned char);
      baseHeader.dataType = returnDataType<unsigned char>();
   }
   else {
      baseHeader.dataSize = (int)sizeof(float);
      baseHeader.dataType = returnDataType<float>();
   }
   baseHeader.nxProcs    = 1;
   baseHeader.nyProcs    = 1;
   baseHeader.nxExtended = preLayerNxExt;
   baseHeader.nyExtended = preLayerNyExt;
   baseHeader.kx0        = 0;
   baseHeader.ky0        = 0;
   baseHeader.nBatch     = 1;
   baseHeader.nBands     = numArbors;
   baseHeader.timestamp  = timestamp;

   weightHeader.nxp        = nxp;
   weightHeader.nyp        = nyp;
   weightHeader.nfp        = nfp;
   weightHeader.minVal     = minVal;
   weightHeader.maxVal     = maxVal;
   weightHeader.numPatches = numPatches;

   return weightHeader;
}

WeightHeader buildSharedWeightHeader(
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      double timestamp,
      bool compress,
      float minVal,
      float maxVal) {
   WeightHeader weightHeader = buildWeightHeader(
         true /* shared weights*/,
         numPatchesX,
         numPatchesY,
         numPatchesF,
         numPatchesX,
         numPatchesY,
         numArbors,
         timestamp,
         nxp,
         nyp,
         nfp,
         compress,
         minVal,
         maxVal);

   return weightHeader;
}

WeightHeader buildNonsharedWeightHeader(
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      bool extended,
      double timestamp,
      PVLayerLoc const *preLayerLoc,
      PVLayerLoc const *postLayerLoc,
      int numColumnProcesses,
      int numRowProcesses,
      float minVal,
      float maxVal,
      bool compress) {
   int numPatchesX, numPatchesY, numPatchesF, numPatchesXExt, numPatchesYExt;
   calcNumberOfPatches(
         preLayerLoc,
         postLayerLoc,
         numColumnProcesses,
         numRowProcesses,
         extended,
         nxp,
         nyp,
         numPatchesX,
         numPatchesY,
         numPatchesF,
         numPatchesXExt,
         numPatchesYExt);

   WeightHeader weightHeader = buildWeightHeader(
         false /*non-shared weights*/,
         numPatchesX,
         numPatchesY,
         numPatchesF,
         numPatchesXExt,
         numPatchesYExt,
         numArbors,
         timestamp,
         nxp,
         nyp,
         nfp,
         compress,
         minVal,
         maxVal);

   return weightHeader;
}

std::size_t weightPatchSize(int numWeightsInPatch, bool compressed) {
   if (compressed) {
      return weightPatchSize<unsigned char>(numWeightsInPatch);
   }
   else {
      return weightPatchSize<float>(numWeightsInPatch);
   }
}

void calcNumberOfPatches(
      PVLayerLoc const *preLayerLoc,
      PVLayerLoc const *postLayerLoc,
      int numColumnProcesses,
      int numRowProcesses,
      bool extended,
      int nxp,
      int nyp,
      int &numPatchesX,
      int &numPatchesY,
      int &numPatchesF,
      int &numPatchesXExt,
      int &numPatchesYExt) {
   int nxPreRestricted = preLayerLoc->nx * numColumnProcesses;
   int nyPreRestricted = preLayerLoc->ny * numRowProcesses;

   numPatchesX = preLayerLoc->nx * numColumnProcesses;
   numPatchesY = preLayerLoc->ny * numRowProcesses;

   numPatchesXExt = numPatchesX;
   numPatchesYExt = numPatchesY;
   if (extended) {
      int marginX = requiredConvolveMargin(preLayerLoc->nx, postLayerLoc->nx, nxp);
      numPatchesXExt += marginX + marginX;
      int marginY = requiredConvolveMargin(preLayerLoc->ny, postLayerLoc->ny, nyp);
      numPatchesYExt += marginY + marginY;
   }
   numPatchesF = preLayerLoc->nf;
}

} // end namespace BufferUtils
} // end namespace PV
