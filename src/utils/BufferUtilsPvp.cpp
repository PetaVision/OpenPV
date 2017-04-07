#include "BufferUtilsPvp.hpp"
#include "utils/conversions.h"

namespace PV {

namespace BufferUtils {

WeightHeader buildWeightHeader(
      bool shared,
      int preLayerNx,
      int preLayerNy,
      int preLayerNf,
      int numArbors,
      bool compress,
      double timestamp,
      int nxp,
      int nyp,
      int nfp,
      float minVal,
      float maxVal,
      int numPatches) {
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

   int numPatchItems = nxp * nyp * nfp;
   if (compress) {
      std::size_t const patchSize = weightPatchSize<unsigned char>(numPatchItems);
      baseHeader.recordSize       = numPatches * patchSize;
      baseHeader.dataSize         = (int)sizeof(unsigned char);
      baseHeader.dataType         = returnDataType<unsigned char>();
   }
   else {
      std::size_t const patchSize = weightPatchSize<float>(numPatchItems);
      baseHeader.recordSize       = numPatches * patchSize;
      baseHeader.dataSize         = (int)sizeof(float);
      baseHeader.dataType         = returnDataType<float>();
   }
   baseHeader.nxProcs   = 1;
   baseHeader.nyProcs   = 1;
   baseHeader.nxGlobal  = baseHeader.nx;
   baseHeader.nyGlobal  = baseHeader.ny;
   baseHeader.kx0       = 0;
   baseHeader.ky0       = 0;
   baseHeader.nBatch    = 1;
   baseHeader.nBands    = numArbors;
   baseHeader.timestamp = timestamp;

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
      PVLayerLoc const *preLayerLoc,
      int numColumnProcesses,
      int numRowProcesses,
      float minVal,
      float maxVal,
      bool compress) {
   WeightHeader weightHeader = buildWeightHeader(
         true /* shared weights*/,
         numPatchesX,
         numPatchesY,
         numPatchesF,
         numArbors,
         compress,
         timestamp,
         nxp,
         nyp,
         nfp,
         minVal,
         maxVal,
         numPatchesX * numPatchesY * numPatchesF);

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
   int numPatchesX, numPatchesY, numPatchesF;
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
         numPatchesF);
   int numPatches = numPatchesX * numPatchesY * numPatchesF;

   WeightHeader weightHeader = buildWeightHeader(
         false /*non-shared weights*/,
         numPatchesX,
         numPatchesY,
         preLayerLoc->nf,
         numArbors,
         compress,
         timestamp,
         nxp,
         nyp,
         nfp,
         minVal,
         maxVal,
         numPatches);

   return weightHeader;
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
      int &numPatchesF) {
   int nxPreRestricted = preLayerLoc->nx * numColumnProcesses;
   int nyPreRestricted = preLayerLoc->ny * numRowProcesses;

   numPatchesX = preLayerLoc->nx * numColumnProcesses;
   numPatchesY = preLayerLoc->ny * numRowProcesses;
   if (extended) {
      int marginX = requiredConvolveMargin(preLayerLoc->nx, postLayerLoc->nx, nxp);
      numPatchesX += marginX + marginX;
      int marginY = requiredConvolveMargin(preLayerLoc->ny, postLayerLoc->ny, nyp);
      numPatchesY += marginY + marginY;
   }
   numPatchesF = preLayerLoc->nf;
}

} // end namespace BufferUtils
} // end namespace PV
