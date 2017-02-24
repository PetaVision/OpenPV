#include "BufferUtilsPvp.hpp"

namespace PV {

namespace BufferUtils {

WeightHeader buildWeightHeader(
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      int numPatches,
      bool shared,
      double timed,
      PVLayerLoc const *preLayerLoc,
      int numColumnProcesses,
      int numRowProcesses,
      float minVal,
      float maxVal,
      bool compress) {
   BufferUtils::WeightHeader weightHeader;
   ActivityHeader &baseHeader = weightHeader.baseHeader;
   baseHeader.headerSize      = (int)sizeof(weightHeader);
   baseHeader.numParams       = baseHeader.headerSize / 4;
   pvAssert(baseHeader.numParams * 4 == baseHeader.headerSize);

   baseHeader.fileType   = shared ? PVP_KERNEL_FILE_TYPE : PVP_WGT_FILE_TYPE;

   PVHalo const &halo = preLayerLoc->halo;
   baseHeader.nx       = preLayerLoc->nx * numColumnProcesses * halo.lt + halo.rt;
   baseHeader.ny       = preLayerLoc->nx * numRowProcesses * halo.dn + halo.up;
   baseHeader.nf       = preLayerLoc->nf;
   baseHeader.numRecords = numArbors;

   const int numPatchItems     = nxp * nyp * nfp;
   if (compress) {
      const std::size_t patchSize = weightPatchSize<unsigned char>(numPatchItems);
      baseHeader.recordSize = numPatches * patchSize;
      baseHeader.dataSize = (int)sizeof(unsigned char);
      baseHeader.dataType = returnDataType<unsigned char>();
   }
   else {
      const std::size_t patchSize = weightPatchSize<float>(numPatchItems);
      baseHeader.recordSize = numPatches * patchSize;
      baseHeader.dataSize = (int)sizeof(float);
      baseHeader.dataType = returnDataType<float>();
   }
   baseHeader.nxProcs    = 1;
   baseHeader.nyProcs    = 1;
   baseHeader.nxGlobal   = baseHeader.nx;
   baseHeader.nyGlobal   = baseHeader.ny;
   baseHeader.kx0        = 0;
   baseHeader.ky0        = 0;
   baseHeader.nBatch     = 1;
   baseHeader.nBands     = numArbors;
   baseHeader.timestamp  = timed;

   weightHeader.nxp        = nxp;
   weightHeader.nyp        = nyp;
   weightHeader.nfp        = nfp;
   weightHeader.minVal     = minVal;
   weightHeader.maxVal     = maxVal;
   weightHeader.numPatches = numPatches;

   return weightHeader;
}

} // end namespace BufferUtils
} // end namespace PV
