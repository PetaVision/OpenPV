#include "BufferUtilsPvp.hpp"
#include "utils/conversions.hpp"

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

   int numPatches = preLayerNxExt * preLayerNyExt * preLayerNf;
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

void writeActivityHeader(FileStream &fStream, ActivityHeader const &header) {
   fStream.setOutPos(0, true);
   fStream.write(&header, sizeof(header));
}

ActivityHeader readActivityHeader(FileStream &fStream) {
   fStream.setInPos(0L, true);
   uint32_t headerSize = 0U;
   fStream.read(&headerSize, sizeof(uint32_t));
   FatalIf(headerSize != static_cast<uint32_t>(80U),
         "%s is not an activity PVP file (headerSize is %u instead of 80)\n",
         fStream.getFileName().c_str(), static_cast<unsigned>(headerSize));
   fStream.setInPos(0L, true);
   ActivityHeader header;
   fStream.read(&header, 80L);
   return header;
}

SparseFileTable buildSparseFileTable(FileStream &fStream, int upToIndex) {
   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         upToIndex > header.nBands,
         "buildSparseFileTable requested frame %d / %d.\n",
         upToIndex,
         header.nBands);

   SparseFileTable result;
   result.valuesIncluded = header.fileType != PVP_ACT_FILE_TYPE;
   int dataSize          = header.dataSize;
   result.frameLengths.resize(upToIndex + 1, 0);
   result.frameStartOffsets.resize(upToIndex + 1, 0);

   for (int f = 0; f < upToIndex + 1; ++f) {
      double timeStamp      = 0;
      long frameLength      = 0;
      long frameStartOffset = fStream.getInPos();
      fStream.read(&timeStamp, sizeof(double));
      fStream.read(&frameLength, sizeof(int));
      result.frameLengths.at(f)      = frameLength;
      result.frameStartOffsets.at(f) = frameStartOffset;
      if (f < upToIndex) {
         fStream.setInPos(frameLength * (long)dataSize, false);
      }
   }
   return result;
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
