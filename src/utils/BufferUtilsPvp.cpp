#include <cstdint>

#include "BufferUtilsPvp.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

namespace BufferUtils {

bool checkIfActivityPVPFile(std::string const &path) {
   return checkIfActivityPVPFile(path.c_str());
}

bool checkIfActivityPVPFile(char const *path) {
   std::ifstream inputFile(path, std::ios_base::in);
   FatalIf(inputFile.fail(), "Unable to open \"%s\": %s\n", path, strerror(errno));
   std::int32_t hdr[3];
   inputFile.read(reinterpret_cast<char *>(hdr), 12);
   FatalIf(!inputFile, "Unable to read inputPath \"%s\"\n", path);

   bool fileTypeisAct =
         (hdr[2] == PVP_NONSPIKING_ACT_FILE_TYPE or hdr[2] == PVP_ACT_SPARSEVALUES_FILE_TYPE);
   bool hitEOF   = inputFile.eof();
   inputFile.close();
   return (!hitEOF and hdr[0] == 80 and hdr[1] == 20 and fileTypeisAct);
}

void writeActivityHeader(FileStream &fStream, ActivityHeader const &header) {
   fStream.setOutPos(0, true);
   fStream.write(&header, sizeof(header));
}

ActivityHeader readActivityHeader(FileStream &fStream) {
   fStream.setInPos(0L, true);
   uint32_t headerSize = 0U;
   fStream.read(&headerSize, sizeof(uint32_t));
   FatalIf(
         headerSize != static_cast<uint32_t>(80U),
         "%s is not an activity PVP file (headerSize is %u instead of 80)\n",
         fStream.getFileName().c_str(),
         static_cast<unsigned>(headerSize));
   fStream.setInPos(0L, true);
   ActivityHeader header;
   fStream.read(&header, 80L);
   return header;
}

SparseFileTable buildSparseFileTable(FileStream &fStream, int upToIndex) {
   fStream.setInPos(0L, std::ios_base::beg);
   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         upToIndex >= header.nBands,
         "buildSparseFileTable() requested frame %d "
         "when there are only %d (zero-indexed) frames.\n",
         upToIndex,
         header.nBands);

   SparseFileTable result;
   int dataSize = header.dataSize;
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

} // end namespace BufferUtils
} // end namespace PV
