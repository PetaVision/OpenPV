#ifndef WEIGHTSFILEIO_HPP_
#define WEIGHTSFILEIO_HPP_

#include "components/Weights.hpp"
#include "io/FileStream.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include <vector>

namespace PV {

/**
 * A class for reading from a weight pvp file stream into a patch atlas,
 * or writing a patch atlas into a weight pvp file stream.
 */

class WeightsFileIO {
  public:
   WeightsFileIO(FileStream *fileStream, MPIBlock const *mpiBlock, Weights *weights);

   ~WeightsFileIO() {}

   double readWeights(int frameNumber);

   void writeWeights(double timestamp, bool compress);

   /**
    * Positions a weight pvp file to the start of the data (i.e. just past the end of the header)
    * of the indicated frame. The header for that frame is read into the buffer pointed by the
    * first argument.
    */
   static void
   moveToFrame(BufferUtils::WeightHeader &header, FileStream &fileStream, int frameNumber);

  private:
   BufferUtils::WeightHeader readHeader(int frameNumber);

   void checkHeader(BufferUtils::WeightHeader const &header);

   bool isCompressedHeader(BufferUtils::WeightHeader const &header);

   double readSharedWeights(int frameNumber, BufferUtils::WeightHeader const &header);

   double readNonsharedWeights(int frameNumber, BufferUtils::WeightHeader const &header);

   void writeSharedWeights(double timestamp, bool compress);

   void writeNonsharedWeights(double timestamp, bool compress);

   /**
    * The size in bytes of one arbor in the PVP file. This is the number of patches times
    * the patch size in bytes. For shared weights, the number of patches is
    * mWeights * getNumDataPatches().
    * For nonshared weights, the number of patches is the number of *global* presynaptic
    * neurons required by a connection of the specified size.
    * (that is, the computation does not use the preLoc.halo, but computes the margin
    * from PatchSizeX and PatchSizeY.
    *
    * In both cases, the patch size in bytes is 8 + nxp*nyp*nfp*dataSize, where
    * dataSize is 1 for compressed weights and 4 for noncompressed weights
    * nxp = mWeights->getPatchSizeX()
    * nyp = mWeights->getPatchSizeY()
    * nfp = mWeights->getPatchSizeF()
    */
   long calcArborSizeFile(bool compressed);

   /**
    * The size in bytes of one arbor, in PVP format, of the weights on one MPI process.
    * For shared weights, this is the same as the value returned by calcArborSizeGlobal.
    * For nonshared weights, the number of patches is the *local* number of extended
    * presynaptic neurons
    */
   long calcArborSizeLocal(bool compressed);

   void calcPatchBox(int &startPatchX, int &endPatchX, int &startPatchY, int &endPatchY);

   void calcPatchRange(
         int nPre,
         int nPost,
         int preStartBorder,
         int preEndBorder,
         int patchSize,
         int &startPatch,
         int &endPatch);

   int calcNeededBorder(int nPre, int nPost, int patchSize);

   void loadWeightsFromBuffer(
         std::vector<unsigned char> const &dataFromFile,
         int arbor,
         float minValue,
         float maxValue,
         bool compressed);

   void decompressPatch(
         unsigned char const *dataFromFile,
         float *destWeights,
         int count,
         float minValue,
         float maxValue);

   void storeSharedPatches(
         std::vector<unsigned char> &dataFromFile,
         int arbor,
         float minValue,
         float maxValue,
         bool compressed);

   void storeNonsharedPatches(
         std::vector<unsigned char> &dataFromFile,
         int arbor,
         float minValue,
         float maxValue,
         bool compressed);

   void compressPatch(
         unsigned char *dataForFile,
         float const *sourceWeights,
         int count,
         float minValue,
         float maxValue);

   /**
    * Writes a patch from the buffer to the current position of the FileStream.
    * patchBuffer contains the patch header; only the active region of the patch is
    * written; bytes outside the active region are left unchanged in the file stream.
    * After the call, the FileStream's write position is at the end of the patch
    * (even if the active region does not extend all the way to the end).
    */
   void writePatch(unsigned char const *patchBuffer, bool compressed);

   // Data members
  private:
   FileStream *mFileStream   = nullptr;
   MPIBlock const *mMPIBlock = nullptr;
   Weights *mWeights         = nullptr;

   int const mRootProcess = 0;
   int const tagbase      = 500;
}; // class WeightsFileIO

} // namespace PV

#endif // WEIGHTSFILEIO_HPP_
