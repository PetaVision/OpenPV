/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <columns/HyPerCol.hpp>
#include <io/FileStreamBuilder.hpp>
#include <structures/Buffer.hpp>
#include <utils/BufferUtilsPvp.hpp>

#include <fstream>

int checkOutput(HyPerCol *hc, int argc, char *argv[]);
Buffer<float> readCorrectBuffer();

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, &checkOutput);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkOutput(HyPerCol *hc, int argc, char *argv[]) {
   int status = PV_SUCCESS;
   std::string outputPathString = hc->getOutputPath();
   Communicator const *communicator = hc->getCommunicator();
   std::shared_ptr<MPIBlock const> mpiBlock = communicator->getIOMPIBlock();
   if (mpiBlock->getRank()) { return PV_SUCCESS; }

   auto fileManager = std::make_shared<PV::FileManager>(mpiBlock, outputPathString);
   auto outputFileStream = FileStreamBuilder(
      fileManager,
      std::string("Output.pvp"),
      false /*isTextFlag*/,
      true /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWritesFlag*/).get();
   BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(*outputFileStream);
   std::string outputFilePath = outputFileStream->getFileName();
   outputFileStream = nullptr;  // done with the stream, since readDenseFromPvp won't use it.

   Buffer<float> correctBuffer = readCorrectBuffer();

   int numFrames = header.nBands;
   Buffer<float> observedBuffer;
   for (int frame = 0; frame < numFrames; ++frame) {
      double timestamp = BufferUtils::readDenseFromPvp(outputFilePath.c_str(), &observedBuffer, frame);
      if (timestamp <= 0.0) { continue; }
      int nx = observedBuffer.getWidth();
      int ny = observedBuffer.getHeight();
      int nf = observedBuffer.getFeatures();
      int N  = observedBuffer.getTotalElements();

      for (int n = 0; n < N; ++n) {
         float observed = observedBuffer.at(n);
         int kf         = featureIndex(n, nx, ny, nf);
         float correct  = correctBuffer.at(kf);
         if (observed != correct) {
            int kx = kxPos(n, nx, ny, nf);
            int ky = kyPos(n, nx, ny, nf);
            ErrorLog().printf(
                  "Error in neuron %d: kx=%d, ky=%d, kf=%d; expected %d, observed %d\n",
                  n,
                  kx,
                  ky,
                  kf,
                  static_cast<int>(correct),
                  static_cast<int>(observed));
            status = PV_FAILURE;
         }
      }
   }

   return status;
}

Buffer<float> readCorrectBuffer() {
   std::ifstream correctBufferStream("input/correct.txt", std::ios_base::in);
   std::vector<float> data;
   while (correctBufferStream) {
      float s;
      correctBufferStream >> s;
      data.push_back(s);
   }
   Buffer<float> result(data, 1, 1, static_cast<int>(data.size()));
   return result;
}
