#include "TestConnProbe.hpp"

namespace PV {

TestConnProbe::TestConnProbe(const char *probename, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(probename, params, comm);
}

TestConnProbe::TestConnProbe() { initialize_base(); }

TestConnProbe::~TestConnProbe() {}

int TestConnProbe::initialize_base() { return PV_SUCCESS; }

void TestConnProbe::initNumValues() { setNumValues(-1); }

Response::Status TestConnProbe::outputState(double simTime, double deltaTime) {
   int numPreExt = getWeights()->getGeometry()->getNumPatches();
   int syw       = getWeights()->getPatchStrideY(); // stride in patch

   for (int kPre = 0; kPre < numPreExt; kPre++) {
      Patch const &patch = getWeights()->getPatch(kPre);
      int nk             = getWeights()->getPatchSizeF() * patch.nx;

      float *data = getWeights()->getDataFromPatchIndex(0, kPre) + patch.offset;
      int ny      = patch.ny;
      for (int y = 0; y < ny; y++) {
         float *dataYStart = data + y * syw;
         for (int k = 0; k < nk; k++) {
            // Initially, origPre=128/255 and clonePre=64/255, and origConn is 5x5 with weight 1.
            // After layers update, origPost=25*128/255 and clonePost=25*128/255.
            // The weight update is then dw = (((128/255)*(25*128/255)) + ((64/255)*(25*64/255)))/2
            // The updated w is then 1 + dw = 1 + (10240/2601) = 4.936947...
            // Because of roundoff issues, the observed value is 4.9369383...
            if (fabs(simTime - 0.0) < (deltaTime / 2.0)) {
               FatalIf(dataYStart[k] != 1.0f, "Test failed.\n");
            }
            else if (fabs(simTime - 1.0) < (deltaTime / 2.0)) {
               FatalIf(fabsf(dataYStart[k] - 4.9369383f) > 1.0e-4f, "Test failed.\n");
            }
         }
      }
   }
   return Response::SUCCESS;
}

} // end of namespace PV
