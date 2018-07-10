#include "TestConnProbe.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

TestConnProbe::TestConnProbe(const char *probename, HyPerCol *hc) {
   initialize_base();
   int status = initialize(probename, hc);
   FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
}

TestConnProbe::TestConnProbe() { initialize_base(); }

TestConnProbe::~TestConnProbe() {}

int TestConnProbe::initialize_base() { return PV_SUCCESS; }

void TestConnProbe::initNumValues() { setNumValues(-1); }

Response::Status TestConnProbe::outputState(double timed) {
   // Grab weights of probe and test for the value of .625/1.5, or .4166666
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
            if (fabs(timed - 0) < (parent->getDeltaTime() / 2)) {
               if (fabsf(dataYStart[k] - 1) > 0.01f) {
                  Fatal() << "dataYStart[k]: " << dataYStart[k] << "\n";
               }
               FatalIf(!(fabsf(dataYStart[k] - 1) <= 0.01f), "Test failed.\n");
            }
            else if (fabs(timed - 1) < (parent->getDeltaTime() / 2)) {
               if (fabsf(dataYStart[k] - 1.375f) > 0.01f) {
                  Fatal() << "dataYStart[k]: " << dataYStart[k] << "\n";
               }
               FatalIf(!(fabsf(dataYStart[k] - 1.375f) <= 0.01f), "Test failed.\n");
            }
         }
      }
   }
   return Response::SUCCESS;
}

} // end of namespace PV
