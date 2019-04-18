#include "TestConnProbe.hpp"

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
   HyPerConn *conn = getTargetHyPerConn();
   int numPreExt   = conn->getPre()->getNumExtended();
   int syw         = conn->getPatchStrideY(); // stride in patch

   for (int kPre = 0; kPre < numPreExt; kPre++) {
      Patch const *patch = conn->getPatch(kPre);
      int nk             = conn->getPatchSizeF() * patch->nx;

      float *data = conn->getWeightsData(0, kPre);
      int ny      = patch->ny;
      for (int y = 0; y < ny; y++) {
         float *dataYStart = data + y * syw;
         for (int k = 0; k < nk; k++) {
            // Initially, origPre=128/255 and clonePre=64/255, and origConn is 5x5 with weight 1.
            // After layers update, origPost=25*128/255 and clonePost=25*128/255.
            // The weight update is then dw = (((128/255)*(25*128/255)) + ((64/255)*(25*64/255)))/2
            // The updated w is then 1 + dw = 1 + (10240/2601) = 4.936947...
            // Because of roundoff issues, the observed value is 4.9369383...
            if (fabs(timed - 0) < (parent->getDeltaTime() / 2)) {
               FatalIf(dataYStart[k] != 1.0f, "Test failed.\n");
            }
            else if (fabs(timed - 1) < (parent->getDeltaTime() / 2)) {
               FatalIf(!(fabsf(dataYStart[k] - 4.9369f) <= 1.0e-4f), "Test failed.\n");
            }
         }
      }
   }
   return Response::SUCCESS;
}

} // end of namespace PV
