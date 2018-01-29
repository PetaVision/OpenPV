#include "MomentumTestConnProbe.hpp"

namespace PV {

MomentumTestConnProbe::MomentumTestConnProbe(const char *probename, HyPerCol *hc) {
   initialize_base();
   int status = initialize(probename, hc);
   FatalIf(!(status == PV_SUCCESS), "Test failed.\n");
}

MomentumTestConnProbe::MomentumTestConnProbe() { initialize_base(); }

MomentumTestConnProbe::~MomentumTestConnProbe() {}

int MomentumTestConnProbe::initialize_base() { return PV_SUCCESS; }

void MomentumTestConnProbe::initNumValues() { setNumValues(-1); }

Response::Status MomentumTestConnProbe::outputState(double timed) {
   // Grab weights of probe and test for the value of .625/1.5, or .4166666
   HyPerConn *conn = getTargetHyPerConn();
   int numPreExt   = conn->getPre()->getNumExtended();
   int syw         = conn->getPatchStrideY(); // stride in patch

   for (int kPre = 0; kPre < numPreExt; kPre++) {
      Patch const *patch = conn->getPatch(kPre);
      int nk             = conn->getPatchSizeF() * patch->nx;

      float *data = conn->getWeightsData(0, kPre);
      int ny      = patch->ny;
      float wCorrect;
      for (int y = 0; y < ny; y++) {
         float *dataYStart = data + y * syw;
         for (int k = 0; k < nk; k++) {
            float wObserved = dataYStart[k];
            if (timed < 3) {
               wCorrect = 0;
            }
            else {
               wCorrect = 0.376471f;
               for (int i = 0; i < (timed - 3); i++) {
                  wCorrect += 0.376471f * expf(-(2 * (i + 1)));
               }
            }
            FatalIf(!(fabsf(wObserved - wCorrect) <= 1e-4f), "Test failed.\n");
         }
      }
   }
   return Response::SUCCESS;
}

} // end of namespace PV
