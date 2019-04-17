#include "MomentumTestConnProbe.hpp"

namespace PV {

MomentumTestConnProbe::MomentumTestConnProbe(
      const char *probename,
      PVParams *params,
      Communicator const *comm) {
   initialize_base();
   initialize(probename, params, comm);
}

MomentumTestConnProbe::MomentumTestConnProbe() { initialize_base(); }

MomentumTestConnProbe::~MomentumTestConnProbe() {}

int MomentumTestConnProbe::initialize_base() { return PV_SUCCESS; }

void MomentumTestConnProbe::initNumValues() { setNumValues(-1); }

Response::Status MomentumTestConnProbe::outputState(double simTime, double deltaTime) {
   int numPreExt = getWeights()->getGeometry()->getNumPatches();
   int syw       = getWeights()->getPatchStrideY(); // stride in patch

   for (int kPre = 0; kPre < numPreExt; kPre++) {
      Patch const &patch = getWeights()->getPatch(kPre);
      int nk             = getWeights()->getPatchSizeF() * patch.nx;

      float *data = getWeights()->getDataFromPatchIndex(0, kPre) + patch.offset;
      int ny      = patch.ny;
      float wCorrect;
      for (int y = 0; y < ny; y++) {
         float *dataYStart = data + y * syw;
         for (int k = 0; k < nk; k++) {
            float wObserved = dataYStart[k];
            if (simTime < 2) {
               wCorrect = 0;
            }
            else {
               wCorrect = 0.0832743f;
               for (int i = 0; i < (simTime - 2); i++) {
                  wCorrect += 0.0832743f * expf(-(0.25 * (i + 1)));
               }
            }
            FatalIf(!(fabsf(wObserved - wCorrect) <= 1.0e-4f), "Test failed.\n");
         }
      }
   }
   return Response::SUCCESS;
}

} // end of namespace PV
