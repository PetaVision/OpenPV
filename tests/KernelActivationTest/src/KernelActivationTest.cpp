/*
 * KernelActivationTest.cpp
 *
 * Tests kernel activations.  If called with no parameter file specified, will run
 * with params file input/KernelActivationTest-MirrorBCOff.params, and then
 * with params file input/KernelActivationTest-MirrorBCOn.params.
 *
 * If called with a parameter file specified, it will run with that parameter.
 *
 * For connections testing full data, since the pre/post values are .5,
 * weight update should make all the weights .5^2 with a pixel roundoff error
 *
 * For connection testing masked data in both pre and post layers, since
 * all the pre/post is constant, and the kernel normalization takes into account how many
 * pre/post pairs are actually being calculated, all the weights should be identical to
 * the full activations.
 */

#include "arch/mpi/mpi.h"
#include "columns/ComponentBasedObject.hpp"
#include "columns/buildandrun.hpp"
#include "components/PatchSize.hpp"
#include "components/WeightsPair.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

int dumpweights(HyPerCol *hc, PV_Init &initObj);
int dumponeweight(ComponentBasedObject *conn);

int main(int argc, char *argv[]) {
   int status;
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   if (initObj.getParams() == NULL) {
      initObj.setParams("input/KernelActivationTest-fullData.params");
      status = buildandrun(&initObj);
      if (status == PV_SUCCESS) {
         initObj.setParams("input/KernelActivationTest-maskData.params");
         status = rebuildandrun(&initObj);
      }
   }
   else {
      status = buildandrun(&initObj);
   }
   return status;
}

int dumpweights(HyPerCol *hc, PV_Init &initObj) {
   int status         = PV_SUCCESS;
   bool existsgenconn = false;
   for (Observer *obj = hc->getNextObject(nullptr); obj != nullptr; obj = hc->getNextObject(obj)) {
      ComponentBasedObject *conn = dynamic_cast<ComponentBasedObject *>(obj);
      if (conn == nullptr) {
         continue;
      }
      // Only test plastic conns
      auto *weightUpdater = conn->getComponentByType<BaseWeightUpdater>();
      if (weightUpdater->getPlasticityFlag()) {
         existsgenconn = true;
         int status1   = dumponeweight(conn);
         if (status == PV_SUCCESS)
            status = status1;
      }
   }
   if (existsgenconn && status != PV_SUCCESS) {
      for (int k = 0; k < 72; k++) {
         InfoLog().printf("=");
      }
      InfoLog().printf("\n");
   }
   int rank                   = hc->getCommunicator()->commRank();
   std::string paramsFilename = initObj.getStringArgument("ParamsFile");
   if (status != PV_SUCCESS) {
      ErrorLog().printf(
            "Rank %d: %s failed with return code %d.\n", rank, paramsFilename.c_str(), status);
   }
   else {
      InfoLog().printf("Rank %d: %s succeeded.\n", rank, paramsFilename.c_str());
   }
   return status;
}

int dumponeweight(ComponentBasedObject *conn) {
   int status           = PV_SUCCESS;
   bool errorfound      = false;
   auto *patchSize      = conn->getComponentByType<PatchSize>();
   int nxp              = patchSize->getPatchSizeX();
   int nyp              = patchSize->getPatchSizeY();
   int nfp              = patchSize->getPatchSizeF();
   auto *connectionData = conn->getComponentByType<ConnectionData>();
   HyPerLayer *pre      = connectionData->getPre();
   bool usingMirrorBCs  = pre->getComponentByType<BoundaryConditions>()->getMirrorBCflag();
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   auto *weightsPair = conn->getComponentByType<WeightsPair>();
   auto *preWeights  = weightsPair->getPreWeights();
   for (int p = 0; p < preWeights->getNumDataPatches(); p++) {
      float *wgtData = preWeights->getDataFromDataIndex(0, p);
      for (int f = 0; f < nfp; f++) {
         for (int x = 0; x < nxp; x++) {
            for (int y = 0; y < nyp; y++) {
               int idx = kIndex(x, y, f, nxp, nyp, nfp);
               // TODO-CER-2014.4.4 - weight conversion
               float wgt = wgtData[idx];
               // New normalization takes into account if pre is not active
               // The pixel value from the input is actually 127, where we divide it by 255.
               // Not exaclty .5, a little less
               // Squared because both pre and post is grabbing it's activity from the image
               float correct =
                     usingMirrorBCs ? powf(127.0f / 255.0f, 2.0f) : (127.0f / 255.0f) * 0.5f;
               if (fabsf(wgt - correct) > 1.0e-5f) {
                  ErrorLog(errorMessage);
                  if (errorfound == false) {
                     errorfound = true;
                     for (int k = 0; k < 72; k++) {
                        InfoLog().printf("=");
                     }
                     errorMessage.printf("\n");
                     errorMessage.printf("Rank %d, %s:\n", rank, conn->getDescription_c());
                  }
                  errorMessage.printf(
                        "Rank %d, Patch %d, x=%d, y=%d, f=%d: weight=%f, correct=%f, off by a "
                        "factor of %f\n",
                        rank,
                        p,
                        x,
                        y,
                        f,
                        (double)wgt,
                        (double)correct,
                        (double)(wgt / correct));
                  status = PV_FAILURE;
               }
            }
         }
      }
   }
   if (status == PV_SUCCESS) {
      InfoLog().printf("Rank %d, %s: Weights are correct.\n", rank, conn->getDescription_c());
   }
   return status;
}
