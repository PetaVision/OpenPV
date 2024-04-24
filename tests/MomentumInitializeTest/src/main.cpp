/*
 * main.cpp for MomentumInitializeTest
 *
 * This test uses two params files,
 *
 * MomentumInitialize_initialrun.params
 * MomentumInitialize_restart.params
 *
 * Each has a PvpLayer, a ConstantLayer, and a MomentumConn
 * on channel -1.
 *
 * The PvpLayer reads the file input/Input.pvp, which is an 8x8x1
 * layer with 64 frames; each frame has one pixel set to one and the
 * rest to zero; the index of the lit pixel corresponds to the frame
 * index.
 *
 * The ConstantLayer is an 8x8x1 layer of all ones.
 *
 * The MomentumConn layer is 5x5, with a weight update period of five.
 *
 * The first params file sets its outputPath to output_initialrun/ and runs
 * until time = 100.
 * The second params file initializes the connection and input layer to the
 * data from the first run at time 50 and runs until time 50.  For the
 * connection, it does so by setting the parameters initPrev_dWFile to the
 * output file of the first run and prev_dWFrameNumber to 50 (since dt = 1 for
 * each run, the frame number corresponds to the simulation time).
 *
 * The ending state of the two runs should be identical. The test passes
 * when the MomentumConn's weights and prev_dW values are equivalent across
 * the two runs.
 */

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <connections/MomentumConn.hpp>
#include <io/FileManager.hpp>
#include <structures/Weights.hpp>
#include <weightupdaters/MomentumUpdater.hpp>

#include <cstdlib>      // EXIT_FAILURE, EXIT_SUCCESS, size_t
#include <cerrno>       // errno, ENOENT
#include <cstring>      // strerror, memcpy
#include <memory>       // unique_ptr
#include <sys/stat.h>   // stat
#include <unistd.h>     // unlink

char const *params_initialrun  = "input/MomentumInitialize_initialrun.params";
char const *params_restart     = "input/MomentumInitialize_restart.params";
char const *weights_initialrun = "output_initialrun/MomentumConn.pvp";
char const *weights_restart    = "output_restart/MomentumConn.pvp";
char const *prev_dW_initialrun = "output_initialrun/MomentumConn.prevDelta.pvp";
char const *prev_dW_restart    = "output_restart/MomentumConn.prevDelta.pvp";

using namespace PV;

int cleanOutputFiles(std::unique_ptr<PV_Init> &pv_init);
int deleteIfPresent(char const *targetFile, FileManager const &fileManager);
std::unique_ptr<Weights> getWeights(HyPerCol &hc);
std::unique_ptr<Weights> getPrev_dW(HyPerCol &hc);
std::unique_ptr<Weights> copyWeights(Weights const *inWeights);
int compare(
      std::unique_ptr<Weights> const &weights1,
      std::unique_ptr<Weights> const &weights2,
      char const *desc);

int main(int argc, char *argv[]) {

   auto *pv_initPtr = new PV_Init(&argc, &argv, false /*do not allow unrecognized arguments*/);
   std::unique_ptr<PV_Init> pv_init(pv_initPtr);
   int status = PV_SUCCESS;
   
   if (!pv_init->getStringArgument("ParamsFile").empty()) {
      if (pv_init->getWorldRank() == 0) {
         ErrorLog().printf(
               "%s should be run without the params file argument.\n", pv_init->getProgramName());
      }
      status = PV_FAILURE;
   }
   cleanOutputFiles(pv_init);
   std::unique_ptr<Weights> weightsInitial, prev_dWInitial;
   if (status == PV_SUCCESS) {
      pv_init->setParams(params_initialrun);
      HyPerCol hc(pv_init.get());
      status = hc.run();
      if (status == PV_SUCCESS) {
         weightsInitial = getWeights(hc);
         prev_dWInitial = getPrev_dW(hc);
      }
   }
   std::unique_ptr<Weights> weightsRestart, prev_dWRestart;
   if (status == PV_SUCCESS) {
      pv_init->setParams(params_restart);
      HyPerCol hc(pv_init.get());
      status = hc.run();
      if (status == PV_SUCCESS) {
         weightsRestart = getWeights(hc);
         prev_dWRestart = getPrev_dW(hc);
      }
   } 
   if (status == PV_SUCCESS) {
      status = compare(weightsInitial, weightsRestart, "Weights");
   }
   if (status == PV_SUCCESS) {
      status = compare(prev_dWInitial, prev_dWRestart, "prev_dW");
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cleanOutputFiles(std::unique_ptr<PV_Init> &pv_init) {
   auto *pv_comm          = pv_init->getCommunicator();
   auto mpiBlock = pv_comm->getIOMPIBlock();
   FileManager fileManager(mpiBlock, ".");

   int status = PV_SUCCESS;
   if (mpiBlock->getRank() == 0) {
      if (status == PV_SUCCESS) {
         status = deleteIfPresent(weights_initialrun, fileManager);
      }
      if (status == PV_SUCCESS) {
         status = deleteIfPresent(weights_restart, fileManager);
      }
      if (status == PV_SUCCESS) {
         status = deleteIfPresent(prev_dW_initialrun, fileManager);
      }
      if (status == PV_SUCCESS) {
         status = deleteIfPresent(prev_dW_restart, fileManager);
      }
   }
   return status;
}

int deleteIfPresent(char const *targetFile, FileManager const &fileManager) {
   int status = PV_SUCCESS;
   if (fileManager.queryFileExists(targetFile)) {
      fileManager.deleteFile(targetFile);
   }
   else if (errno != 0) {
      ErrorLog().printf("Unable to delete \"%s\": %s\n", targetFile, std::strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}

std::unique_ptr<Weights> getWeights(HyPerCol &hc) {
   auto *conn    = dynamic_cast<MomentumConn*>(hc.getObjectFromName("MomentumConn"));
   auto *wgtPair = conn->getComponentByType<WeightsPair>();
   Weights const *foundWeights     = wgtPair->getPreWeights();
   std::unique_ptr<Weights> result = copyWeights(foundWeights);
   return result;
}

std::unique_ptr<Weights> getPrev_dW(HyPerCol &hc) {
   auto *conn            = dynamic_cast<MomentumConn*>(hc.getObjectFromName("MomentumConn"));
   auto *momentumUpdater = conn->getComponentByType<MomentumUpdater>();
   Weights const *foundPrev_dW     = momentumUpdater->getPrevDeltaWeights();
   std::unique_ptr<Weights> result = copyWeights(foundPrev_dW);
   return result;
}

std::unique_ptr<Weights> copyWeights(Weights const *inWeights) {
   Weights *resultPtr = new Weights(
         inWeights->getName(),
         inWeights->getPatchSizeX(),
         inWeights->getPatchSizeY(),
         inWeights->getPatchSizeF(),
         &inWeights->getGeometry()->getPreLoc(),
         &inWeights->getGeometry()->getPostLoc(),
         inWeights->getNumArbors(),
         inWeights->getSharedFlag(),
         inWeights->getTimestamp());
   std::unique_ptr<Weights> result(resultPtr);
   result->allocateDataStructures();
   for (int a = 0; a < inWeights->getNumArbors(); ++a) {
      float *outData      = result->getData(a);
      float const *inData = inWeights->getData(a);
      std::size_t patchSize  = static_cast<std::size_t>(inWeights->getPatchSizeOverall());
      std::size_t numPatches = static_cast<std::size_t>(inWeights->getNumDataPatches());
      memcpy(outData, inData, patchSize * numPatches * sizeof(float));
   }
   return result;
}

int compare(
      std::unique_ptr<Weights> const &weights1,
      std::unique_ptr<Weights> const &weights2,
      char const *desc) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) {
      if (weights1->getNumArbors() != weights2->getNumArbors()) {
         ErrorLog().printf(
               "%s: Number of arbors do not agree (%d vs %d)\n",
               desc, weights1->getNumArbors(), weights2->getNumArbors());
         status = PV_FAILURE;
      }
   }
   int numArbors = weights1->getNumArbors();
   if (status == PV_SUCCESS) {
      if (weights1->getNumDataPatchesX() != weights2->getNumDataPatchesX()) {
         ErrorLog().printf(
               "%s: Number of patches in x-direction do not agree (%d vs %d)\n",
               desc, weights1->getNumDataPatchesX(), weights2->getNumDataPatchesX());
         status = PV_FAILURE;
      }
   }
   int numPatchesX = weights1->getNumDataPatchesX();
   if (status == PV_SUCCESS) {
      if (weights1->getNumDataPatchesY() != weights2->getNumDataPatchesY()) {
         ErrorLog().printf(
               "%s: Number of patches in y-direction do not agree (%d vs %d)\n",
               desc, weights1->getNumDataPatchesY(), weights2->getNumDataPatchesY());
         status = PV_FAILURE;
      }
   }
   int numPatchesY = weights1->getNumDataPatchesY();
   if (status == PV_SUCCESS) {
      if (weights1->getNumDataPatchesF() != weights2->getNumDataPatchesF()) {
         ErrorLog().printf(
               "%s: Number of patches in f-direction do not agree (%d vs %d)\n",
               desc, weights1->getNumDataPatchesF(), weights2->getNumDataPatchesF());
         status = PV_FAILURE;
      }
   }
   int numPatchesF = weights1->getNumDataPatchesF();
   if (status == PV_SUCCESS) {
      if (weights1->getPatchSizeX() != weights2->getPatchSizeX()) {
         ErrorLog().printf(
               "%s: Patch size in x-direction do not agree (%d vs %d)\n",
               desc, weights1->getPatchSizeX(), weights2->getPatchSizeX());
         status = PV_FAILURE;
      }
   }
   int nxp = weights1->getPatchSizeX();
   if (status == PV_SUCCESS) {
      if (weights1->getPatchSizeY() != weights2->getPatchSizeY()) {
         ErrorLog().printf(
               "%s: Patch size in y-direction do not agree (%d vs %d)\n",
               desc, weights1->getPatchSizeY(), weights2->getPatchSizeY());
         status = PV_FAILURE;
      }
   }
   int nyp = weights1->getPatchSizeY();
   if (status == PV_SUCCESS) {
      if (weights1->getPatchSizeF() != weights2->getPatchSizeF()) {
         ErrorLog().printf(
               "%s: Patch size in f-direction do not agree (%d vs %d)\n",
               desc, weights1->getPatchSizeF(), weights2->getPatchSizeF());
         status = PV_FAILURE;
      }
   }
   int nfp = weights1->getPatchSizeF();

   for (int a = 0; a < numArbors; ++a) {
       float const *arborData1 = weights1->getData(a);
       float const *arborData2 = weights2->getData(a);
       int numValuesPerArbor = nxp * nyp * nfp * numPatchesX * numPatchesY * numPatchesF;
       for (int k = 0; k < numValuesPerArbor; ++k) {
           if (arborData1[k] != arborData2[k]) {
              ErrorLog().printf(
                    "%s: arbor %d, value %d discrepancy (%f versus %f)\n",
                    desc, a, k, (double)arborData1[k], (double)arborData2[k]);
              status = PV_FAILURE;
           }
       }
   }
   return status;
}
