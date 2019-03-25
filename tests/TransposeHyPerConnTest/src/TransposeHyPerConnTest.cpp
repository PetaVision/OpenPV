/*
 * TransposeHyPerConnTest.cpp
 *
 *  Created on: Jul 31, 2013
 *      Author: Pete Schultz
 */

/*
 * test_transpose_transpose.cpp
 *
 * Tests whether TransposeConn transposes the weights correctly by taking
 * the transpose twice and checking whether the result is identical to the
 * original.  Based on an old unit test that stopped getting maintained.
 */

#include <columns/Communicator.hpp>
#include <columns/ComponentBasedObject.hpp>
#include <columns/buildandrun.hpp>
#include <components/PatchSize.hpp>
#include <components/SharedWeights.hpp>
#include <components/WeightsPair.hpp>
#include <float.h>
#include <layers/ANNLayer.hpp>

using namespace PV;

int testTransposeConn(
      HyPerCol *hc,
      char const *originalName,
      char const *transposeName,
      char const *transposeOfTransposeName);
int testTransposeOfTransposeWeights(
      ComponentBasedObject *originalMap,
      ComponentBasedObject *transpose,
      ComponentBasedObject *transposeOfTranspose,
      const char *message);
int testWeightsEqual(ComponentBasedObject *conn1, ComponentBasedObject *conn2);
int testPatchesEqual(
      Patch const *patch1,
      Patch const *patch2,
      int index,
      const char *conn1name,
      const char *conn2name);
int verifyEqual(
      int val1,
      int val2,
      const char *description,
      const char *name1,
      const char *name2,
      int status_in);
int testDataPatchEqual(
      float *w1,
      float *w2,
      int patchSize,
      const char *name1,
      const char *name2,
      int status_in);
int dumpWeights(ComponentBasedObject *conn);

int main(int argc, char *argv[]) {
   PV_Init *initObj     = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   Communicator *icComm = initObj->getCommunicator();

   if (initObj->getParams() != nullptr) {
      int rank = icComm->globalCommRank();
      if (rank == 0) {
         ErrorLog().printf(
               "%s does not take -p as an option.  Instead the necessary params file is "
               "hard-coded.\n",
               initObj->getProgramName());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   initObj->setParams("input/TransposeHyPerConnTest.params");

   // Don't call buildandrun because it will delete hc before returning. (I could use the customexit
   // hook)
   HyPerCol *hc = build(initObj);
   hc->run(); // Weight values are initialized when run calls allocateDataStructures

   int status = PV_SUCCESS;

   status = testTransposeConn(
         hc,
         "OriginalMapForOneToOneTest",
         "TransposeForOneToOneTestOfTransposeConn",
         "TransposeOfTransposeForOneToOneTestOfTransposeConn");
   status = testTransposeConn(
         hc,
         "OriginalMapForManyToOneTest",
         "TransposeForManyToOneTestOfTransposeConn",
         "TransposeOfTransposeForManyToOneTestOfTransposeConn");
   status = testTransposeConn(
         hc,
         "OriginalMapForOneToManyTest",
         "TransposeForOneToManyTestOfTransposeConn",
         "TransposeOfTransposeForOneToManyTestOfTransposeConn");

   delete hc;
   delete initObj;
   return status;
}

int testTransposeConn(
      HyPerCol *hc,
      char const *originalName,
      char const *transposeName,
      char const *transposeOfTransposeName) {
   auto *originalMap = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(originalName));
   FatalIf(!originalMap, "Connection \"%s\" does not exist.\n", originalName);
   FatalIf(
         originalMap->getComponentByType<SharedWeights>()->getSharedWeights(),
         "%s uses shared weights, but this test requires shared weights to be off.\n",
         originalMap->getDescription_c());

   auto *transpose = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(transposeName));
   FatalIf(!transpose, "Connection \"%s\" does not exist.\n", transposeName);

   auto *transposeOfTranspose =
         dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(transposeOfTransposeName));
   FatalIf(!transposeOfTranspose, "Connection \"%s\" does not exist.\n", transposeOfTransposeName);

   int status = testTransposeOfTransposeWeights(
         originalMap, transpose, transposeOfTranspose, "One-to-one case, TransposeConn");
   return status;
}

int testTransposeOfTransposeWeights(
      ComponentBasedObject *originalMap,
      ComponentBasedObject *transpose,
      ComponentBasedObject *transposeOfTranspose,
      const char *message) {
   int status = testWeightsEqual(originalMap, transposeOfTranspose);
   if (status == PV_SUCCESS) {
      InfoLog().printf("%s: TestTransposeConn passed.\n", message);
   }
   else {
      dumpWeights(originalMap);
      dumpWeights(transpose);
      dumpWeights(transposeOfTranspose);
      Fatal().printf("%s: TestTransposeConn failed.\n", message);
   }
   return status;
}

int testWeightsEqual(ComponentBasedObject *conn1, ComponentBasedObject *conn2) {
   int status = PV_SUCCESS;

   auto patchSize1 = conn1->getComponentByType<PatchSize>();
   auto patchSize2 = conn2->getComponentByType<PatchSize>();
   status          = verifyEqual(
         patchSize1->getPatchSizeX(),
         patchSize2->getPatchSizeX(),
         "nxp",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         patchSize1->getPatchSizeY(),
         patchSize2->getPatchSizeY(),
         "nyp",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         patchSize1->getPatchSizeF(),
         patchSize2->getPatchSizeF(),
         "nfp",
         conn1->getName(),
         conn2->getName(),
         status);
   int numItems =
         patchSize1->getPatchSizeX() * patchSize1->getPatchSizeY() * patchSize1->getPatchSizeF();

   int numArbors = conn1->getComponentByType<ArborList>()->getNumAxonalArbors();
   status        = verifyEqual(
         numArbors,
         conn2->getComponentByType<ArborList>()->getNumAxonalArbors(),
         "numAxonalArbors",
         conn1->getName(),
         conn2->getName(),
         status);

   auto *preWeights1 = conn1->getComponentByType<WeightsPair>()->getPreWeights();
   auto *preWeights2 = conn2->getComponentByType<WeightsPair>()->getPreWeights();

   int numGeometryPatches = preWeights1->getGeometry()->getNumPatches();
   status                 = verifyEqual(
         numGeometryPatches,
         preWeights2->getGeometry()->getNumPatches(),
         "numGeometryPatches",
         conn1->getName(),
         conn2->getName(),
         status);

   int numDataPatches = preWeights1->getNumDataPatches();
   status             = verifyEqual(
         numDataPatches,
         preWeights2->getNumDataPatches(),
         "numDataPatches",
         conn1->getName(),
         conn2->getName(),
         status);

   if (status != PV_SUCCESS)
      return status;

   for (int patchindex = 0; patchindex < numGeometryPatches; patchindex++) {
      int status1 = testPatchesEqual(
            &preWeights1->getPatch(patchindex),
            &preWeights2->getPatch(patchindex),
            patchindex,
            conn1->getName(),
            conn2->getName());
      if (status1 != PV_SUCCESS) {
         status = status1;
         break;
      }
   }

   if (status != PV_SUCCESS) {
      return status;
   }

   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int dataindex = 0; dataindex < numDataPatches; dataindex++) {
         float *w1 = preWeights1->getData(arbor) + numItems * dataindex;
         float *w2 = preWeights2->getData(arbor) + numItems * dataindex;
         status = testDataPatchEqual(w1, w2, numItems, conn1->getName(), conn2->getName(), status);
         if (status != PV_SUCCESS)
            break;
      }
   }
   return status;
}

int testPatchesEqual(
      Patch const *patch1,
      Patch const *patch2,
      int index,
      const char *conn1name,
      const char *conn2name) {
   int status = PV_SUCCESS;
   status     = verifyEqual(patch1->nx, patch2->nx, "nx", conn1name, conn2name, status);
   status     = verifyEqual(patch1->ny, patch2->ny, "ny", conn1name, conn2name, status);
   status     = verifyEqual(patch1->offset, patch2->offset, "offset", conn1name, conn2name, status);

   return status;
}

int verifyEqual(
      int val1,
      int val2,
      const char *description,
      const char *name1,
      const char *name2,
      int status_in) {
   int status_out = status_in;
   if (val1 != val2) {
      ErrorLog().printf(
            "TransposeHyPerConnTest: %s of \"%s\" and \"%s\" are not equal (%d versus %d).\n",
            description,
            name1,
            name2,
            val1,
            val2);
      status_out = PV_FAILURE;
   }
   return status_out;
}

int testDataPatchEqual(
      float *wgts1,
      float *wgts2,
      int patchSize,
      const char *name1,
      const char *name2,
      int status_in) {
   int status_out = status_in;
   for (int w = 0; w < patchSize; w++) {
      float w1 = wgts1[w];
      FatalIf(!w1, "Test failed.\n"); // All original weights should be in the range [1,2]
      // (values of wMinInit and wMaxInit)
      float w2 = wgts2[w];
      // w2 will be either 0 or in the range [1,2].  It's nonzero iff the weight has any pre-neurons
      // in restricted space.
      if (w2 && w1 != w2) { // If the weight is from an extended neuron, and sharedWeights is off,
         // the transpose will be zero.
         ErrorLog().printf(
               "TransposeHyPerConnTest: value %d of \"%s\" and \"%s\" are not equal (%f versus "
               "%f).\n",
               w,
               name1,
               name2,
               (double)wgts1[w],
               (double)wgts2[w]);
         status_out = PV_FAILURE;
         if (status_out != PV_SUCCESS)
            break;
      }
   }
   return status_out;
}

int dumpWeights(ComponentBasedObject *conn) {
   ErrorLog().printf("Dumping weights for connection %s\n", conn->getName());
   auto *patchSize = conn->getComponentByType<PatchSize>();
   int nxp         = patchSize->getPatchSizeX();
   int nyp         = patchSize->getPatchSizeY();
   int nfp         = patchSize->getPatchSizeF();
   int numArbors   = conn->getComponentByType<ArborList>()->getNumAxonalArbors();
   ErrorLog().printf(
         "    nxp = %d, nyp = %d, nfp = %d, numAxonalArbors = %d\n", nxp, nyp, nfp, numArbors);
   auto *preWeights     = conn->getComponentByType<WeightsPair>()->getPreWeights();
   int const numPatches = preWeights->getGeometry()->getNumPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int kn = 0; kn < numPatches; kn++) {
         Patch const &kp = preWeights->getPatch(kn);
         int nx          = kp.nx;
         int ny          = kp.ny;
         int offset      = kp.offset;
         ErrorLog().printf("    Weight Patch %d: nx=%d, ny=%d, offset=%d\n", kn, nx, ny, offset);
      }
   }
   int const numDataPatches = preWeights->getNumDataPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int n = 0; n < numDataPatches; n++) {
         float *weightsData = preWeights->getDataFromDataIndex(arbor, n);
         for (int k = 0; k < nxp * nyp * nfp; k++) {
            ErrorLog().printf(
                  "    Arbor %d, Data Patch %d, Index %4d, (x=%3d, y=%3d, f=%3d): Value %g\n",
                  arbor,
                  n,
                  k,
                  kxPos(k, nxp, nyp, nfp),
                  kyPos(k, nxp, nyp, nfp),
                  featureIndex(k, nxp, nyp, nfp),
                  (double)weightsData[k]);
         }
      }
   }
   return PV_SUCCESS;
}
