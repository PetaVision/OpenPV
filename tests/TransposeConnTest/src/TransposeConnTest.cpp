/*
 * TransposeConnTest.cpp
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
#include <columns/buildandrun.hpp>
#include <connections/TransposeConn.hpp>
#include <float.h>
#include <layers/ANNLayer.hpp>

using namespace PV;

int testTransposeConn(
      HyPerCol *hc,
      char const *originalName,
      char const *transposeName,
      char const *transposeOfTransposeName);
int testTransposeOfTransposeWeights(
      HyPerConn *originalMap,
      TransposeConn *transpose,
      TransposeConn *transposeOfTranspose,
      const char *message);
int testWeightsEqual(HyPerConn *conn1, HyPerConn *conn2);
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
int dumpWeights(HyPerConn *conn);

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

   initObj->setParams("input/TransposeConnTest.params");

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
   HyPerConn *originalMap = dynamic_cast<HyPerConn *>(hc->getObjectFromName(originalName));
   FatalIf(!originalMap, "Connection \"%s\" does not exist.\n", originalName);
   FatalIf(
         !originalMap->getSharedWeights(),
         "%s does not use shared weights, but this test requires shared weights to be on.\n",
         originalMap->getDescription_c());

   TransposeConn *transpose = dynamic_cast<TransposeConn *>(hc->getObjectFromName(transposeName));
   FatalIf(!transpose, "Connection \"%s\" does not exist.\n", transposeName);

   TransposeConn *transposeOfTranspose =
         dynamic_cast<TransposeConn *>(hc->getObjectFromName(transposeOfTransposeName));
   FatalIf(!transposeOfTranspose, "Connection \"%s\" does not exist.\n", transposeOfTransposeName);

   int status = testTransposeOfTransposeWeights(
         originalMap, transpose, transposeOfTranspose, "One-to-one case, TransposeConn");
   return status;
}

int testTransposeOfTransposeWeights(
      HyPerConn *originalMap,
      TransposeConn *transpose,
      TransposeConn *transposeOfTranspose,
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

int testWeightsEqual(HyPerConn *conn1, HyPerConn *conn2) {
   int status = PV_SUCCESS;

   status = verifyEqual(
         conn1->getPatchSizeX(),
         conn2->getPatchSizeX(),
         "nxp",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         conn1->getPatchSizeY(),
         conn2->getPatchSizeY(),
         "nyp",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         conn1->getPatchSizeF(),
         conn2->getPatchSizeF(),
         "nfp",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         conn1->getNumAxonalArbors(),
         conn2->getNumAxonalArbors(),
         "numAxonalArbors",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         conn1->getNumGeometryPatches(),
         conn2->getNumGeometryPatches(),
         "numGeometryPatches",
         conn1->getName(),
         conn2->getName(),
         status);
   status = verifyEqual(
         conn1->getNumDataPatches(),
         conn2->getNumDataPatches(),
         "numDataPatches",
         conn1->getName(),
         conn2->getName(),
         status);

   if (status != PV_SUCCESS)
      return status;

   int numGeometryPatches = conn1->getNumGeometryPatches();
   FatalIf(!(numGeometryPatches == conn2->getNumGeometryPatches()), "Test failed.\n");
   for (int patchindex = 0; patchindex < numGeometryPatches; patchindex++) {
      int status1 = testPatchesEqual(
            conn1->getPatch(patchindex),
            conn2->getPatch(patchindex),
            patchindex,
            conn1->getName(),
            conn2->getName());
      if (status1 != PV_SUCCESS) {
         status = status1;
         break;
      }
   }

   if (status != PV_SUCCESS)
      return status;

   int numArbors = conn1->getNumAxonalArbors();
   FatalIf(!(numArbors == conn2->getNumAxonalArbors()), "Test failed.\n");
   int numDataPatches = conn1->getNumDataPatches();
   FatalIf(!(numDataPatches == conn2->getNumDataPatches()), "Test failed.\n");
   int patchSize = conn1->getPatchSizeX() * conn1->getPatchSizeY() * conn1->getPatchSizeF();
   FatalIf(
         !(patchSize == conn2->getPatchSizeX() * conn2->getPatchSizeY() * conn2->getPatchSizeF()),
         "Test failed.\n");
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int dataindex = 0; dataindex < numDataPatches; dataindex++) {
         float *w1 = conn1->getWeightsDataStart(arbor) + patchSize * dataindex;
         float *w2 = conn2->getWeightsDataStart(arbor) + patchSize * dataindex;
         status = testDataPatchEqual(w1, w2, patchSize, conn1->getName(), conn2->getName(), status);
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
            "TransposeConnTest: %s of \"%s\" and \"%s\" are not equal (%d versus %d).\n",
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
      float *w1,
      float *w2,
      int patchSize,
      const char *name1,
      const char *name2,
      int status_in) {
   int status_out = status_in;
   for (int w = 0; w < patchSize; w++) {
      if (w1[w] != w2[w]) {
         ErrorLog().printf(
               "TransposeConnTest: value %d of \"%s\" and \"%s\" are not equal (%f versus %f).\n",
               w,
               name1,
               name2,
               (double)w1[w],
               (double)w2[w]);
         status_out = PV_FAILURE;
         if (status_out != PV_SUCCESS)
            break;
      }
   }
   return status_out;
}

int dumpWeights(HyPerConn *conn) {
   ErrorLog().printf("Dumping weights for connection %s\n", conn->getName());
   int nxp       = conn->getPatchSizeX();
   int nyp       = conn->getPatchSizeY();
   int nfp       = conn->getPatchSizeF();
   int numArbors = conn->getNumAxonalArbors();
   ErrorLog().printf(
         "    nxp = %d, nyp = %d, nfp = %d, numAxonalArbors = %d\n", nxp, nyp, nfp, numArbors);
   int numPatches = conn->getNumGeometryPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int kn = 0; kn < numPatches; kn++) {
         Patch const *kp = conn->getPatch(kn);
         int nx          = kp->nx;
         int ny          = kp->ny;
         int offset      = kp->offset;
         ErrorLog().printf("    Weight Patch %d: nx=%d, ny=%d, offset=%d\n", kn, nx, ny, offset);
      }
   }
   int numDataPatches = conn->getNumDataPatches();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      for (int n = 0; n < numDataPatches; n++) {
         for (int k = 0; k < nxp * nyp * nfp; k++) {
            ErrorLog().printf(
                  "    Arbor %d, Data Patch %d, Index %4d, (x=%3d, y=%3d, f=%3d): Value %g\n",
                  arbor,
                  n,
                  k,
                  kxPos(k, nxp, nyp, nfp),
                  kyPos(k, nxp, nyp, nfp),
                  featureIndex(k, nxp, nyp, nfp),
                  (double)conn->getWeightsData(arbor, n)[k]);
         }
      }
   }
   return PV_SUCCESS;
}
