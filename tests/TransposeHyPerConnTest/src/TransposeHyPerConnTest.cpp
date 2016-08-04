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

#include <float.h>
#include <columns/buildandrun.hpp>
#include <connections/TransposeConn.hpp>
#include <layers/ANNLayer.hpp>

using namespace PV;

int testTransposeOfTransposeWeights(HyPerConn * originalMap, TransposeConn * transpose, TransposeConn * transposeOfTranspose, const char * message);
int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2);
int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index, const char * conn1name, const char * conn2name);
int verifyEqual(int val1, int val2, const char * description, const char * name1, const char * name2, int status_in);
int testDataPatchEqual(pvdata_t * w1, pvdata_t * w2, int patchSize, const char * name1, const char * name2, int status_in);
int dumpWeights(HyPerConn * conn);

int main(int argc, char * argv[]) {
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   Communicator * icComm = initObj->getCommunicator();

   if (initObj->getParamsFile() != NULL) {
      int rank = icComm->globalCommRank();
      if (rank==0) {
         pvErrorNoExit().printf("%s does not take -p as an option.  Instead the necessary params file is hard-coded.\n", initObj->getProgramName());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   initObj->setParams("input/TransposeHyPerConnTest.params");

   // Don't call buildandrun because it will delete hc before returning. (I could use the customexit hook)
   HyPerCol * hc = build(initObj);
   hc->run(); // Weight values are initialized when run calls allocateDataStructures

   int status = PV_SUCCESS;

   HyPerConn * originalMap = NULL;
   TransposeConn * transpose = NULL;
   TransposeConn * transposeOfTranspose = NULL;

   BaseConnection * baseConn;
   baseConn = hc->getConnFromName("Original Map for One to One Test");
   originalMap = dynamic_cast<HyPerConn *>(baseConn);
   //pvErrorIf(!(originalMap->usingSharedWeights()), "Test failed.\n");
   transpose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose for One to One Test of TransposeConn"));
   transposeOfTranspose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose of Transpose for One to One Test of TransposeConn"));
   status = testTransposeOfTransposeWeights(originalMap, transpose, transposeOfTranspose, "One-to-one case, TransposeConn");

   baseConn = hc->getConnFromName("Original Map for Many to One Test");
   originalMap = dynamic_cast<HyPerConn *>(baseConn);
   //pvErrorIf(!(originalMap->usingSharedWeights()), "Test failed.\n");
   transpose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose for Many to One Test of TransposeConn"));
   transposeOfTranspose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose of Transpose for Many to One Test of TransposeConn"));
   status = testTransposeOfTransposeWeights(originalMap, transpose, transposeOfTranspose, "Many-to-one case, TransposeConn");

   baseConn = hc->getConnFromName("Original Map for One to Many Test");
   originalMap = dynamic_cast<HyPerConn *>(baseConn);
   //pvErrorIf(!(originalMap->usingSharedWeights()), "Test failed.\n");
   transpose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose for One to Many Test of TransposeConn"));
   transposeOfTranspose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose of Transpose for One to Many Test of TransposeConn"));
   status = testTransposeOfTransposeWeights(originalMap, transpose, transposeOfTranspose, "One-to-many case, TransposeConn");

   baseConn = hc->getConnFromName("Original Map for One to One Test");
   originalMap = dynamic_cast<HyPerConn *>(baseConn);
   //pvErrorIf(!(originalMap->usingSharedWeights()), "Test failed.\n");
   transpose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose for One to One Test of FeedbackConn"));
   transposeOfTranspose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose of Transpose for One to One Test of FeedbackConn"));
   status = testTransposeOfTransposeWeights(originalMap, transpose, transposeOfTranspose, "One-to-one case, FeedbackConn");

   baseConn = hc->getConnFromName("Original Map for Many to One Test");
   originalMap = dynamic_cast<HyPerConn *>(baseConn);
   //pvErrorIf(!(originalMap->usingSharedWeights()), "Test failed.\n");
   transpose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose for Many to One Test of FeedbackConn"));
   transposeOfTranspose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose of Transpose for Many to One Test of FeedbackConn"));
   status = testTransposeOfTransposeWeights(originalMap, transpose, transposeOfTranspose, "Many-to-one case, FeedbackConn");

   baseConn = hc->getConnFromName("Original Map for One to Many Test");
   originalMap = dynamic_cast<HyPerConn *>(baseConn);
   //pvErrorIf(!(originalMap->usingSharedWeights()), "Test failed.\n");
   transpose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose for One to Many Test of FeedbackConn"));
   transposeOfTranspose = dynamic_cast<TransposeConn *>(hc->getConnFromName("Transpose of Transpose for One to Many Test of FeedbackConn"));
   status = testTransposeOfTransposeWeights(originalMap, transpose, transposeOfTranspose, "One-to-many case, FeedbackConn");

   delete hc;
   delete initObj;
   return status;
}

//int manyToOneForTransposeConn(int argc, char * argv[]) {
//   HyPerCol * hc = new HyPerCol("column", argc, argv);
//   // Layers
//   const char * layerAname = "Layer A";
//   const char * layerB1to1name = "Layer B One to one";
//   const char * layerB_ManyTo1Name = "Layer B Many to one";
//   const char * layerB1toManyName = "Layer B One to many";
//   const char * originalConnName = "Many to one original map";
//   const char * transposeConnName = "Many to one transpose";
//   const char * transposeOfTransposeConnName = "Many to one double transpose";
//
//   ANNLayer * layerA = new ANNLayer(layerAname, hc);
//   pvErrorIf(!(layerA), "Test failed.\n");
//   ANNLayer * layerB_ManyTo1 = new ANNLayer(layerB_ManyTo1Name, hc);
//   pvErrorIf(!(layerB_ManyTo1), "Test failed.\n");
//   new ANNLayer(layerB1to1name, hc); // This layer and the next are unused in this test, but get created anyway
//   new ANNLayer(layerB1toManyName, hc); // to cause the params to be read, so we don't get unused-parameter warnings.
//
//   // Connections
//   HyPerConn * originalMapManyto1 = new HyPerConn(originalConnName, hc);
//   pvErrorIf(!(originalMapManyto1), "Test failed.\n");
//   TransposeConn * transposeManyto1 = new TransposeConn(transposeConnName, hc);
//   pvErrorIf(!(transposeManyto1), "Test failed.\n");
//   TransposeConn * transposeOfTransposeManyto1 = new TransposeConn(transposeOfTransposeConnName, hc);
//   pvErrorIf(!(transposeOfTransposeManyto1), "Test failed.\n");
//
//   hc->run(); // Weight values are initialized when run calls allocateDataStructures
//
//   int status = testTransposeOfTransposeWeights(originalMapManyto1, transposeManyto1, transposeOfTransposeManyto1, "Many-to-one case, TransposeConn");
//   delete hc;
//   return status;
//}

int testTransposeOfTransposeWeights(HyPerConn * originalMap, TransposeConn * transpose, TransposeConn * transposeOfTranspose, const char * message) {
   int status = testWeightsEqual(originalMap, transposeOfTranspose);
   if( status == PV_SUCCESS ) {
      pvInfo().printf("%s: TestTransposeConn passed.\n", message);
   }
   else {
      dumpWeights(originalMap);
      dumpWeights(transpose);
      dumpWeights(transposeOfTranspose);
      pvErrorNoExit().printf("%s: TestTransposeConn failed.\n", message);
   }
   return status;
}

int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2) {
   int status = PV_SUCCESS;

   status = verifyEqual(conn1->xPatchSize(), conn2->xPatchSize(), "nxp", conn1->getName(), conn2->getName(), status);
   status = verifyEqual(conn1->yPatchSize(), conn2->yPatchSize(), "nyp", conn1->getName(), conn2->getName(), status);
   status = verifyEqual(conn1->fPatchSize(), conn2->fPatchSize(), "nfp", conn1->getName(), conn2->getName(), status);
   status = verifyEqual(conn1->numberOfAxonalArborLists(), conn2->numberOfAxonalArborLists(), "numAxonalArbors", conn1->getName(), conn2->getName(), status);
   status = verifyEqual(conn1->getNumWeightPatches(), conn2->getNumWeightPatches(), "numWeightPatches", conn1->getName(), conn2->getName(), status);
   status = verifyEqual(conn1->getNumDataPatches(), conn2->getNumDataPatches(), "numDataPatches", conn1->getName(), conn2->getName(), status);

   if (status != PV_SUCCESS) return status;

   int numWeightPatches = conn1->getNumWeightPatches();
   pvErrorIf(!(numWeightPatches == conn2->getNumWeightPatches()), "Test failed.\n");
   for( int patchindex = 0; patchindex < numWeightPatches; patchindex++ ) {
      int status1 = testPatchesEqual( conn1->getWeights(patchindex, LOCAL), conn2->getWeights(patchindex, LOCAL), patchindex, conn1->getName(), conn2->getName());
      if( status1 != PV_SUCCESS ) {
         status = status1;
         break;
      }
   }

   if (status != PV_SUCCESS) return status;

   int numArbors = conn1->numberOfAxonalArborLists();
   pvErrorIf(!(numArbors == conn2->numberOfAxonalArborLists()), "Test failed.\n");
   int numDataPatches = conn1->getNumDataPatches();
   pvErrorIf(!(numDataPatches == conn2->getNumDataPatches()), "Test failed.\n");
   int patchSize = conn1->xPatchSize()*conn1->yPatchSize()*conn1->fPatchSize();
   pvErrorIf(!(patchSize == conn2->xPatchSize()*conn2->yPatchSize()*conn2->fPatchSize()), "Test failed.\n");
   for (int arbor=0; arbor<numArbors; arbor++) {
      for (int dataindex = 0; dataindex < numDataPatches; dataindex++) {
         pvwdata_t * w1 = conn1->get_wDataStart(arbor)+patchSize*dataindex;
         pvwdata_t * w2 = conn2->get_wDataStart(arbor)+patchSize*dataindex;
         status = testDataPatchEqual(w1, w2, patchSize, conn1->getName(), conn2->getName(), status);
         if (status != PV_SUCCESS) break;
      }
   }
   return status;
}

int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index, const char * conn1name, const char * conn2name) {
   int status = PV_SUCCESS;
   status = verifyEqual(patch1->nx, patch2->nx, "nx", conn1name, conn2name, status);
   status = verifyEqual(patch1->ny, patch2->ny, "ny", conn1name, conn2name, status);
   status = verifyEqual(patch1->offset, patch2->offset, "offset", conn1name, conn2name, status);

   return status;
}

int verifyEqual(int val1, int val2, const char * description, const char * name1, const char * name2, int status_in) {
   int status_out = status_in;
   if (val1 != val2) {
      pvErrorNoExit().printf("TransposeHyPerConnTest: %s of \"%s\" and \"%s\" are not equal (%d versus %d).\n", description, name1, name2, val1, val2);
      status_out = PV_FAILURE;
   }
   return status_out;
}

int testDataPatchEqual(pvdata_t * wgts1, pvdata_t * wgts2, int patchSize, const char * name1, const char * name2, int status_in) {
   int status_out = status_in;
   for (int w=0; w<patchSize; w++) {
      pvdata_t w1 = wgts1[w];
      pvErrorIf(!(w1), "Test failed.\n"); // All original weights should be in the range [1,2] (values of wMinInit and wMaxInit)
      pvdata_t w2 = wgts2[w];
      // w2 will be either 0 or in the range [1,2].  It's nonzero iff the weight has any pre-neurons in restricted space.
      if (w2 && w1!=w2) { // If the weight is from an extended neuron, and sharedWeights is off, the transpose will be zero.
         pvErrorNoExit().printf("TransposeHyPerConnTest: value %d of \"%s\" and \"%s\" are not equal (%f versus %f).\n", w, name1, name2, wgts1[w], wgts2[w]);
         status_out = PV_FAILURE;
         if (status_out != PV_SUCCESS) break;
      }
   }
   return status_out;
}

int dumpWeights(HyPerConn * conn) {
   pvErrorNoExit().printf("Dumping weights for connection %s\n", conn->getName() );
   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int numArbors = conn->numberOfAxonalArborLists();
   pvErrorNoExit().printf("    nxp = %d, nyp = %d, nfp = %d, numAxonalArbors = %d\n",
           nxp, nyp, nfp, numArbors);
   int numPatches = conn->getNumWeightPatches();
   for (int arbor=0; arbor<numArbors; arbor++) {
      for(int kn = 0; kn < numPatches; kn++) {
         PVPatch * kp = conn->getWeights(kn, 0);
         int nx = kp->nx;
         int ny = kp->ny;
         int offset = kp->offset;
         pvErrorNoExit().printf("    Weight Patch %d: nx=%d, ny=%d, offset=%d\n",
               kn, nx, ny, offset);
      }
   }
   int numDataPatches = conn->getNumDataPatches();
   for (int arbor=0; arbor<numArbors; arbor++) {
      for(int n=0; n<numDataPatches; n++) {
         for(int k=0; k<nxp*nyp*nfp; k++) {
            pvErrorNoExit().printf("    Arbor %d, Data Patch %d, Index %4d, (x=%3d, y=%3d, f=%3d): Value %g\n",
                    arbor, n, k, kxPos(k, nxp, nyp, nfp), kyPos(k, nxp, nyp, nfp),
                    featureIndex(k, nxp, nyp, nfp), conn->get_wData(arbor, n)[k]);
         }
      }
   }
   return PV_SUCCESS;
}
