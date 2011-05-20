/*
 * test_transpose_transpose.cpp
 *
 * tests whether FeedbackConn transposes the weights correctly by taking
 * the transpose twice and checking whether the result is identical to the
 * original.
 */

#include <float.h>
#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/connections/TransposeConn.hpp"
#include "../src/connections/FeedbackConn.hpp"
#include "../src/layers/ANNLayer.hpp"

using namespace PV;

int oneToOneForTransposeConn(int argc, char * argv[]);
int manyToOneForTransposeConn(int argc, char * argv[]);
int oneToManyForTransposeConn(int argc, char * argv[]);
int oneToOneForFeedbackConn(int argc, char * argv[]);
int manyToOneForFeedbackConn(int argc, char * argv[]);
int oneToManyForFeedbackConn(int argc, char * argv[]);

int testTransposeOfTransposeWeights(KernelConn * originalMap, TransposeConn * transpose, TransposeConn * transposeOfTranspose, const char * message);
int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2);
int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index);
int testDimensionsEqual(int index, int val1, int val2, const char * pName);
int dumpWeights(KernelConn * kconn, FILE * stream);

int main(int argc, char * argv[]) {

   int status = PV_SUCCESS;
   int status1;
   status1 = oneToOneForFeedbackConn(argc, argv);
   status = (status == PV_SUCCESS && status1 == PV_SUCCESS) ? PV_SUCCESS : PV_FAILURE;
   status1 = manyToOneForFeedbackConn(argc, argv);
   status = (status == PV_SUCCESS && status1 == PV_SUCCESS) ? PV_SUCCESS : PV_FAILURE;
   status1 = oneToManyForFeedbackConn(argc, argv);
   status = (status == PV_SUCCESS && status1 == PV_SUCCESS) ? PV_SUCCESS : PV_FAILURE;
   status1 = oneToOneForTransposeConn(argc, argv);
   status = (status == PV_SUCCESS && status1 == PV_SUCCESS) ? PV_SUCCESS : PV_FAILURE;
   status1 = manyToOneForTransposeConn(argc, argv);
   status = (status == PV_SUCCESS && status1 == PV_SUCCESS) ? PV_SUCCESS : PV_FAILURE;
   status1 = oneToManyForTransposeConn(argc, argv);
   status = (status == PV_SUCCESS && status1 == PV_SUCCESS) ? PV_SUCCESS : PV_FAILURE;

   return status;
}

int oneToOneForTransposeConn(int argc, char * argv[]) {
   HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);
   // Layers
   ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
   ANNLayer * layerB1to1 = new ANNLayer("test_transpose_transpose Layer B One to one", hc);

   // Connections
   KernelConn * originalMap1to1 = new KernelConn("test_transpose_transpose One to one original map", hc, layerA, layerB1to1, CHANNEL_EXC);
   assert(originalMap1to1);
   TransposeConn * transpose1to1 = new TransposeConn("test_transpose_transpose One to one transpose", hc, layerB1to1, layerA, CHANNEL_INHB, originalMap1to1);
   assert(transpose1to1);
   TransposeConn * transposeOfTranspose1to1 = new TransposeConn("test_transpose_transpose One to one transpose of transpose", hc, layerA, layerB1to1, CHANNEL_INHB, transpose1to1);
   assert(transposeOfTranspose1to1);

   int status = testTransposeOfTransposeWeights(originalMap1to1, transpose1to1, transposeOfTranspose1to1, "One-to-one case, TransposeConn");
   delete hc;
   return status;
}
int manyToOneForTransposeConn(int argc, char * argv[]) {
   HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);
   // Layers
   ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
   ANNLayer * layerBManyTo1 = new ANNLayer("test_transpose_transpose Layer B Many to one", hc);

   // Connections
   KernelConn * originalMapManyTo1 = new KernelConn("test_transpose_transpose Many to one original map", hc, layerA, layerBManyTo1, CHANNEL_EXC);
   assert(originalMapManyTo1);
   TransposeConn * transposeManyTo1 = new TransposeConn("test_transpose_transpose Many to one transpose", hc, layerBManyTo1, layerA, CHANNEL_INHB, originalMapManyTo1);
   assert(transposeManyTo1);
   TransposeConn * transposeOfTransposeManyTo1 = new TransposeConn("test_transpose_transpose Many to one transpose of transpose", hc, layerA, layerBManyTo1, CHANNEL_INHB, transposeManyTo1);
   assert(transposeOfTransposeManyTo1);

   int status = testTransposeOfTransposeWeights(originalMapManyTo1, transposeManyTo1, transposeOfTransposeManyTo1, "Many-to-one case, FeedbackConn");
   delete hc;
   return status;
}

int oneToManyForTransposeConn(int argc, char * argv[]) {
   HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);
   // Layers
   ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
   ANNLayer * layerB1toMany = new ANNLayer("test_transpose_transpose Layer B One to many", hc);

   // Connections
   KernelConn * originalMap1toMany = new KernelConn("test_transpose_transpose One to many original map", hc, layerA, layerB1toMany, CHANNEL_EXC);
   assert(originalMap1toMany);
   TransposeConn * transpose1toMany = new TransposeConn("test_transpose_transpose One to many transpose", hc, layerB1toMany, layerA, CHANNEL_INHB, originalMap1toMany);
   assert(transpose1toMany);
   TransposeConn * transposeOfTranspose1toMany = new TransposeConn("test_transpose_transpose One to many transpose of transpose", hc, layerA, layerB1toMany, CHANNEL_INHB, transpose1toMany);
   assert(transposeOfTranspose1toMany);

   int status = testTransposeOfTransposeWeights(originalMap1toMany, transpose1toMany, transposeOfTranspose1toMany, "One-to-many case, FeedbackConn");
   delete hc;
   return status;
}

int oneToOneForFeedbackConn(int argc, char * argv[]) {
   HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);
   // Layers
   ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
   ANNLayer * layerB1to1 = new ANNLayer("test_transpose_transpose Layer B One to one", hc);

   // Connections
   KernelConn * originalMap1to1 = new KernelConn("test_transpose_transpose One to one original map", hc, layerA, layerB1to1, CHANNEL_EXC);
   assert(originalMap1to1);
   FeedbackConn * transpose1to1 = new FeedbackConn("test_transpose_transpose One to one transpose", hc, CHANNEL_INHB, originalMap1to1);
   assert(transpose1to1);
   FeedbackConn * transposeOfTranspose1to1 = new FeedbackConn("test_transpose_transpose One to one transpose of transpose", hc, CHANNEL_INHB, transpose1to1);
   assert(transposeOfTranspose1to1);

   int status = testTransposeOfTransposeWeights(originalMap1to1, transpose1to1, transposeOfTranspose1to1, "One-to-one case, FeedbackConn");
   delete hc;
   return status;
}

int manyToOneForFeedbackConn(int argc, char * argv[]) {
   HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);
   // Layers
   ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
   ANNLayer * layerBManyTo1 = new ANNLayer("test_transpose_transpose Layer B Many to one", hc);

   // Connections
   KernelConn * originalMapManyTo1 = new KernelConn("test_transpose_transpose Many to one original map", hc, layerA, layerBManyTo1, CHANNEL_EXC);
   assert(originalMapManyTo1);
   FeedbackConn * transposeManyTo1 = new FeedbackConn("test_transpose_transpose Many to one transpose", hc, CHANNEL_INHB, originalMapManyTo1);
   assert(transposeManyTo1);
   FeedbackConn * transposeOfTransposeManyTo1 = new FeedbackConn("test_transpose_transpose Many to one transpose of transpose", hc, CHANNEL_INHB, transposeManyTo1);
   assert(transposeOfTransposeManyTo1);

   int status = testTransposeOfTransposeWeights(originalMapManyTo1, transposeManyTo1, transposeOfTransposeManyTo1, "Many-to-one case, FeedbackConn");
   delete hc;
   return status;
}

int oneToManyForFeedbackConn(int argc, char * argv[]) {
   HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);
   // Layers
   ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
   ANNLayer * layerB1toMany = new ANNLayer("test_transpose_transpose Layer B One to many", hc);

   // Connections
   KernelConn * originalMap1toMany = new KernelConn("test_transpose_transpose One to many original map", hc, layerA, layerB1toMany, CHANNEL_EXC);
   assert(originalMap1toMany);
   FeedbackConn * transpose1toMany = new FeedbackConn("test_transpose_transpose One to many transpose", hc, CHANNEL_INHB, originalMap1toMany);
   assert(transpose1toMany);
   FeedbackConn * transposeOfTranspose1toMany = new FeedbackConn("test_transpose_transpose One to many transpose of transpose", hc, CHANNEL_INHB, transpose1toMany);
   assert(transposeOfTranspose1toMany);

   int status = testTransposeOfTransposeWeights(originalMap1toMany, transpose1toMany, transposeOfTranspose1toMany, "One-to-many case, FeedbackConn");
   delete hc;
   return status;
}

int testTransposeOfTransposeWeights(KernelConn * originalMap, TransposeConn * transpose, TransposeConn * transposeOfTranspose, const char * message) {
   int status = testWeightsEqual(originalMap, transposeOfTranspose);
   if( status == PV_SUCCESS ) {
      printf("%s: test_transpose_transpose passed.\n", message);
   }
   else {
      fprintf(stderr, "%s: test_transpose_transpose passed.\n", message);
      dumpWeights(originalMap, stdout);
      dumpWeights(transpose, stdout);
      dumpWeights(transposeOfTranspose, stdout);
   }
}

int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2) {
   int status = PV_SUCCESS;
   int numWeightPatches = conn1->numWeightPatches(0);
   if( numWeightPatches != conn2->numWeightPatches(0) ) {
       fprintf(stderr, "testEqualWeights:  numWeightPatches not equal.\n");
       return PV_FAILURE;
   }

   for( int patchindex = 0; patchindex < numWeightPatches; patchindex++ ) {
       int status1 = testPatchesEqual( conn1->axonalArbor(patchindex, LOCAL)->weights, conn2->axonalArbor(patchindex, LOCAL)->weights, patchindex);
       if( status != PV_SUCCESS || status1 != PV_SUCCESS ) {
           status = status1;
       }
   }
   return status;
}

int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index) {
   int nx = patch1->nx;
   if( testDimensionsEqual(index, nx, patch2->nx, "nx") != PV_SUCCESS) return PV_FAILURE;
   int ny = patch1->ny;
   if( testDimensionsEqual(index, ny, patch2->ny, "nx") != PV_SUCCESS) return PV_FAILURE;
   int nf = patch1->nf;
   if( testDimensionsEqual(index, nf, patch2->nf, "nx") != PV_SUCCESS) return PV_FAILURE;
   if( testDimensionsEqual(index, patch1->sx, patch2->sx, "nx") != PV_SUCCESS) return PV_FAILURE;
   if( testDimensionsEqual(index, patch1->sy, patch2->sy, "nx") != PV_SUCCESS) return PV_FAILURE;
   if( testDimensionsEqual(index, patch1->sf, patch2->sf, "nx") != PV_SUCCESS) return PV_FAILURE;

   int status = PV_SUCCESS;
   int n = nx*ny*nf;
   for(int k=0; k<n; k++) {
       if( fabs(patch1->data[k] - patch2->data[k]) > 10*FLT_EPSILON ) {
           fprintf(stderr, "testWeightsEqual: index into layer = %d, index into patch = %d,\n", index, k);
           fprintf(stderr, "    (%f versus %f).\n",patch1->data[k],patch2->data[k]);
           status = PV_FAILURE;
       }
   }
   return status;
}

int testDimensionsEqual(int index, int val1, int val2, const char * pName) {
   if( val1 != val2 ) {
      fprintf(stderr, "testDimensionsEqual: index into layer %d: %s not equal (%d versus %d).\n", index, pName, val1, val2);
      return PV_FAILURE;
   }
   else return PV_SUCCESS;
}

int dumpWeights(KernelConn * kconn, FILE * stream) {
   int numKernelPatches = kconn->numDataPatches(0);
   fprintf(stream, "Dumping weights for connection %s\n", kconn->getName() );
   for(int kn = 0; kn < numKernelPatches; kn++) {
       PVPatch * kp = kconn->getKernelPatch(kn);
       int nx = kp->nx;
       int ny = kp->ny;
       int nf = kp->nf;
       fprintf(stream, "Kernel Patch %d: nx=%d, ny=%d, nf=%d\n",
               kn, nx, ny, nf);
       for(int k=0; k<nx*ny*nf; k++) {
           fprintf(stream, "    Index %4d, x=%3d, y=%3d, f=%3d: Value %g\n", k,
                   kxPos(k, nx, ny, nf), kyPos(k, nx, ny, nf),
                   featureIndex(k, nx, ny, nf), kp->data[k]);
       }
   }
   return PV_SUCCESS;
}
