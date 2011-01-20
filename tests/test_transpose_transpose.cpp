/*
 * test_transpose_transpose.cpp
 *
 * tests whether FeedbackConn transposes the weights correctly by taking
 * the transpose twice and checking whether the result is identical to the
 * original.
 */

#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/connections/GenerativeConn.hpp"
#include "../src/connections/FeedbackConn.hpp"
#include "../src/layers/ANNLayer.hpp"

using namespace PV;

int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2);
int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index);
int dumpWeights(KernelConn * kconn, FILE * stream);

int main(int argc, char * argv[]) {
    HyPerCol * hc = new HyPerCol("test_transpose_transpose column", argc, argv);

    // Layers
    ANNLayer * layerA = new ANNLayer("test_transpose_transpose Layer A", hc);
    ANNLayer * layerB1to1 = new ANNLayer("test_transpose_transpose Layer B One to one", hc);
    ANNLayer * layerBManyTo1 = new ANNLayer("test_transpose_transpose Layer B Many to one", hc);
    ANNLayer * layerB1toMany = new ANNLayer("test_transpose_transpose Layer B One to many", hc);

    // Connections
    GenerativeConn * originalMap1to1 = new GenerativeConn("test_transpose_transpose One to one original map", hc, layerA, layerB1to1, CHANNEL_EXC);
    assert(originalMap1to1);
    FeedbackConn * transpose1to1 = new FeedbackConn("test_transpose_transpose One to one transpose", hc, CHANNEL_INHB, originalMap1to1);
    assert(transpose1to1);
    FeedbackConn * transposeOfTranspose1to1 = new FeedbackConn("test_transpose_transpose One to one transpose of transpose", hc, CHANNEL_INHB, transpose1to1);
    assert(transposeOfTranspose1to1);

    int status1to1 = testWeightsEqual(originalMap1to1, transposeOfTranspose1to1);
    if(status1to1 == EXIT_SUCCESS) {
        printf("One-to-one case: test_transpose_transpose passed.\n");
    }
    else {
        fprintf(stderr, "One-to-one case: test_transpose_transpose failed; dumping weights...\n");
        dumpWeights(originalMap1to1, stdout);
        dumpWeights(transpose1to1, stdout);
        dumpWeights(transposeOfTranspose1to1, stdout);
    }

    GenerativeConn * originalMapManyTo1 = new GenerativeConn("test_transpose_transpose Many to one original map", hc, layerA, layerBManyTo1, CHANNEL_EXC);
    assert(originalMapManyTo1);
    FeedbackConn * transposeManyTo1 = new FeedbackConn("test_transpose_transpose Many to one transpose", hc, CHANNEL_INHB, originalMapManyTo1);
    assert(transposeManyTo1);
    FeedbackConn * transposeOfTransposeManyTo1 = new FeedbackConn("test_transpose_transpose Many to one transpose of transpose", hc, CHANNEL_INHB, transposeManyTo1);
    assert(transposeOfTransposeManyTo1);

    int statusManyTo1 = testWeightsEqual(originalMapManyTo1, transposeOfTransposeManyTo1);
    if(statusManyTo1 == EXIT_SUCCESS) {
        printf("Many-to-one case: test_transpose_transpose passed.\n");
    }
    else {
        fprintf(stderr, "Many-to-one case: testTransposeOfTranspose failed; dumping weights...\n");
        dumpWeights(originalMapManyTo1, stdout);
        dumpWeights(transposeManyTo1, stdout);
        dumpWeights(transposeOfTransposeManyTo1, stdout);
    }

    GenerativeConn * originalMap1toMany = new GenerativeConn("test_transpose_transpose One to many original map", hc, layerA, layerB1toMany, CHANNEL_EXC);
    assert(originalMap1toMany);
    FeedbackConn * transpose1toMany = new FeedbackConn("test_transpose_transpose One to many transpose", hc, CHANNEL_INHB, originalMap1toMany);
    assert(transpose1toMany);
    FeedbackConn * transposeOfTranspose1toMany = new FeedbackConn("test_transpose_transpose One to many transpose of transpose", hc, CHANNEL_INHB, transpose1toMany);
    assert(transposeOfTranspose1toMany);

    int status1toMany = testWeightsEqual(originalMap1toMany, transposeOfTranspose1toMany);
    if(status1toMany == EXIT_SUCCESS) {
        printf("One-to-many case: test_transpose_transpose passed.\n");
    }
    else {
        fprintf(stderr, "One-to-many case: testTransposeOfTranspose failed; dumping weights...");
        dumpWeights(originalMap1toMany, stdout);
        dumpWeights(transpose1toMany, stdout);
        dumpWeights(transposeOfTranspose1toMany, stdout);
    }

    int status = ( status1to1==EXIT_SUCCESS &&
                   statusManyTo1==EXIT_SUCCESS &&
                   status1toMany==EXIT_SUCCESS ) ? EXIT_SUCCESS : EXIT_FAILURE;
    return status;
}

int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2) {

    int status = EXIT_SUCCESS;
    int numWeightPatches = conn1->numWeightPatches(0);
    if( numWeightPatches != conn2->numWeightPatches(0) ) {
        fprintf(stderr, "testEqualWeights:  numWeightPatches not equal.\n");
        return EXIT_FAILURE;
    }

    for( int patchindex = 0; patchindex < numWeightPatches; patchindex++ ) {
        int status1 = testPatchesEqual( conn1->axonalArbor(patchindex, LOCAL)->weights, conn2->axonalArbor(patchindex, LOCAL)->weights, patchindex);
        if(status1 != EXIT_SUCCESS) {
            status = status1;
        }
    }
    return status;
}

int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index) {
    int nx1 = patch1->nx;
    int nx2 = patch2->nx;
    if( nx1 != nx2 ) {
        fprintf(stderr, "testWeightsEqual: index %d: nx not equal (%d versus %d).\n", index, nx1, nx2);
        return EXIT_FAILURE;
    }
    int ny1 = patch1->ny;
    int ny2 = patch2->ny;
    if( ny1 != ny2 ) {
        fprintf(stderr, "testWeightsEqual: index %d: ny not equal (%d versus %d).\n", index, ny1, ny2);
        return EXIT_FAILURE;
    }
    int nf1 = patch1->nf;
    int nf2 = patch2->nf;
    if( nf1 != nf2 ) {
        fprintf(stderr, "testWeightsEqual: index %d: nf not equal (%d versus %d).\n", index, nf1, nf2);
        return EXIT_FAILURE;
    }
    int sx1 = patch1->sx;
    int sx2 = patch2->sx;
    if( nf1 != nf2 ) {
        fprintf(stderr, "testWeightsEqual: index %d: sx not equal (%d versus %d).\n", index, sx1, sx2);
        return EXIT_FAILURE;
    }
    int sy1 = patch1->sy;
    int sy2 = patch2->sy;
    if( sy1 != sy2 ) {
        fprintf(stderr, "testWeightsEqual: index %d: nf not equal (%d versus %d).\n", index, sy1, sy2);
        return EXIT_FAILURE;
    }
    int sf1 = patch1->sf;
    int sf2 = patch2->sf;
    if( sf1 != sf2 ) {
        fprintf(stderr, "testWeightsEqual: index %d: sf not equal (%d versus %d).\n", index, sf1, sf2);
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;
    int n = nx1*ny1*nf1;
    for(int k=0; k<n; k++) {
        if( fabs(patch1->data[k] - patch2->data[k]) > 0.0001 ) {
            fprintf(stderr, "testWeightsEqual: index into layer = %d, index into patch = %d,\n", index, k);
            fprintf(stderr, "    (%f versus %f).\n",patch1->data[k],patch2->data[k]);
            status = EXIT_FAILURE;
        }
    }
    return status;
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
    return EXIT_SUCCESS;
}
