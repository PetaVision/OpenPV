/*
 * pv.cpp
 *
 */

#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

#include "../PetaVision/src/columns/HyPerCol.hpp"
#include "../PetaVision/src/connections/HyPerConn.hpp"
#include "../PetaVision/src/connections/KernelConn.hpp"
#include "../PetaVision/src/connections/GeislerConn.hpp"
#include "../PetaVision/src/connections/CocircConn.hpp"
#include "../PetaVision/src/layers/Movie.hpp"
#include "../PetaVision/src/layers/Image.hpp"
#include "../PetaVision/src/layers/Retina.hpp"
#include "../PetaVision/src/layers/NonspikingLayer.hpp"
#include "../PetaVision/src/layers/GeislerLayer.hpp"
#include "../PetaVision/src/io/ConnectionProbe.hpp"
#include "../PetaVision/src/io/GLDisplay.hpp"
#include "../PetaVision/src/io/PostConnProbe.hpp"
#include "../PetaVision/src/io/LinearActivityProbe.hpp"
#include "../PetaVision/src/io/PointProbe.hpp"
#include "../PetaVision/src/io/StatsProbe.hpp"
#include "../PetaVision/src/layers/PVLayer.h"

#include "GenerativeLayer.hpp"
#include "TrainingGenLayer.hpp"
#include "GenerativeConn.hpp"
#include "FeedbackConn.hpp"
#include "IdentConn.hpp"
#include "LateralConn.hpp"
#include "L2NormProbe.hpp"
#include "SparsityTermProbe.hpp"
#include "GenColProbe.hpp"
#include "ChannelProbe.hpp"
#include "VProbe.hpp"

using namespace PV;

int setupThreeLayerGenerativeModel(HyPerCol * hc);
int testCheckPatchSize(HyPerCol * hc);
int testTransposeOfTranspose(HyPerCol * hc, const char * dumpweightsfile);
int mnistTrain(HyPerCol * hc);
int mnistTest(HyPerCol * hc);
int mnistNoOverlap(HyPerCol * hc);

int testWeightsEqual(HyPerConn * conn1, HyPerConn * conn2);
int testPatchesEqual(PVPatch * patch1, PVPatch * patch2, int index);
int dumpWeights(KernelConn * kconn, FILE * stream);

int parse_getopt_str_extension(int argc, char * argv[], const char * opt, char ** sVal);

int main(int argc, char * argv[]) {
    // create the managing hypercolumn
    //

    char * routinename;
    char * paramsfilename;

#define NUMROUTINES 6
    const char * allowedroutinenames[] = {
            "setupThreeLayerGenerativeModel",
            "testCheckPatchSize",
            "transposeOfTranspose",
            "mnistTrain",
            "mnistTest",
            "mnistNoOverlap"
    };  // NUMROUTINES is the number of entries in this array

    const char * paramsfiles[] = {
            "input/params-generativelateral.pv",
            "input/params-testcheckPatchSize.pv",
            "input/params-testTransposeOfTranspose.pv",
            "input/params-mnist-train.pv",
            "input/params-mnist-test.pv",
            "input/params-mnist-no-overlap.pv"
    };

    int status = parse_getopt_str_extension(argc, argv, "--routine", &routinename);
    int choice = -1;
    if( status == 0 ) {
        for( int k=0; k<NUMROUTINES; k++ ) {
            if( !strcmp(routinename, allowedroutinenames[k]) ) {
                choice = k;
                break;
            }
        }
    }

    if( choice < 0 ) {
        printf("Please specify a routine using the --routine option.\n");
        printf("Acceptable routines are:\n");
        for(int k=0; k<NUMROUTINES; k++) {
            printf("%2d. %s\n", k, allowedroutinenames[k]);
        }
        printf("----\n");
        printf("Enter your selection as a number from 0 to %d: ", NUMROUTINES-1);
        std::cin >> choice;
        if( choice < 0 || choice >= NUMROUTINES ) {
            fprintf(stderr, "\"%d\" is out of range.  Exiting.\n", choice);
            exit(EXIT_FAILURE);
        }
        printf("You picked %d.  Excellent choice.\n", choice);
    }

    const char * optionp = "-p";
    int paramsfileindex = parse_getopt_str_extension(argc, argv, optionp, &paramsfilename);
    int myargc = argc + ( paramsfileindex < 0 ? 2 : 0 );
    char ** myargv = (char **) malloc((size_t) (myargc+1) * sizeof(char *) );
    assert(myargv != NULL);
    for(int k=0; k<argc; k++) myargv[k] = argv[k];
    if(paramsfileindex == -1 ) {
        assert( myargc == argc+2 );
        myargv[argc] = (char *) malloc((strlen(optionp)+1)*sizeof(char));
        assert(myargv[argc]);
        strcpy(myargv[argc], optionp);
        paramsfileindex = argc+1;
    }
    else if ( paramsfileindex >= argc) {
        fprintf(stderr, "paramsfileindex should be less than argc, but paramsfileindex=%d and argc=%d.  Aborting.\n", paramsfileindex, argc);
        exit(EXIT_FAILURE);
    }
    // Now paramsfileindex is in the range [0,myargc) and points to the slot right after "-p"
    myargv[paramsfileindex] = (char *) malloc((strlen(paramsfiles[choice])+1)*sizeof(char));
    assert(myargv[paramsfileindex]);
    strcpy(myargv[paramsfileindex], paramsfiles[choice]);
    myargv[myargc] = NULL;

    HyPerCol * hc = new HyPerCol("column", myargc, myargv);
    switch( choice ) {
    case 0:
        setupThreeLayerGenerativeModel(hc);
        break;
    case 1:
        testCheckPatchSize(hc);
        break;
    case 2:
#define DUMPWEIGHTSFILE "output/dumpweights.txt"
        status = testTransposeOfTranspose(hc, DUMPWEIGHTSFILE);
        if( status != EXIT_SUCCESS ) {
            fprintf(stderr, "testTransposeOfTranspose failed.\n");
            exit(status);
        }
        break;
    case 3:
        status = mnistTrain(hc);
        if( status != EXIT_SUCCESS ) {
            fprintf(stderr, "mnistTrain error.\n");
            exit(status);
        }
        break;
    case 4:
        status = mnistTest(hc);
        if( status != EXIT_SUCCESS ) {
            fprintf(stderr, "mnistTest error.\n");
            exit(status);
        }
        break;
    case 5:
        status = mnistNoOverlap(hc);
        if( status != EXIT_SUCCESS ) {
            fprintf(stderr, "mnistNoOverlap error.\n");
            exit(status);
        }
        break;
    }

    hc->run();

    /* clean up (HyPerCol owns layers and connections, don't delete them) */
    delete hc;

    return EXIT_SUCCESS;
}

int setupThreeLayerGenerativeModel(HyPerCol * hc) {

    /* This routine sets up a generative model with three layers:
     * A retina, a generative layer A, and a generative layer B.
     * There are feedforward connections from R to A to B.
     * There are feedback connections from B to A.
     * There are lateral connections within A.
     * To set this up in PetaVision, auxiliary layers
     * must be defined:  the "AnaRetina" between the retina and layer A;
     * the "ParaLayer A" beside layer A;
     * and the "AnaLayer A" between Layer A and Layer B.
     * A feedback connection from A to the AnaRetina provides A'a,
     * a feedforward connection from the retina to the Anaretina provides r,
     * so that the AnaRetina collects r-A'a.  A feedforward connection to
     * layer A then provides A(r-A'a).
     * In a similar fashion, AnaLayer A collects (a-B'b) and a feedforward
     * connection to Layer B provides B(a-B'b).
     * A lateral connection from A to ParaLayer A provides (I-A~)a to the
     * ParaLayer A, and then a feedback connection provides (I-A~)(I-A~)a
     * to layer A.
     */

    GenColProbe * hcprobe = new GenColProbe();
    hc->insertProbe(hcprobe);
    PVParams * params = hc->parameters();

    // Layers

    const char * fileOfFileNames = params->getFilename("ImageFileList");
    if( !fileOfFileNames ) {
        fprintf(stderr, "No ImageFileList was defined in parameters file\n");
        delete hc;
        return EXIT_FAILURE;
    }
    const char * outputDir = params->getFilename("OutputDir");
    if( !outputDir ) {
        outputDir = OUTPUT_PATH;
        fprintf(stderr, "No OutputDir was defined in parameters file; using %s\n",outputDir);
    }

    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames);

    Retina * retina = new Retina("Retina", hc);
    L2NormProbe * l2norm_retina = new L2NormProbe("Retina      :");
    retina->insertProbe(l2norm_retina);

    NonspikingLayer * anaretina = new NonspikingLayer("AnaRetina", hc);
    L2NormProbe * l2norm_anaretina = new L2NormProbe("AnaRetina   :");
    anaretina->insertProbe(l2norm_anaretina);
    hcprobe->addTerm(l2norm_anaretina, anaretina);

//    ChannelProbe * anaretina_exc = new ChannelProbe("anaretina_exc.txt", CHANNEL_EXC);
//    anaretina->insertProbe(anaretina_exc);
//    ChannelProbe * anaretina_inh = new ChannelProbe("anaretina_inh.txt", CHANNEL_INH);
//    anaretina->insertProbe(anaretina_inh);

    GenerativeLayer * layerA = new GenerativeLayer("Layer A", hc);
    L2NormProbe * l2norm_layerA = new L2NormProbe("Layer A     :");
    SparsityTermProbe * sparsity_layerA = new SparsityTermProbe("Layer A     :");
    layerA->insertProbe(l2norm_layerA);
    layerA->insertProbe(sparsity_layerA);
    hcprobe->addTerm(sparsity_layerA, layerA);

//    ChannelProbe * layerA_exc = new ChannelProbe("layerA_exc.txt", CHANNEL_EXC);
//    layerA->insertProbe(layerA_exc);
//    ChannelProbe * layerA_inh = new ChannelProbe("layerA_inh.txt", CHANNEL_INH);
//    layerA->insertProbe(layerA_inh);

    NonspikingLayer * paralayerA = new NonspikingLayer("ParaLayer A", hc);
    L2NormProbe * l2norm_paralayerA = new L2NormProbe("ParaLayer A :");
    paralayerA->insertProbe(l2norm_paralayerA);
    hcprobe->addTerm(l2norm_paralayerA, paralayerA);

//    ChannelProbe * paralayerA_exc = new ChannelProbe("paralayerA_exc.txt", CHANNEL_EXC);
//    paralayerA->insertProbe(paralayerA_exc);
//    ChannelProbe * paralayerA_inh = new ChannelProbe("paralayerA_inh.txt", CHANNEL_INH);
//    paralayerA->insertProbe(paralayerA_inh);

    NonspikingLayer * analayerA = new NonspikingLayer("AnaLayer A", hc);
    L2NormProbe * l2norm_analayerA = new L2NormProbe("AnaLayer A  :");
    analayerA->insertProbe(l2norm_analayerA);
    hcprobe->addTerm(l2norm_analayerA, analayerA);

//    ChannelProbe * analayerA_exc = new ChannelProbe("analayerA_exc.txt", CHANNEL_EXC);
//    analayerA->insertProbe(analayerA_exc);
//    ChannelProbe * analayerA_inh = new ChannelProbe("analayerA_inh.txt", CHANNEL_INH);
//    analayerA->insertProbe(analayerA_inh);

    GenerativeLayer * layerB = new GenerativeLayer("Layer B", hc);
    L2NormProbe * l2norm_layerB = new L2NormProbe("Layer B     :");
    SparsityTermProbe * sparsity_layerB = new SparsityTermProbe("Layer B     :");
    layerB->insertProbe(l2norm_layerB);
    layerB->insertProbe(sparsity_layerB);
    hcprobe->addTerm(sparsity_layerB, layerB);

//    ChannelProbe * layerB_exc = new ChannelProbe("layerB_exc.txt", CHANNEL_EXC);
//    layerB->insertProbe(layerB_exc);
//    ChannelProbe * layerB_inh = new ChannelProbe("layerB_inh.txt", CHANNEL_INH);
//    layerB->insertProbe(layerB_inh);

    // Connections
    KernelConn * slideshow_retina = new KernelConn("Slideshow to Retina", hc, slideshow, retina, CHANNEL_EXC);
    assert(slideshow_retina);
    IdentConn * retina_anaretina = new IdentConn("Retina to AnaRetina", hc, retina, anaretina, CHANNEL_EXC);
    assert(retina_anaretina);
    GenerativeConn * anaretina_layerA = new GenerativeConn("AnaRetina to Layer A", hc, anaretina, layerA, CHANNEL_EXC);
    assert(anaretina_layerA);
    FeedbackConn * layerA_anaretinaFB = new FeedbackConn("Layer A to AnaRetina Feedback", hc, CHANNEL_INH, anaretina_layerA);
    assert(layerA_anaretinaFB);
    LateralConn * layerA_paralayerA = new LateralConn("Layer A to ParaLayer A", hc, layerA, paralayerA, CHANNEL_EXC);
    assert(layerA_paralayerA);
    FeedbackConn * paralayerA_layerAFB = new FeedbackConn("ParaLayer A to Layer A Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerA_layerAFB);
    IdentConn * layerA_analayerA = new IdentConn("Layer A to AnaLayer A", hc, layerA, analayerA, CHANNEL_EXC);
    assert(layerA_analayerA);
    IdentConn * analayerA_layerAFB = new IdentConn("AnaLayer A to Layer A Feedback", hc, analayerA, layerA, CHANNEL_INH);
    assert(analayerA_layerAFB);
    GenerativeConn * analayerA_layerB = new GenerativeConn("AnaLayer A to Layer B", hc, analayerA, layerB, CHANNEL_EXC);
    assert(analayerA_layerB);
    FeedbackConn * layerB_analayerAFB = new FeedbackConn("Layer B to AnaLayer A Feedback", hc, CHANNEL_INH, analayerA_layerB);
    assert(layerB_analayerAFB);

    return EXIT_SUCCESS;
}

int testCheckPatchSize(HyPerCol * hc) {

    GenColProbe * hcprobe = new GenColProbe();
    hc->insertProbe(hcprobe);

    // create the visualization display
    //
    //GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);

    PVParams * params = hc->parameters();

    const char * fileOfFileNames = params->getFilename("ImageFileList");
    if( !fileOfFileNames ) {
        fprintf(stderr, "No ImageFileList was defined in parameters file\n");
        delete hc;
        return EXIT_FAILURE;
    }
    const char * outputDir = params->getFilename("OutputDir");
    if( !outputDir ) {
        outputDir = OUTPUT_PATH;
        fprintf(stderr, "No OutputDir was defined in parameters file; using %s\n",outputDir);
    }

    // Layers
    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames);
    Retina * retina = new Retina("Retina", hc);
    GenerativeLayer * layerA = new GenerativeLayer("Layer A", hc);

    // Connections
    KernelConn * slideshow_retina = new KernelConn("Slideshow to Retina", hc, slideshow, retina, CHANNEL_EXC);
    assert(slideshow_retina);
    KernelConn * retina_layerA = new KernelConn("Retina to Layer A", hc, retina, layerA, CHANNEL_EXC);
    assert(retina_layerA);

    return EXIT_SUCCESS;
}

int testTransposeOfTranspose(HyPerCol * hc, const char * dumpweightsfile) {
    // This should be moved to the tests directory of petavision once the
    // generative classes are moved there.

    GenColProbe * hcprobe = new GenColProbe();
    hc->insertProbe(hcprobe);

    PVParams * params = hc->parameters();

    const char * fileOfFileNames = params->getFilename("ImageFileList");
    if( !fileOfFileNames ) {
        fprintf(stderr, "No ImageFileList was defined in parameters file\n");
        delete hc;
        return EXIT_FAILURE;
    }
    const char * outputDir = params->getFilename("OutputDir");
    if( !outputDir ) {
        outputDir = OUTPUT_PATH;
        fprintf(stderr, "No OutputDir was defined in parameters file; using %s\n",outputDir);
    }
    FILE * dumpWeightStream;
    if( dumpweightsfile != NULL ) {
        dumpWeightStream = fopen(dumpweightsfile, "w");
        assert(dumpWeightStream);
    }
    else {
        dumpWeightStream = stdout;
    }

    // Layers
    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames);
    Retina * retina = new Retina("Retina", hc);
    NonspikingLayer * layerA = new NonspikingLayer("Layer A", hc);
    NonspikingLayer * layerB1to1 = new NonspikingLayer("Layer B One to one", hc);
    NonspikingLayer * layerBManyTo1 = new NonspikingLayer("Layer B Many to one", hc);
    NonspikingLayer * layerB1toMany = new NonspikingLayer("Layer B One to many", hc);

    // Connections
    KernelConn * slideshow_retina = new KernelConn("Slideshow to Retina", hc, slideshow, retina, CHANNEL_EXC);
    assert(slideshow_retina);
    KernelConn * retina_layerA = new KernelConn("Retina to Layer A", hc, retina, layerA, CHANNEL_EXC);
    assert(retina_layerA);
    GenerativeConn * originalMap1to1 = new GenerativeConn("One to one original map", hc, layerA, layerB1to1, CHANNEL_EXC);
    assert(originalMap1to1);
    FeedbackConn * transpose1to1 = new FeedbackConn("One to one transpose", hc, CHANNEL_INHB, originalMap1to1);
    assert(transpose1to1);
    FeedbackConn * transposeOfTranspose1to1 = new FeedbackConn("One to one transpose of transpose", hc, CHANNEL_INHB, transpose1to1);
    assert(transposeOfTranspose1to1);

    int status1to1 = testWeightsEqual(originalMap1to1, transposeOfTranspose1to1);
    if(status1to1 == EXIT_SUCCESS) {
        printf("One-to-one case: testTransposeOfTranspose passed.\n");
    }
    else {
        fprintf(stderr, "One-to-one case: testTransposeOfTranspose failed; dumping weights to ");
        dumpweightsfile != NULL ? fprintf(stderr, "%s\n", dumpweightsfile) :
                          fprintf(stderr, "standard output\n");
        dumpWeights(originalMap1to1, dumpWeightStream);
        dumpWeights(transpose1to1, dumpWeightStream);
        dumpWeights(transposeOfTranspose1to1, dumpWeightStream);
    }

    GenerativeConn * originalMapManyTo1 = new GenerativeConn("Many to one original map", hc, layerA, layerBManyTo1, CHANNEL_EXC);
    assert(originalMapManyTo1);
    FeedbackConn * transposeManyTo1 = new FeedbackConn("Many to one transpose", hc, CHANNEL_INHB, originalMapManyTo1);
    assert(transposeManyTo1);
    FeedbackConn * transposeOfTransposeManyTo1 = new FeedbackConn("Many to one transpose of transpose", hc, CHANNEL_INHB, transposeManyTo1);
    assert(transposeOfTransposeManyTo1);

    int statusManyTo1 = testWeightsEqual(originalMapManyTo1, transposeOfTransposeManyTo1);
    if(statusManyTo1 == EXIT_SUCCESS) {
        printf("Many-to-one case: testTransposeOfTranspose passed.\n");
    }
    else {
        fprintf(stderr, "Many-to-one case: testTransposeOfTranspose failed; dumping weights to ");
        dumpweightsfile != NULL ? fprintf(stderr, "%s\n", dumpweightsfile) :
                          fprintf(stderr, "standard output\n");
        dumpWeights(originalMapManyTo1, dumpWeightStream);
        dumpWeights(transposeManyTo1, dumpWeightStream);
        dumpWeights(transposeOfTransposeManyTo1, dumpWeightStream);
    }

    GenerativeConn * originalMap1toMany = new GenerativeConn("One to many original map", hc, layerA, layerB1toMany, CHANNEL_EXC);
    assert(originalMap1toMany);
    FeedbackConn * transpose1toMany = new FeedbackConn("One to many transpose", hc, CHANNEL_INHB, originalMap1toMany);
    assert(transpose1toMany);
    FeedbackConn * transposeOfTranspose1toMany = new FeedbackConn("One to many transpose of transpose", hc, CHANNEL_INHB, transpose1toMany);
    assert(transposeOfTranspose1toMany);

    int status1toMany = testWeightsEqual(originalMap1toMany, transposeOfTranspose1toMany);
    if(status1toMany == EXIT_SUCCESS) {
        printf("One-to-many case: testTransposeOfTranspose passed.\n");
    }
    else {
        fprintf(stderr, "One-to-many case: testTransposeOfTranspose failed; dumping weights to ");
        dumpweightsfile != NULL ? fprintf(stderr, "%s\n", dumpweightsfile) :
                          fprintf(stderr, "standard output\n");
        dumpWeights(originalMap1toMany, dumpWeightStream);
        dumpWeights(transpose1toMany, dumpWeightStream);
        dumpWeights(transposeOfTranspose1toMany, dumpWeightStream);
    }

    if( dumpWeightStream != NULL && dumpWeightStream != stdout ) fclose( dumpWeightStream );
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

int mnistTrain(HyPerCol * hc) {

    GenColProbe * hcprobe = new GenColProbe();
    hc->insertProbe(hcprobe);
    PVParams * params = hc->parameters();

    const char * fileOfFileNames = params->getFilename("ImageFileList");
    if( !fileOfFileNames ) {
        fprintf(stderr, "No ImageFileList was defined in parameters file\n");
        delete hc;
        return EXIT_FAILURE;
    }
    const char * outputDir = params->getFilename("OutputDir");
    if( !outputDir ) {
        outputDir = OUTPUT_PATH;
        fprintf(stderr, "No OutputDir was defined in parameters file; using %s\n",outputDir);
    }

    // Layers
    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames);

    Retina * retina = new Retina("Retina", hc);
    L2NormProbe * l2norm_retina = new L2NormProbe("Retina      :");
    retina->insertProbe(l2norm_retina);

    NonspikingLayer * anaretina = new NonspikingLayer("AnaRetina", hc);
    L2NormProbe * l2norm_anaretina = new L2NormProbe("AnaRetina   :");
    anaretina->insertProbe(l2norm_anaretina);
    hcprobe->addTerm(l2norm_anaretina, anaretina);

    GenerativeLayer * layerA = new GenerativeLayer("Layer A", hc);
    L2NormProbe * l2norm_layerA = new L2NormProbe("Layer A     :");
    SparsityTermProbe * sparsity_layerA = new SparsityTermProbe("Layer A     :");
    layerA->insertProbe(l2norm_layerA);
    layerA->insertProbe(sparsity_layerA);
    hcprobe->addTerm(sparsity_layerA, layerA);

    NonspikingLayer * paralayerA = new NonspikingLayer("ParaLayer A", hc);
    L2NormProbe * l2norm_paralayerA = new L2NormProbe("ParaLayer A :");
    paralayerA->insertProbe(l2norm_paralayerA);
    hcprobe->addTerm(l2norm_paralayerA, paralayerA);

    NonspikingLayer * analayerA = new NonspikingLayer("AnaLayer A", hc);
    L2NormProbe * l2norm_analayerA = new L2NormProbe("AnaLayer A  :");
    analayerA->insertProbe(l2norm_analayerA);
    hcprobe->addTerm(l2norm_analayerA, analayerA);

    GenerativeLayer * layerB = new GenerativeLayer("Layer B", hc);
    L2NormProbe * l2norm_layerB = new L2NormProbe("Layer B     :");
    SparsityTermProbe * sparsity_layerB = new SparsityTermProbe("Layer B     :");
    layerB->insertProbe(l2norm_layerB);
    layerB->insertProbe(sparsity_layerB);
    hcprobe->addTerm(sparsity_layerB, layerB);

    NonspikingLayer * paralayerB = new NonspikingLayer("ParaLayer B", hc);
    L2NormProbe * l2norm_paralayerB = new L2NormProbe("ParaLayer B :");
    paralayerB->insertProbe(l2norm_paralayerB);
    hcprobe->addTerm(l2norm_paralayerB, paralayerB);

    NonspikingLayer * analayerB = new NonspikingLayer("AnaLayer B", hc);
    L2NormProbe * l2norm_analayerB = new L2NormProbe("AnaLayer B  :");
    analayerB->insertProbe(l2norm_analayerB);
    hcprobe->addTerm(l2norm_analayerB, analayerB);

    GenerativeLayer * layerC = new GenerativeLayer("Layer C", hc);
    L2NormProbe * l2norm_layerC = new L2NormProbe("Layer C     :");
    SparsityTermProbe * sparsity_layerC = new SparsityTermProbe("Layer C     :");
    layerC->insertProbe(l2norm_layerC);
    layerC->insertProbe(sparsity_layerC);
    hcprobe->addTerm(sparsity_layerC, layerC);

    NonspikingLayer * paralayerC = new NonspikingLayer("ParaLayer C", hc);
    L2NormProbe * l2norm_paralayerC = new L2NormProbe("ParaLayer C :");
    paralayerC->insertProbe(l2norm_paralayerC);
    hcprobe->addTerm(l2norm_paralayerC, paralayerC);

    NonspikingLayer * analayerC = new NonspikingLayer("AnaLayer C", hc);
    L2NormProbe * l2norm_analayerC = new L2NormProbe("AnaLayer C  :");
    analayerC->insertProbe(l2norm_analayerC);
    hcprobe->addTerm(l2norm_analayerC, analayerC);

    float displayPeriod = hc->parameters()->value("Slideshow", "displayPeriod");
    TrainingGenLayer * traininglayer = new TrainingGenLayer("IT", hc, "input/mnist/train/trainlabels.txt", displayPeriod, 3.0f);

    // Connections
    KernelConn * slideshow_retina = new KernelConn("Slideshow to Retina", hc, slideshow, retina, CHANNEL_EXC);
    assert(slideshow_retina);
    IdentConn * retina_anaretina = new IdentConn("Retina to AnaRetina", hc, retina, anaretina, CHANNEL_EXC);
    assert(retina_anaretina);
    GenerativeConn * anaretina_layerA = new GenerativeConn("AnaRetina to Layer A", hc, anaretina, layerA, CHANNEL_EXC);
    assert(anaretina_layerA);
    FeedbackConn * layerA_anaretinaFB = new FeedbackConn("Layer A to AnaRetina Feedback", hc, CHANNEL_INH, anaretina_layerA);
    assert(layerA_anaretinaFB);
    LateralConn * layerA_paralayerA = new LateralConn("Layer A to ParaLayer A", hc, layerA, paralayerA, CHANNEL_EXC);
    assert(layerA_paralayerA);
    FeedbackConn * paralayerA_layerAFB = new FeedbackConn("ParaLayer A to Layer A Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerA_layerAFB);
    IdentConn * layerA_analayerA = new IdentConn("Layer A to AnaLayer A", hc, layerA, analayerA, CHANNEL_EXC);
    assert(layerA_analayerA);
    IdentConn * analayerA_layerAFB = new IdentConn("AnaLayer A to Layer A Feedback", hc, analayerA, layerA, CHANNEL_INH);
    assert(analayerA_layerAFB);
    GenerativeConn * analayerA_layerB = new GenerativeConn("AnaLayer A to Layer B", hc, analayerA, layerB, CHANNEL_EXC);
    assert(analayerA_layerB);
    FeedbackConn * layerB_analayerAFB = new FeedbackConn("Layer B to AnaLayer A Feedback", hc, CHANNEL_INH, analayerA_layerB);
    assert(layerB_analayerAFB);
    LateralConn * layerB_paralayerB = new LateralConn("Layer B to ParaLayer B", hc, layerB, paralayerB, CHANNEL_EXC);
    assert(layerB_paralayerB);
    FeedbackConn * paralayerB_layerBFB = new FeedbackConn("ParaLayer B to Layer B Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerB_layerBFB);
    IdentConn * layerB_analayerB = new IdentConn("Layer B to AnaLayer B", hc, layerB, analayerB, CHANNEL_EXC);
    assert(layerB_analayerB);
    IdentConn * analayerB_layerBFB = new IdentConn("AnaLayer B to Layer B Feedback", hc, analayerB, layerB, CHANNEL_INH);
    assert(analayerB_layerBFB);
    GenerativeConn * analayerB_layerC = new GenerativeConn("AnaLayer B to Layer C", hc, analayerB, layerC, CHANNEL_EXC);
    assert(analayerB_layerC);
    FeedbackConn * traininglayer_analayerBFB = new FeedbackConn("Layer C to AnaLayer B Feedback", hc, CHANNEL_INH, analayerB_layerC);
    assert(traininglayer_analayerBFB);

    LateralConn * layerC_paralayerC = new LateralConn("Layer C to ParaLayer C", hc, layerC, paralayerC, CHANNEL_EXC);
    assert(layerC_paralayerC);
    FeedbackConn * paralayerC_layerCFB = new FeedbackConn("ParaLayer C to Layer C Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerC_layerCFB);
    IdentConn * layerC_analayerC = new IdentConn("Layer C to AnaLayer C", hc, layerC, analayerC, CHANNEL_EXC);
    assert(layerC_analayerC);
    IdentConn * analayerC_layerCFB = new IdentConn("AnaLayer C to Layer C Feedback", hc, analayerC, layerC, CHANNEL_INH);
    assert(analayerC_layerCFB);
    GenerativeConn * analayerC_traininglayer = new GenerativeConn("AnaLayer C to IT", hc, analayerC, traininglayer, CHANNEL_EXC);
    assert(analayerC_traininglayer);
    FeedbackConn * traininglayer_analayerCFB = new FeedbackConn("IT to AnaLayer C Feedback", hc, CHANNEL_INH, analayerC_traininglayer);
    assert(traininglayer_analayerCFB);

    return EXIT_SUCCESS;

}

int mnistTest(HyPerCol * hc) {

    GenColProbe * hcprobe = new GenColProbe();
    hc->insertProbe(hcprobe);
    PVParams * params = hc->parameters();

    const char * fileOfFileNames = params->getFilename("ImageFileList");
    if( !fileOfFileNames ) {
        fprintf(stderr, "No ImageFileList was defined in parameters file\n");
        delete hc;
        return EXIT_FAILURE;
    }
    const char * outputDir = params->getFilename("OutputDir");
    if( !outputDir ) {
        outputDir = OUTPUT_PATH;
        fprintf(stderr, "No OutputDir was defined in parameters file; using %s\n",outputDir);
    }

    // Layers
    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames);

    Retina * retina = new Retina("Retina", hc);
    L2NormProbe * l2norm_retina = new L2NormProbe("Retina      :");
    retina->insertProbe(l2norm_retina);

    NonspikingLayer * anaretina = new NonspikingLayer("AnaRetina", hc);
    L2NormProbe * l2norm_anaretina = new L2NormProbe("AnaRetina   :");
    anaretina->insertProbe(l2norm_anaretina);
    hcprobe->addTerm(l2norm_anaretina, anaretina);

    GenerativeLayer * layerA = new GenerativeLayer("Layer A", hc);
    L2NormProbe * l2norm_layerA = new L2NormProbe("Layer A     :");
    SparsityTermProbe * sparsity_layerA = new SparsityTermProbe("Layer A     :");
    layerA->insertProbe(l2norm_layerA);
    layerA->insertProbe(sparsity_layerA);
    hcprobe->addTerm(sparsity_layerA, layerA);

    NonspikingLayer * paralayerA = new NonspikingLayer("ParaLayer A", hc);
    L2NormProbe * l2norm_paralayerA = new L2NormProbe("ParaLayer A :");
    paralayerA->insertProbe(l2norm_paralayerA);
    hcprobe->addTerm(l2norm_paralayerA, paralayerA);

    NonspikingLayer * analayerA = new NonspikingLayer("AnaLayer A", hc);
    L2NormProbe * l2norm_analayerA = new L2NormProbe("AnaLayer A  :");
    analayerA->insertProbe(l2norm_analayerA);
    hcprobe->addTerm(l2norm_analayerA, analayerA);

    GenerativeLayer * layerB = new GenerativeLayer("Layer B", hc);
    L2NormProbe * l2norm_layerB = new L2NormProbe("Layer B     :");
    SparsityTermProbe * sparsity_layerB = new SparsityTermProbe("Layer B     :");
    layerB->insertProbe(l2norm_layerB);
    layerB->insertProbe(sparsity_layerB);
    hcprobe->addTerm(sparsity_layerB, layerB);

    NonspikingLayer * paralayerB = new NonspikingLayer("ParaLayer B", hc);
    L2NormProbe * l2norm_paralayerB = new L2NormProbe("ParaLayer B :");
    paralayerB->insertProbe(l2norm_paralayerB);
    hcprobe->addTerm(l2norm_paralayerB, paralayerB);

    NonspikingLayer * analayerB = new NonspikingLayer("AnaLayer B", hc);
    L2NormProbe * l2norm_analayerB = new L2NormProbe("AnaLayer B  :");
    analayerB->insertProbe(l2norm_analayerB);
    hcprobe->addTerm(l2norm_analayerB, analayerB);

    GenerativeLayer * layerC = new GenerativeLayer("Layer C", hc);
    L2NormProbe * l2norm_layerC = new L2NormProbe("Layer C     :");
    SparsityTermProbe * sparsity_layerC = new SparsityTermProbe("Layer C     :");
    layerC->insertProbe(l2norm_layerC);
    layerC->insertProbe(sparsity_layerC);
    hcprobe->addTerm(sparsity_layerC, layerC);

    NonspikingLayer * paralayerC = new NonspikingLayer("ParaLayer C", hc);
    L2NormProbe * l2norm_paralayerC = new L2NormProbe("ParaLayer C :");
    paralayerC->insertProbe(l2norm_paralayerC);
    hcprobe->addTerm(l2norm_paralayerC, paralayerC);

    NonspikingLayer * analayerC = new NonspikingLayer("AnaLayer C", hc);
    L2NormProbe * l2norm_analayerC = new L2NormProbe("AnaLayer C  :");
    analayerC->insertProbe(l2norm_analayerC);
    hcprobe->addTerm(l2norm_analayerC, analayerC);

    GenerativeLayer * itLayer = new GenerativeLayer("IT", hc);
    VProbe * vprobe_it = new VProbe();
    itLayer->insertProbe(vprobe_it);

    // Connections
    KernelConn * slideshow_retina = new KernelConn("Slideshow to Retina", hc, slideshow, retina, CHANNEL_EXC);
    assert(slideshow_retina);
    IdentConn * retina_anaretina = new IdentConn("Retina to AnaRetina", hc, retina, anaretina, CHANNEL_EXC);
    assert(retina_anaretina);
    GenerativeConn * anaretina_layerA = new GenerativeConn("AnaRetina to Layer A", hc, anaretina, layerA, CHANNEL_EXC);
    assert(anaretina_layerA);
    FeedbackConn * layerA_anaretinaFB = new FeedbackConn("Layer A to AnaRetina Feedback", hc, CHANNEL_INH, anaretina_layerA);
    assert(layerA_anaretinaFB);
    LateralConn * layerA_paralayerA = new LateralConn("Layer A to ParaLayer A", hc, layerA, paralayerA, CHANNEL_EXC);
    assert(layerA_paralayerA);
    FeedbackConn * paralayerA_layerAFB = new FeedbackConn("ParaLayer A to Layer A Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerA_layerAFB);
    IdentConn * layerA_analayerA = new IdentConn("Layer A to AnaLayer A", hc, layerA, analayerA, CHANNEL_EXC);
    assert(layerA_analayerA);
    IdentConn * analayerA_layerAFB = new IdentConn("AnaLayer A to Layer A Feedback", hc, analayerA, layerA, CHANNEL_INH);
    assert(analayerA_layerAFB);
    GenerativeConn * analayerA_layerB = new GenerativeConn("AnaLayer A to Layer B", hc, analayerA, layerB, CHANNEL_EXC);
    assert(analayerA_layerB);
    FeedbackConn * layerB_analayerAFB = new FeedbackConn("Layer B to AnaLayer A Feedback", hc, CHANNEL_INH, analayerA_layerB);
    assert(layerB_analayerAFB);
    LateralConn * layerB_paralayerB = new LateralConn("Layer B to ParaLayer B", hc, layerB, paralayerB, CHANNEL_EXC);
    assert(layerB_paralayerB);
    FeedbackConn * paralayerB_layerBFB = new FeedbackConn("ParaLayer B to Layer B Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerB_layerBFB);
    IdentConn * layerB_analayerB = new IdentConn("Layer B to AnaLayer B", hc, layerB, analayerB, CHANNEL_EXC);
    assert(layerB_analayerB);
    IdentConn * analayerB_layerBFB = new IdentConn("AnaLayer B to Layer B Feedback", hc, analayerB, layerB, CHANNEL_INH);
    assert(analayerB_layerBFB);
    GenerativeConn * analayerB_layerC = new GenerativeConn("AnaLayer B to Layer C", hc, analayerB, layerC, CHANNEL_EXC);
    assert(analayerB_layerC);
    FeedbackConn * traininglayer_analayerBFB = new FeedbackConn("Layer C to AnaLayer B Feedback", hc, CHANNEL_INH, analayerB_layerC);
    assert(traininglayer_analayerBFB);

    LateralConn * layerC_paralayerC = new LateralConn("Layer C to ParaLayer C", hc, layerC, paralayerC, CHANNEL_EXC);
    assert(layerC_paralayerC);
    FeedbackConn * paralayerC_layerCFB = new FeedbackConn("ParaLayer C to Layer C Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerC_layerCFB);
    IdentConn * layerC_analayerC = new IdentConn("Layer C to AnaLayer C", hc, layerC, analayerC, CHANNEL_EXC);
    assert(layerC_analayerC);
    IdentConn * analayerC_layerCFB = new IdentConn("AnaLayer C to Layer C Feedback", hc, analayerC, layerC, CHANNEL_INH);
    assert(analayerC_layerCFB);
    GenerativeConn * analayerC_it = new GenerativeConn("AnaLayer C to IT", hc, analayerC, itLayer, CHANNEL_EXC);
    assert(analayerC_it);
    FeedbackConn * it_analayerCFB = new FeedbackConn("IT to AnaLayer C Feedback", hc, CHANNEL_INH, analayerC_it);
    assert(it_analayerCFB);

    return EXIT_SUCCESS;

}

int parse_getopt_str_extension(int argc, char * argv[], const char * opt, char ** sVal) {
      for ( int i = 1; i < argc-1; i++ ) {
         if ( !strcmp(argv[i], opt) ) {
            *sVal = argv[i+1];
            return i+1;  // return the index such that argv[i] is the value of the desired option
         }
      }
      return -1;  // not found
}

int mnistNoOverlap(HyPerCol * hc) {

    GenColProbe * hcprobe = new GenColProbe();
    hc->insertProbe(hcprobe);
    PVParams * params = hc->parameters();

    const char * fileOfFileNames = params->getFilename("ImageFileList");
    if( !fileOfFileNames ) {
        fprintf(stderr, "No ImageFileList was defined in parameters file\n");
        delete hc;
        return EXIT_FAILURE;
    }
    const char * outputDir = params->getFilename("OutputDir");
    if( !outputDir ) {
        outputDir = OUTPUT_PATH;
        fprintf(stderr, "No OutputDir was defined in parameters file; using %s\n",outputDir);
    }

    // Layers
    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames);

    Retina * retina = new Retina("Retina", hc);
    L2NormProbe * l2norm_retina = new L2NormProbe("Retina      :");
    retina->insertProbe(l2norm_retina);

    NonspikingLayer * anaretina = new NonspikingLayer("AnaRetina", hc);
    L2NormProbe * l2norm_anaretina = new L2NormProbe("AnaRetina   :");
    anaretina->insertProbe(l2norm_anaretina);
    hcprobe->addTerm(l2norm_anaretina, anaretina);

    GenerativeLayer * layerA = new GenerativeLayer("Layer A", hc);
    L2NormProbe * l2norm_layerA = new L2NormProbe("Layer A     :");
    SparsityTermProbe * sparsity_layerA = new SparsityTermProbe("Layer A     :");
    layerA->insertProbe(l2norm_layerA);
    layerA->insertProbe(sparsity_layerA);
    hcprobe->addTerm(sparsity_layerA, layerA);

    NonspikingLayer * paralayerA = new NonspikingLayer("ParaLayer A", hc);
    L2NormProbe * l2norm_paralayerA = new L2NormProbe("ParaLayer A :");
    paralayerA->insertProbe(l2norm_paralayerA);
    hcprobe->addTerm(l2norm_paralayerA, paralayerA);

    NonspikingLayer * analayerA = new NonspikingLayer("AnaLayer A", hc);
    L2NormProbe * l2norm_analayerA = new L2NormProbe("AnaLayer A  :");
    analayerA->insertProbe(l2norm_analayerA);
    hcprobe->addTerm(l2norm_analayerA, analayerA);

    GenerativeLayer * layerB = new GenerativeLayer("Layer B", hc);
    L2NormProbe * l2norm_layerB = new L2NormProbe("Layer B     :");
    SparsityTermProbe * sparsity_layerB = new SparsityTermProbe("Layer B     :");
    layerB->insertProbe(l2norm_layerB);
    layerB->insertProbe(sparsity_layerB);
    hcprobe->addTerm(sparsity_layerB, layerB);

    NonspikingLayer * paralayerB = new NonspikingLayer("ParaLayer B", hc);
    L2NormProbe * l2norm_paralayerB = new L2NormProbe("ParaLayer B :");
    paralayerB->insertProbe(l2norm_paralayerB);
    hcprobe->addTerm(l2norm_paralayerB, paralayerB);

    NonspikingLayer * analayerB = new NonspikingLayer("AnaLayer B", hc);
    L2NormProbe * l2norm_analayerB = new L2NormProbe("AnaLayer B  :");
    analayerB->insertProbe(l2norm_analayerB);
    hcprobe->addTerm(l2norm_analayerB, analayerB);

    GenerativeLayer * layerC = new GenerativeLayer("Layer C", hc);
    L2NormProbe * l2norm_layerC = new L2NormProbe("Layer C     :");
    SparsityTermProbe * sparsity_layerC = new SparsityTermProbe("Layer C     :");
    layerC->insertProbe(l2norm_layerC);
    layerC->insertProbe(sparsity_layerC);
    hcprobe->addTerm(sparsity_layerC, layerC);

    NonspikingLayer * paralayerC = new NonspikingLayer("ParaLayer C", hc);
    L2NormProbe * l2norm_paralayerC = new L2NormProbe("ParaLayer C :");
    paralayerC->insertProbe(l2norm_paralayerC);
    hcprobe->addTerm(l2norm_paralayerC, paralayerC);

    NonspikingLayer * analayerC = new NonspikingLayer("AnaLayer C", hc);
    L2NormProbe * l2norm_analayerC = new L2NormProbe("AnaLayer C  :");
    analayerC->insertProbe(l2norm_analayerC);
    hcprobe->addTerm(l2norm_analayerC, analayerC);

    float displayPeriod = hc->parameters()->value("Slideshow", "displayPeriod");
    TrainingGenLayer * traininglayer = new TrainingGenLayer("IT", hc, "input/mnist/train/trainlabels.txt", displayPeriod, 3.0f);

    // Connections
    KernelConn * slideshow_retina = new KernelConn("Slideshow to Retina", hc, slideshow, retina, CHANNEL_EXC);
    assert(slideshow_retina);
    IdentConn * retina_anaretina = new IdentConn("Retina to AnaRetina", hc, retina, anaretina, CHANNEL_EXC);
    assert(retina_anaretina);
    GenerativeConn * anaretina_layerA = new GenerativeConn("AnaRetina to Layer A", hc, anaretina, layerA, CHANNEL_EXC);
    assert(anaretina_layerA);
    FeedbackConn * layerA_anaretinaFB = new FeedbackConn("Layer A to AnaRetina Feedback", hc, CHANNEL_INH, anaretina_layerA);
    assert(layerA_anaretinaFB);
    LateralConn * layerA_paralayerA = new LateralConn("Layer A to ParaLayer A", hc, layerA, paralayerA, CHANNEL_EXC);
    assert(layerA_paralayerA);
    FeedbackConn * paralayerA_layerAFB = new FeedbackConn("ParaLayer A to Layer A Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerA_layerAFB);
    IdentConn * layerA_analayerA = new IdentConn("Layer A to AnaLayer A", hc, layerA, analayerA, CHANNEL_EXC);
    assert(layerA_analayerA);
    IdentConn * analayerA_layerAFB = new IdentConn("AnaLayer A to Layer A Feedback", hc, analayerA, layerA, CHANNEL_INH);
    assert(analayerA_layerAFB);
    GenerativeConn * analayerA_layerB = new GenerativeConn("AnaLayer A to Layer B", hc, analayerA, layerB, CHANNEL_EXC);
    assert(analayerA_layerB);
    FeedbackConn * layerB_analayerAFB = new FeedbackConn("Layer B to AnaLayer A Feedback", hc, CHANNEL_INH, analayerA_layerB);
    assert(layerB_analayerAFB);
    LateralConn * layerB_paralayerB = new LateralConn("Layer B to ParaLayer B", hc, layerB, paralayerB, CHANNEL_EXC);
    assert(layerB_paralayerB);
    FeedbackConn * paralayerB_layerBFB = new FeedbackConn("ParaLayer B to Layer B Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerB_layerBFB);
    IdentConn * layerB_analayerB = new IdentConn("Layer B to AnaLayer B", hc, layerB, analayerB, CHANNEL_EXC);
    assert(layerB_analayerB);
    IdentConn * analayerB_layerBFB = new IdentConn("AnaLayer B to Layer B Feedback", hc, analayerB, layerB, CHANNEL_INH);
    assert(analayerB_layerBFB);
    GenerativeConn * analayerB_layerC = new GenerativeConn("AnaLayer B to Layer C", hc, analayerB, layerC, CHANNEL_EXC);
    assert(analayerB_layerC);
    FeedbackConn * traininglayer_analayerBFB = new FeedbackConn("Layer C to AnaLayer B Feedback", hc, CHANNEL_INH, analayerB_layerC);
    assert(traininglayer_analayerBFB);

    LateralConn * layerC_paralayerC = new LateralConn("Layer C to ParaLayer C", hc, layerC, paralayerC, CHANNEL_EXC);
    assert(layerC_paralayerC);
    FeedbackConn * paralayerC_layerCFB = new FeedbackConn("ParaLayer C to Layer C Feedback", hc, CHANNEL_INH, layerA_paralayerA);
    assert(paralayerC_layerCFB);
    IdentConn * layerC_analayerC = new IdentConn("Layer C to AnaLayer C", hc, layerC, analayerC, CHANNEL_EXC);
    assert(layerC_analayerC);
    IdentConn * analayerC_layerCFB = new IdentConn("AnaLayer C to Layer C Feedback", hc, analayerC, layerC, CHANNEL_INH);
    assert(analayerC_layerCFB);
    GenerativeConn * analayerC_traininglayer = new GenerativeConn("AnaLayer C to IT", hc, analayerC, traininglayer, CHANNEL_EXC);
    assert(analayerC_traininglayer);
    FeedbackConn * traininglayer_analayerCFB = new FeedbackConn("IT to AnaLayer C Feedback", hc, CHANNEL_INH, analayerC_traininglayer);
    assert(traininglayer_analayerCFB);

    return EXIT_SUCCESS;

}
