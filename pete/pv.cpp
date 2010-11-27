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
#include "../PetaVision/src/layers/V1.hpp"
#include "../PetaVision/src/layers/GeislerLayer.hpp"
#include "../PetaVision/src/io/ConnectionProbe.hpp"
#include "../PetaVision/src/io/GLDisplay.hpp"
#include "../PetaVision/src/io/PostConnProbe.hpp"
#include "../PetaVision/src/io/LinearActivityProbe.hpp"
#include "../PetaVision/src/io/PointProbe.hpp"
#include "../PetaVision/src/io/StatsProbe.hpp"
#include "../PetaVision/src/layers/PVLayer.h"

#include "GV1.hpp"
#include "GenerativeLayer.hpp"
#include "GenerativeConn.hpp"
#include "FeedbackConn.hpp"
#include "IdentConn.hpp"
#include "LateralConn.hpp"
#include "L2NormProbe.hpp"
#include "SparsityTermProbe.hpp"
#include "GenColProbe.hpp"
#include "ChannelProbe.hpp"

using namespace PV;

int printdivider(unsigned int n);

int main(int argc, char* argv[]) {
    // create the managing hypercolumn
    //
    HyPerCol * hc = new HyPerCol("column", argc, argv);
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

    float display_period = 100.0; // Each image stays up for display_period seconds
    // The HyPerCol parameter "dt" controls how much time elapses each time step.

    // Layers
    Movie * slideshow = new Movie("Slideshow", hc, fileOfFileNames, display_period);

    Retina * retina = new Retina("Retina", hc);
    L2NormProbe * l2norm_retina = new L2NormProbe("Retina      :");
    retina->insertProbe(l2norm_retina);

    GV1 * anaretina = new GV1("AnaRetina", hc);
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

    GV1 * paralayerA = new GV1("ParaLayer A", hc);
    L2NormProbe * l2norm_paralayerA = new L2NormProbe("ParaLayer A :");
    paralayerA->insertProbe(l2norm_paralayerA);
    hcprobe->addTerm(l2norm_paralayerA, paralayerA);

//    ChannelProbe * paralayerA_exc = new ChannelProbe("paralayerA_exc.txt", CHANNEL_EXC);
//    paralayerA->insertProbe(paralayerA_exc);
//    ChannelProbe * paralayerA_inh = new ChannelProbe("paralayerA_inh.txt", CHANNEL_INH);
//    paralayerA->insertProbe(paralayerA_inh);

    GV1 * analayerA = new GV1("AnaLayer A", hc);
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
    assert(anaretina_layerA); // Change the name to EvolveConn?
    FeedbackConn * layerA_anaretinaFB = new FeedbackConn("Layer A to AnaRetina Feedback", hc, CHANNEL_INH, anaretina_layerA);
    assert(layerA_anaretinaFB);
    LateralConn * layerA_paralayerA = new LateralConn("Layer A to ParaLayer A", hc, layerA, paralayerA, CHANNEL_EXC);
    assert(layerA_paralayerA); // layerA_paralayerA will be a LateralConn
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

    hc->run();

    /* clean up (HyPerCol owns layers and connections, don't delete them) */
    delete hc;

    return EXIT_SUCCESS;
}
