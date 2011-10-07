/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"

#include <src/io/PointProbe.hpp>            // linear activity of a point

#include <src/io/PointLIFProbe.hpp>

#include <src/io/StatsProbe.hpp>

#define MAIN_USES_ADDCUSTOM

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
int customexit(HyPerCol * hc, int argc, char * argv[]);
// customexit is for adding objects not supported by build().

#endif // MAIN_USES_ADDCUSTOM


int main(int argc, char * argv[]) {

    int status;
#ifdef MAIN_USES_ADDCUSTOM
    status = buildandrun(argc, argv, &addcustom , &customexit);
#else
    status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]) {
    int status = 0;


    // the probe pointers !

    HyPerLayer * Cone          = getLayerFromName("Cone", hc);
    if (Cone == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};

    HyPerLayer * BipolarON       = getLayerFromName("BipolarON", hc);
    if (BipolarON == NULL) {fprintf(stdout,"Can't find BipolarON pointer"); exit(-1);};
    HyPerLayer * Horizontal    = getLayerFromName("Horizontal", hc);
    if (Horizontal == NULL) {fprintf(stdout,"Can't find Horizontal pointer"); exit(-1);};
    HyPerLayer * GanglionON      = getLayerFromName("GanglionON", hc);
    if (GanglionON == NULL) {fprintf(stdout,"Can't find GanglionON pointer"); exit(-1);};
    HyPerLayer * AmacrineON      = getLayerFromName("AmacrineON", hc);
    if (AmacrineON == NULL) {fprintf(stdout,"Can't find AmacrineON pointer"); exit(-1);};
    HyPerLayer * SynchronicityON = getLayerFromName("SynchronicityON", hc); // for analysis
    if (SynchronicityON == NULL) {fprintf(stdout,"Can't find SynchronicityON pointer"); exit(-1);};
    HyPerLayer * RetinaON      = getLayerFromName("RetinaON", hc);
    if (RetinaON == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};


    HyPerLayer * BipolarOFF       = getLayerFromName("BipolarOFF", hc);
    if (BipolarOFF == NULL) {fprintf(stdout,"Can't find BipolarOFF pointer"); exit(-1);};
    HyPerLayer * GanglionOFF      = getLayerFromName("GanglionOFF", hc);
    if (GanglionOFF == NULL) {fprintf(stdout,"Can't find GanglionOFF pointer"); exit(-1);};
    HyPerLayer * AmacrineOFF      = getLayerFromName("AmacrineOFF", hc);
    if (AmacrineOFF == NULL) {fprintf(stdout,"Can't find AmacrineOFF pointer"); exit(-1);};
    HyPerLayer * SynchronicityOFF = getLayerFromName("SynchronicityOFF", hc); // for analysis
    if (SynchronicityOFF == NULL) {fprintf(stdout,"Can't find SynchronicityOFF pointer"); exit(-1);};
    HyPerLayer * RetinaOFF      = getLayerFromName("RetinaOFF", hc);
    if (RetinaOFF == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};


    int locX = 128;
    int locY = 128;      // probing the center
    int locF = 0;        // feature 0

    // remember to delete the probes at the end

    LayerProbe * ptprobeCone = new PointLIFProbe("ptCone.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeCone);

    LayerProbe * ptprobeConeScr = new PointLIFProbe(locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeScr);

    LayerProbe * statsCone = new StatsProbe("statsCone.txt",hc,BufActivity,"Cone:");
    Cone->insertProbe(statsCone);
 //----------------------------------------------------------------------
    LayerProbe * ptprobeBipolarON = new PointLIFProbe("ptBipolarON.txt",hc,locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarON);
    LayerProbe * ptprobeBipolarONSrc = new PointLIFProbe(locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarONSrc);

    LayerProbe * statsBipolarON = new StatsProbe("statsBipolarON.txt",hc,BufActivity,"BipolarON:");
    BipolarON->insertProbe(statsBipolarON);
 //-----------------------------------------------------------------------
    LayerProbe * ptprobeHorizontal = new PointLIFProbe("ptHorizontal.txt",hc,locX/2., locY/2., locF, "Horizontal:");
    Horizontal->insertProbe(ptprobeHorizontal);

    LayerProbe * statsHorizontal = new StatsProbe("statsHorizontal.txt",hc,BufActivity,"Horizontal:");
    Horizontal->insertProbe(statsHorizontal);
 //----------------------------------------------------------------------
    LayerProbe * ptprobeGanglionON = new PointLIFProbe("ptGanglionON.txt",hc,locX/2., locY/2., locF, "GanglionON:");
    GanglionON->insertProbe(ptprobeGanglionON);
    //LayerProbe * ptprobeGanglionONSrc = new PointLIFProbe(locX, locY, locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONSrc);

    LayerProbe * statsGanglionON = new StatsProbe("statsGanglionON.txt",hc,BufActivity,"GanglionON:");
    GanglionON->insertProbe(statsGanglionON);
    LayerProbe * statsGanglionONScr = new StatsProbe(BufActivity,"GanglionON:");
    GanglionON->insertProbe(statsGanglionONScr);

 //----------------------------------------------------------------------
    LayerProbe * ptprobeAmacrineON = new PointLIFProbe("ptAmacrineON.txt",hc,locX/4., locY/4., locF, "AmacrineON:");
    AmacrineON->insertProbe(ptprobeAmacrineON);

    LayerProbe * statsAmacrineON = new StatsProbe("statsAmacrineON.txt",hc,BufActivity,"AmacrineON:");
    AmacrineON->insertProbe(statsAmacrineON);

    //----------------------------------------------------------------------

    LayerProbe * statsSynchronicityON = new StatsProbe(BufActivity,"SynchronicityON:");
    SynchronicityON->insertProbe(statsSynchronicityON);
    //----------------------------------------------------------------------


    LayerProbe * ptprobeRetinaON = new PointProbe("ptRetinaON.txt",hc,locX, locY, locF, "RetinaON:");
    RetinaON->insertProbe(ptprobeRetinaON);

    LayerProbe * statsRetinaON = new StatsProbe("statsRetinaON.txt",hc,BufActivity,"RetinaON:");
    RetinaON->insertProbe(statsRetinaON);

    LayerProbe * statsRetinaONSrc = new StatsProbe(BufActivity,"RetinaON:");
    RetinaON->insertProbe(statsRetinaONSrc);

    //----------------------------------------------------------------------
        LayerProbe * ptprobeBipolarOFF = new PointLIFProbe("ptBipolarOFF.txt",hc,locX, locY, locF, "BipolarOFF:");
        BipolarOFF->insertProbe(ptprobeBipolarOFF);
        LayerProbe * ptprobeBipolarOFFSrc = new PointLIFProbe(locX, locY, locF, "BipolarOFF:");
        BipolarOFF->insertProbe(ptprobeBipolarOFFSrc);

        LayerProbe * statsBipolarOFF = new StatsProbe("statsBipolarOFF.txt",hc,BufActivity,"BipolarOFF:");
        BipolarOFF->insertProbe(statsBipolarOFF);
     //-----------------------------------------------------------------------
        LayerProbe * ptprobeGanglionOFF = new PointLIFProbe("ptGanglionOFF.txt",hc,locX/2., locY/2., locF, "GanglionOFF:");
        GanglionOFF->insertProbe(ptprobeGanglionOFF);
        //LayerProbe * ptprobeGanglionOFFSrc = new PointLIFProbe(locX, locY, locF, "GanglionOFF:");
        //GanglionOFF->insertProbe(ptprobeGanglionOFFSrc);

        LayerProbe * statsGanglionOFF = new StatsProbe("statsGanglionOFF.txt",hc,BufActivity,"GanglionOFF:");
        GanglionOFF->insertProbe(statsGanglionOFF);
        LayerProbe * statsGanglionOFFScr = new StatsProbe(BufActivity,"GanglionOFF:");
        GanglionOFF->insertProbe(statsGanglionOFFScr);

     //----------------------------------------------------------------------
        LayerProbe * ptprobeAmacrineOFF = new PointLIFProbe("ptAmacrineOFF.txt",hc,locX/4., locY/4., locF, "AmacrineOFF:");
        AmacrineOFF->insertProbe(ptprobeAmacrineOFF);

        LayerProbe * statsAmacrineOFF = new StatsProbe("statsAmacrineOFF.txt",hc,BufActivity,"AmacrineOFF:");
        AmacrineOFF->insertProbe(statsAmacrineOFF);

        //----------------------------------------------------------------------

        LayerProbe * statsSynchronicityOFF = new StatsProbe(BufActivity,"SynchronicityOFF:");
        SynchronicityOFF->insertProbe(statsSynchronicityOFF);
        //----------------------------------------------------------------------


        LayerProbe * ptprobeRetinaOFF = new PointProbe("ptRetinaOFF.txt",hc,locX, locY, locF, "RetinaOFF:");
        RetinaOFF->insertProbe(ptprobeRetinaOFF);

        LayerProbe * statsRetinaOFF = new StatsProbe("statsRetinaOFF.txt",hc,BufActivity,"RetinaOFF:");
        RetinaOFF->insertProbe(statsRetinaOFF);

        LayerProbe * statsRetinaOFFSrc = new StatsProbe(BufActivity,"RetinaOFF:");
        RetinaOFF->insertProbe(statsRetinaOFFSrc);


    // ---- calibration probes

    locY = 225;      // probing the bottom
    locF = 0;        // feature 0

    //const int ptb  = 248;  //black
    //const int pt10 =   8;  //100 %
    //const int pt08 =  59;  // 80 %
    //const int pt06 = 109;  // 60 %
    //const int pt04 = 159;  // 40 %
    //const int pt02 = 209;  // 20 %
    //const int pt00 = 249;  // 00 %

    const int ptb  = 244;  //black
    const int pt10 =  12;  //100 %
    const int pt08 =  64;  // 80 %
    const int pt06 = 116;  // 60 %
    const int pt04 = 168;  // 40 %
    const int pt02 = 220;  // 20 %
    const int pt00 = 244;  // 00 %


    locX = ptb;    // 3 + 12.8 /2 + n * 25.6
    LayerProbe * ptprobeConeB = new PointLIFProbe("ptConeB.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeB);

    locX = pt10;    // 3 + 12.8 /2 + n * 25.6
    LayerProbe * ptprobeCone10 = new PointLIFProbe("ptCone10.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeCone10);

    locX = pt08;    // 3 + 12.8 /2 + n * 25.6
    LayerProbe * ptprobeCone08 = new PointLIFProbe("ptCone08.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeCone08);

    locX = pt06;    // 3 + 12.8 /2 + n * 25.6
    LayerProbe * ptprobeCone06 = new PointLIFProbe("ptCone06.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeCone06);

    locX = pt04;    // 3 + 12.8 /2 + n * 25.6
    LayerProbe * ptprobeCone04 = new PointLIFProbe("ptCone04.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeCone04);

    locX = pt02;    // 3 + 12.8 /2 + n * 25.6
    LayerProbe * ptprobeCone02 = new PointLIFProbe("ptCone02.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeCone02);

  //--------------------------------------------------------------------------------------------


    locX = 128;
    locY = 128;   // probing the center DOT !
    LayerProbe * ptprobeConeP1 = new PointLIFProbe("ptConeP1.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeP1);
    LayerProbe * ptprobeHorizontalP1 = new PointLIFProbe("ptHorizontalP1.txt",hc,locX/2., locY/2., locF, "Horizontal:");
    Horizontal->insertProbe(ptprobeHorizontalP1);
    LayerProbe * ptprobeBipolarONP1 = new PointLIFProbe("ptBipolarONP1.txt",hc,locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarONP1);
    LayerProbe * ptprobeGanglionONP1 = new PointLIFProbe("ptGanglionONP1.txt",hc,locX/2., locY/2., locF, "GanglionON:");
    GanglionON->insertProbe(ptprobeGanglionONP1);
    LayerProbe * ptprobeAmacrineONP1 = new PointLIFProbe("ptAmacrineONP1.txt",hc,locX/4., locY/4., locF, "AmacrineON:");
    AmacrineON->insertProbe(ptprobeAmacrineONP1);

    LayerProbe * ptprobeBipolarOFFP1 = new PointLIFProbe("ptBipolarOFFP1.txt",hc,locX, locY, locF, "BipolarOFF:");
    BipolarOFF->insertProbe(ptprobeBipolarOFFP1);
    LayerProbe * ptprobeGanglionOFFP1 = new PointLIFProbe("ptGanglionOFFP1.txt",hc,locX/2., locY/2., locF, "GanglionOFF:");
    GanglionOFF->insertProbe(ptprobeGanglionOFFP1);
    LayerProbe * ptprobeAmacrineOFFP1 = new PointLIFProbe("ptAmacrineOFFP1.txt",hc,locX/4., locY/4., locF, "AmacrineOFF:");
    AmacrineOFF->insertProbe(ptprobeAmacrineOFFP1);


    locX = 128-15.;
    locY = 128+15.;   // probing the four surround patches
    LayerProbe * ptprobeConeP3 = new PointLIFProbe("ptConeP3.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeP3);
    LayerProbe * ptprobeHorizontalP3 = new PointLIFProbe("ptHorizontalP3.txt",hc,locX/2., locY/2., locF, "Horizontal:");
    Horizontal->insertProbe(ptprobeHorizontalP3);
    LayerProbe * ptprobeBipolarONP3 = new PointLIFProbe("ptBipolarONP3.txt",hc,locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarONP3);
    LayerProbe * ptprobeGanglionONP3 = new PointLIFProbe("ptGanglionONP3.txt",hc,locX/2., locY/2., locF, "GanglionON:");
    GanglionON->insertProbe(ptprobeGanglionONP3);
    LayerProbe * ptprobeAmacrineONP3 = new PointLIFProbe("ptAmacrineONP3.txt",hc,locX/4., locY/4., locF, "AmacrineON:");
    AmacrineON->insertProbe(ptprobeAmacrineONP3);

    LayerProbe * ptprobeBipolarOFFP3 = new PointLIFProbe("ptBipolarOFFP3.txt",hc,locX, locY, locF, "BipolarOFF:");
    BipolarOFF->insertProbe(ptprobeBipolarOFFP3);
    LayerProbe * ptprobeGanglionOFFP3 = new PointLIFProbe("ptGanglionOFFP3.txt",hc,locX/2., locY/2., locF, "GanglionOFF:");
    GanglionOFF->insertProbe(ptprobeGanglionOFFP3);
    LayerProbe * ptprobeAmacrineOFFP3 = new PointLIFProbe("ptAmacrineOFFP3.txt",hc,locX/4., locY/4., locF, "AmacrineOFF:");
    AmacrineOFF->insertProbe(ptprobeAmacrineOFFP3);


    locX = 128-15;
    locY = 128-15;   // probing the four surround patches
    LayerProbe * ptprobeConeP5 = new PointLIFProbe("ptConeP5.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeP5);
    LayerProbe * ptprobeHorizontalP5 = new PointLIFProbe("ptHorizontalP5.txt",hc,locX/2., locY/2., locF, "Horizontal:");
    Horizontal->insertProbe(ptprobeHorizontalP5);
    LayerProbe * ptprobeBipolarONP5 = new PointLIFProbe("ptBipolarONP5.txt",hc,locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarONP5);
    LayerProbe * ptprobeGanglionONP5 = new PointLIFProbe("ptGanglionONP5.txt",hc,locX/2., locY/2., locF, "GanglionON:");
    GanglionON->insertProbe(ptprobeGanglionONP5);
    LayerProbe * ptprobeAmacrineONP5 = new PointLIFProbe("ptAmacrineONP5.txt",hc,locX/4., locY/4., locF, "AmacrineON:");
    AmacrineON->insertProbe(ptprobeAmacrineONP5);

    LayerProbe * ptprobeBipolarOFFP5 = new PointLIFProbe("ptBipolarOFFP5.txt",hc,locX, locY, locF, "BipolarOFF:");
     BipolarOFF->insertProbe(ptprobeBipolarOFFP5);
     LayerProbe * ptprobeGanglionOFFP5 = new PointLIFProbe("ptGanglionOFFP5.txt",hc,locX/2., locY/2., locF, "GanglionOFF:");
     GanglionOFF->insertProbe(ptprobeGanglionOFFP5);
     LayerProbe * ptprobeAmacrineOFFP5 = new PointLIFProbe("ptAmacrineOFFP5.txt",hc,locX/4., locY/4., locF, "AmacrineOFF:");
     AmacrineOFF->insertProbe(ptprobeAmacrineOFFP5);


    locX = 128+15;
    locY = 128+15;   // probing the four surround patches
    LayerProbe * ptprobeConeP7 = new PointLIFProbe("ptConeP7.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeP7);
    LayerProbe * ptprobeHorizontalP7 = new PointLIFProbe("ptHorizontalP7.txt",hc,locX/2., locY/2., locF, "Horizontal:");
    Horizontal->insertProbe(ptprobeHorizontalP7);
    LayerProbe * ptprobeBipolarONP7 = new PointLIFProbe("ptBipolarONP7.txt",hc,locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarONP7);
    LayerProbe * ptprobeGanglionONP7 = new PointLIFProbe("ptGanglionONP7.txt",hc,locX/2., locY/2., locF, "GanglionON:");
    GanglionON->insertProbe(ptprobeGanglionONP7);
    LayerProbe * ptprobeAmacrineONP7 = new PointLIFProbe("ptAmacrineONP7.txt",hc,locX/4., locY/4., locF, "AmacrineON:");
    AmacrineON->insertProbe(ptprobeAmacrineONP7);

    LayerProbe * ptprobeBipolarOFFP7 = new PointLIFProbe("ptBipolarOFFP7.txt",hc,locX, locY, locF, "BipolarOFF:");
    BipolarOFF->insertProbe(ptprobeBipolarOFFP7);
    LayerProbe * ptprobeGanglionOFFP7 = new PointLIFProbe("ptGanglionOFFP7.txt",hc,locX/2., locY/2., locF, "GanglionOFF:");
    GanglionOFF->insertProbe(ptprobeGanglionOFFP7);
    LayerProbe * ptprobeAmacrineOFFP7 = new PointLIFProbe("ptAmacrineOFFP7.txt",hc,locX/4., locY/4., locF, "AmacrineOFF:");
    AmacrineOFF->insertProbe(ptprobeAmacrineOFFP7);


    locX = 128+15;
    locY = 128-15;   // probing the four surround patches
    LayerProbe * ptprobeConeP9 = new PointLIFProbe("ptConeP9.txt",hc,locX, locY, locF, "Cone:");
    Cone->insertProbe(ptprobeConeP9);
    LayerProbe * ptprobeHorizontalP9 = new PointLIFProbe("ptHorizontalP9.txt",hc,locX/2., locY/2., locF, "Horizontal:");
    Horizontal->insertProbe(ptprobeHorizontalP9);
    LayerProbe * ptprobeBipolarONP9 = new PointLIFProbe("ptBipolarONP9.txt",hc,locX, locY, locF, "BipolarON:");
    BipolarON->insertProbe(ptprobeBipolarONP9);
    LayerProbe * ptprobeGanglionONP9 = new PointLIFProbe("ptGanglionONP9.txt",hc,locX/2., locY/2., locF, "GanglionON:");
    GanglionON->insertProbe(ptprobeGanglionONP9);
    LayerProbe * ptprobeAmacrineONP9 = new PointLIFProbe("ptAmacrineONP9.txt",hc,locX/4., locY/4., locF, "AmacrineON:");
    AmacrineON->insertProbe(ptprobeAmacrineONP9);

    LayerProbe * ptprobeBipolarOFFP9 = new PointLIFProbe("ptBipolarOFFP9.txt",hc,locX, locY, locF, "BipolarOFF:");
    BipolarOFF->insertProbe(ptprobeBipolarOFFP9);
    LayerProbe * ptprobeGanglionOFFP9 = new PointLIFProbe("ptGanglionOFFP9.txt",hc,locX/2., locY/2., locF, "GanglionOFF:");
    GanglionOFF->insertProbe(ptprobeGanglionOFFP9);
    LayerProbe * ptprobeAmacrineOFFP9 = new PointLIFProbe("ptAmacrineOFFP9.txt",hc,locX/4., locY/4., locF, "AmacrineOFF:");
    AmacrineOFF->insertProbe(ptprobeAmacrineOFFP9);


    //



  // ---- calibration probes ON

       locY = 225;      // probing the bottom of the bar
       locF = 0;        // feature 0

       locX = ptb;    // 3 + 12.8 /2 + n * 25.6
       LayerProbe * ptprobeBipolarONB = new PointLIFProbe("ptBipolarONB.txt",hc,locX, locY, locF, "BipolarON:");
       BipolarON->insertProbe(ptprobeBipolarONB);

       locX = pt10;    // 3 + 12.8 /2 + n * 25.6
       LayerProbe * ptprobeBipolarON10 = new PointLIFProbe("ptBipolarON10.txt",hc,locX, locY, locF, "BipolarON:");
       BipolarON->insertProbe(ptprobeBipolarON10);

       locX = pt08;    // 3 + 12.8 /2 + n * 25.6
       LayerProbe * ptprobeBipolarON08 = new PointLIFProbe("ptBipolarON08.txt",hc,locX, locY, locF, "BipolarON:");
       BipolarON->insertProbe(ptprobeBipolarON08);

       locX = pt06;    // 3 + 12.8 /2 + n * 25.6
       LayerProbe * ptprobeBipolarON06 = new PointLIFProbe("ptBipolarON06.txt",hc,locX, locY, locF, "BipolarON:");
       BipolarON->insertProbe(ptprobeBipolarON06);

       locX = pt04;    // 3 + 12.8 /2 + n * 25.6
       LayerProbe * ptprobeBipolarON04 = new PointLIFProbe("ptBipolarON04.txt",hc,locX, locY, locF, "BipolarON:");
       BipolarON->insertProbe(ptprobeBipolarON04);

       locX = pt02;    // 3 + 12.8 /2 + n * 25.6
       LayerProbe * ptprobeBipolarON02 = new PointLIFProbe("ptBipolarON02.txt",hc,locX, locY, locF, "BipolarON:");
       BipolarON->insertProbe(ptprobeBipolarON02);

     //

       // ---- calibration probes NSCALE IS 2

          locY = 225/2.;      // probing the middle of the bar
          locF = 0;        // feature 0

          locX = ptb/2.;    // mid between the first two bars
          LayerProbe * ptprobeHorizontalB = new PointLIFProbe("ptHorizontalB.txt",hc,locX, locY, locF, "Horizontal:");
          Horizontal->insertProbe(ptprobeHorizontalB);

          locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
          LayerProbe * ptprobeHorizontal10 = new PointLIFProbe("ptHorizontal10.txt",hc,locX, locY, locF, "Horizontal:");
          Horizontal->insertProbe(ptprobeHorizontal10);

          locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
          LayerProbe * ptprobeHorizontal08 = new PointLIFProbe("ptHorizontal08.txt",hc,locX, locY, locF, "Horizontal:");
          Horizontal->insertProbe(ptprobeHorizontal08);

          locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
          LayerProbe * ptprobeHorizontal06 = new PointLIFProbe("ptHorizontal06.txt",hc,locX, locY, locF, "Horizontal:");
          Horizontal->insertProbe(ptprobeHorizontal06);

          locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
          LayerProbe * ptprobeHorizontal04 = new PointLIFProbe("ptHorizontal04.txt",hc,locX, locY, locF, "Horizontal:");
          Horizontal->insertProbe(ptprobeHorizontal04);

          locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
          LayerProbe * ptprobeHorizontal02 = new PointLIFProbe("ptHorizontal02.txt",hc,locX, locY, locF, "Horizontal:");
          Horizontal->insertProbe(ptprobeHorizontal02);

        //


          // ---- calibration probes NSCALE IS 2

             locY = 225/2.;      // probing the middle of the bar
             locF = 0;        // feature 0

             locX = ptb/2.;    // mid between the first two bars
             LayerProbe * ptprobeGanglionONB = new PointLIFProbe("ptGanglionONB.txt",hc,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionONB);

             locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
             LayerProbe * ptprobeGanglionON10 = new PointLIFProbe("ptGanglionON10.txt",hc,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionON10);

             locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
             LayerProbe * ptprobeGanglionON08 = new PointLIFProbe("ptGanglionON08.txt",hc,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionON08);

             locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
             LayerProbe * ptprobeGanglionON06 = new PointLIFProbe("ptGanglionON06.txt",hc,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionON06);

             locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
             LayerProbe * ptprobeGanglionON04 = new PointLIFProbe("ptGanglionON04.txt",hc,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionON04);

             locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
             LayerProbe * ptprobeGanglionON02 = new PointLIFProbe("ptGanglionON02.txt",hc,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionON02);

           //

             // ---- calibration probes NSCALE IS 4 --- 0.25

                locY = 225/4.;      // probing the middle of the bar
                locF = 0;        // feature 0

                locX = ptb/4.;    // mid between the first two bars
                LayerProbe * ptprobeAmacrineONB = new PointLIFProbe("ptAmacrineONB.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineONB);

                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                LayerProbe * ptprobeAmacrineON10 = new PointLIFProbe("ptAmacrineON10.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineON10);

                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                LayerProbe * ptprobeAmacrineON08 = new PointLIFProbe("ptAmacrineON08.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineON08);

                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                LayerProbe * ptprobeAmacrineON06 = new PointLIFProbe("ptAmacrineON06.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineON06);

                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                LayerProbe * ptprobeAmacrineON04 = new PointLIFProbe("ptAmacrineON04.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineON04);

                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                LayerProbe * ptprobeAmacrineON02 = new PointLIFProbe("ptAmacrineON02.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineON02);

                // ---- calibration probes NSCALE IS 4

                locX = 	60/4.;
                locY =  15.;      // probing
                LayerProbe * ptprobeAmacrineONRU1 = new PointLIFProbe("ptAmacrineONRU1.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineONRU1);

                locY =  44;      // probing
                LayerProbe * ptprobeAmacrineONRD1 = new PointLIFProbe("ptAmacrineONRD1.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineONRD1);


                locY =  75;      // probing
                LayerProbe * ptprobeAmacrineONRU2 = new PointLIFProbe("ptAmacrineONRU2.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineONRU2);

                locY =  104;      // probing
                LayerProbe * ptprobeAmacrineONRD2 = new PointLIFProbe("ptAmacrineONRD2.txt",hc,locX, locY, locF, "AmacrineON:");
                AmacrineON->insertProbe(ptprobeAmacrineONRD2);




                // ---- calibration probes NSCALE IS 2

                locX = 	60/2.;
                locY =  15.;      // probing
                LayerProbe * ptprobeGanglionONRU1 = new PointLIFProbe("ptGanglionONRU1.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRU1);

                locY =  44;      // probing
                LayerProbe * ptprobeGanglionONRD1 = new PointLIFProbe("ptGanglionONRD1.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRD1);


                locY =  75;      // probing
                LayerProbe * ptprobeGanglionONRU2 = new PointLIFProbe("ptGanglionONRU2.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRU2);

                locY =  104;      // probing
                LayerProbe * ptprobeGanglionONRD2 = new PointLIFProbe("ptGanglionONRD2.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRD2);


                locX = 	58/2.;
                locY =  15.;      // probing
                LayerProbe * ptprobeGanglionONRU1l = new PointLIFProbe("ptGanglionONRU1l.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRU1l);

                locY =  44;      // probing
                LayerProbe * ptprobeGanglionONRD1l = new PointLIFProbe("ptGanglionONRD1l.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRD1l);


                locY =  75;      // probing
                LayerProbe * ptprobeGanglionONRU2l = new PointLIFProbe("ptGanglionONRU2l.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRU2l);

                locY =  104;      // probing
                LayerProbe * ptprobeGanglionONRD2l = new PointLIFProbe("ptGanglionONRD2l.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRD2l);



                locX = 	62/2.;
                locY =  15.;      // probing
                LayerProbe * ptprobeGanglionONRU1r = new PointLIFProbe("ptGanglionONRU1r.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRU1r);

                locY =  44;      // probing
                LayerProbe * ptprobeGanglionONRD1r = new PointLIFProbe("ptGanglionONRD1r.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRD1r);


                locY =  75;      // probing
                LayerProbe * ptprobeGanglionONRU2r = new PointLIFProbe("ptGanglionONRU2r.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRU2r);

                locY =  104;      // probing
                LayerProbe * ptprobeGanglionONRD2r = new PointLIFProbe("ptGanglionONRD2r.txt",hc,locX, locY, locF, "GanglionON:");
                GanglionON->insertProbe(ptprobeGanglionONRD2r);


                //------- SynchronicityON layer
                locX = 	60/2.;
                locY =  15.;      // probing
                LayerProbe * ptprobeSynchronicityONRU1 = new PointLIFProbe("ptSynchronicityONRU1.txt",hc,locX, locY, locF, "SynchronicityON:");
                SynchronicityON->insertProbe(ptprobeSynchronicityONRU1);

                locY =  44;      // probing
                LayerProbe * ptprobeSynchronicityONRD1 = new PointLIFProbe("ptSynchronicityONRD1.txt",hc,locX, locY, locF, "SynchronicityON:");
                SynchronicityON->insertProbe(ptprobeSynchronicityONRD1);


                locY =  75;      // probing
                LayerProbe * ptprobeSynchronicityONRU2 = new PointLIFProbe("ptSynchronicityONRU2.txt",hc,locX, locY, locF, "SynchronicityON:");
                SynchronicityON->insertProbe(ptprobeSynchronicityONRU2);

                locY =  104;      // probing
                LayerProbe * ptprobeSynchronicityONRD2 = new PointLIFProbe("ptSynchronicityONRD2.txt",hc,locX, locY, locF, "SynchronicityON:");
                SynchronicityON->insertProbe(ptprobeSynchronicityONRD2);

                // ---- calibration probes

                       locY = 225;      // probing the bottom of the bar
                       locF = 0;        // feature 0

                       locX = ptb;    // 3 + 12.8 /2 + n * 25.6
                       LayerProbe * ptprobeBipolarOFFB = new PointLIFProbe("ptBipolarOFFB.txt",hc,locX, locY, locF, "BipolarOFF:");
                       BipolarOFF->insertProbe(ptprobeBipolarOFFB);

                       locX = pt10;    // 3 + 12.8 /2 + n * 25.6
                       LayerProbe * ptprobeBipolarOFF10 = new PointLIFProbe("ptBipolarOFF10.txt",hc,locX, locY, locF, "BipolarOFF:");
                       BipolarOFF->insertProbe(ptprobeBipolarOFF10);

                       locX = pt08;    // 3 + 12.8 /2 + n * 25.6
                       LayerProbe * ptprobeBipolarOFF08 = new PointLIFProbe("ptBipolarOFF08.txt",hc,locX, locY, locF, "BipolarOFF:");
                       BipolarOFF->insertProbe(ptprobeBipolarOFF08);

                       locX = pt06;    // 3 + 12.8 /2 + n * 25.6
                       LayerProbe * ptprobeBipolarOFF06 = new PointLIFProbe("ptBipolarOFF06.txt",hc,locX, locY, locF, "BipolarOFF:");
                       BipolarOFF->insertProbe(ptprobeBipolarOFF06);

                       locX = pt04;    // 3 + 12.8 /2 + n * 25.6
                       LayerProbe * ptprobeBipolarOFF04 = new PointLIFProbe("ptBipolarOFF04.txt",hc,locX, locY, locF, "BipolarOFF:");
                       BipolarOFF->insertProbe(ptprobeBipolarOFF04);

                       locX = pt02;    // 3 + 12.8 /2 + n * 25.6
                       LayerProbe * ptprobeBipolarOFF02 = new PointLIFProbe("ptBipolarOFF02.txt",hc,locX, locY, locF, "BipolarOFF:");
                       BipolarOFF->insertProbe(ptprobeBipolarOFF02);

                     //

                       // ---- calibration probes NSCALE IS 2
                        // ---- calibration probes NSCALE IS 2

                             locY = 225/2.;      // probing the middle of the bar
                             locF = 0;        // feature 0

                             locX = ptb/2.;    // mid between the first two bars
                             LayerProbe * ptprobeGanglionOFFB = new PointLIFProbe("ptGanglionOFFB.txt",hc,locX, locY, locF, "GanglionOFF:");
                             GanglionOFF->insertProbe(ptprobeGanglionOFFB);

                             locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
                             LayerProbe * ptprobeGanglionOFF10 = new PointLIFProbe("ptGanglionOFF10.txt",hc,locX, locY, locF, "GanglionOFF:");
                             GanglionOFF->insertProbe(ptprobeGanglionOFF10);

                             locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
                             LayerProbe * ptprobeGanglionOFF08 = new PointLIFProbe("ptGanglionOFF08.txt",hc,locX, locY, locF, "GanglionOFF:");
                             GanglionOFF->insertProbe(ptprobeGanglionOFF08);

                             locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
                             LayerProbe * ptprobeGanglionOFF06 = new PointLIFProbe("ptGanglionOFF06.txt",hc,locX, locY, locF, "GanglionOFF:");
                             GanglionOFF->insertProbe(ptprobeGanglionOFF06);

                             locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
                             LayerProbe * ptprobeGanglionOFF04 = new PointLIFProbe("ptGanglionOFF04.txt",hc,locX, locY, locF, "GanglionOFF:");
                             GanglionOFF->insertProbe(ptprobeGanglionOFF04);

                             locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
                             LayerProbe * ptprobeGanglionOFF02 = new PointLIFProbe("ptGanglionOFF02.txt",hc,locX, locY, locF, "GanglionOFF:");
                             GanglionOFF->insertProbe(ptprobeGanglionOFF02);

                           //

                             // ---- calibration probes NSCALE IS 4 --- 0.25

                                locY = 225/4.;      // probing the middle of the bar
                                locF = 0;        // feature 0

                                locX = ptb/4.;    // mid between the first two bars
                                LayerProbe * ptprobeAmacrineOFFB = new PointLIFProbe("ptAmacrineOFFB.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFFB);

                                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                                LayerProbe * ptprobeAmacrineOFF10 = new PointLIFProbe("ptAmacrineOFF10.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFF10);

                                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                                LayerProbe * ptprobeAmacrineOFF08 = new PointLIFProbe("ptAmacrineOFF08.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFF08);

                                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                                LayerProbe * ptprobeAmacrineOFF06 = new PointLIFProbe("ptAmacrineOFF06.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFF06);

                                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                                LayerProbe * ptprobeAmacrineOFF04 = new PointLIFProbe("ptAmacrineOFF04.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFF04);

                                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                                LayerProbe * ptprobeAmacrineOFF02 = new PointLIFProbe("ptAmacrineOFF02.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFF02);

                                // ---- calibration probes NSCALE IS 4

                                locX = 	60/4.;
                                locY =  15.;      // probing
                                LayerProbe * ptprobeAmacrineOFFRU1 = new PointLIFProbe("ptAmacrineOFFRU1.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFFRU1);

                                locY =  44;      // probing
                                LayerProbe * ptprobeAmacrineOFFRD1 = new PointLIFProbe("ptAmacrineOFFRD1.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFFRD1);


                                locY =  75;      // probing
                                LayerProbe * ptprobeAmacrineOFFRU2 = new PointLIFProbe("ptAmacrineOFFRU2.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFFRU2);

                                locY =  104;      // probing
                                LayerProbe * ptprobeAmacrineOFFRD2 = new PointLIFProbe("ptAmacrineOFFRD2.txt",hc,locX, locY, locF, "AmacrineOFF:");
                                AmacrineOFF->insertProbe(ptprobeAmacrineOFFRD2);




                                // ---- calibration probes NSCALE IS 2

                                locX = 	60/2.;
                                locY =  15.;      // probing
                                LayerProbe * ptprobeGanglionOFFRU1 = new PointLIFProbe("ptGanglionOFFRU1.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRU1);

                                locY =  44;      // probing
                                LayerProbe * ptprobeGanglionOFFRD1 = new PointLIFProbe("ptGanglionOFFRD1.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRD1);


                                locY =  75;      // probing
                                LayerProbe * ptprobeGanglionOFFRU2 = new PointLIFProbe("ptGanglionOFFRU2.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRU2);

                                locY =  104;      // probing
                                LayerProbe * ptprobeGanglionOFFRD2 = new PointLIFProbe("ptGanglionOFFRD2.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRD2);


                                locX = 	58/2.;
                                locY =  15.;      // probing
                                LayerProbe * ptprobeGanglionOFFRU1l = new PointLIFProbe("ptGanglionOFFRU1l.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRU1l);

                                locY =  44;      // probing
                                LayerProbe * ptprobeGanglionOFFRD1l = new PointLIFProbe("ptGanglionOFFRD1l.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRD1l);


                                locY =  75;      // probing
                                LayerProbe * ptprobeGanglionOFFRU2l = new PointLIFProbe("ptGanglionOFFRU2l.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRU2l);

                                locY =  104;      // probing
                                LayerProbe * ptprobeGanglionOFFRD2l = new PointLIFProbe("ptGanglionOFFRD2l.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRD2l);



                                locX = 	62/2.;
                                locY =  15.;      // probing
                                LayerProbe * ptprobeGanglionOFFRU1r = new PointLIFProbe("ptGanglionOFFRU1r.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRU1r);

                                locY =  44;      // probing
                                LayerProbe * ptprobeGanglionOFFRD1r = new PointLIFProbe("ptGanglionOFFRD1r.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRD1r);


                                locY =  75;      // probing
                                LayerProbe * ptprobeGanglionOFFRU2r = new PointLIFProbe("ptGanglionOFFRU2r.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRU2r);

                                locY =  104;      // probing
                                LayerProbe * ptprobeGanglionOFFRD2r = new PointLIFProbe("ptGanglionOFFRD2r.txt",hc,locX, locY, locF, "GanglionOFF:");
                                GanglionOFF->insertProbe(ptprobeGanglionOFFRD2r);


                                //------- SynchronicityOFF layer
                                locX = 	60/2.;
                                locY =  15.;      // probing
                                LayerProbe * ptprobeSynchronicityOFFRU1 = new PointLIFProbe("ptSynchronicityOFFRU1.txt",hc,locX, locY, locF, "SynchronicityOFF:");
                                SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRU1);

                                locY =  44;      // probing
                                LayerProbe * ptprobeSynchronicityOFFRD1 = new PointLIFProbe("ptSynchronicityOFFRD1.txt",hc,locX, locY, locF, "SynchronicityOFF:");
                                SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRD1);


                                locY =  75;      // probing
                                LayerProbe * ptprobeSynchronicityOFFRU2 = new PointLIFProbe("ptSynchronicityOFFRU2.txt",hc,locX, locY, locF, "SynchronicityOFF:");
                                SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRU2);

                                locY =  104;      // probing
                                LayerProbe * ptprobeSynchronicityOFFRD2 = new PointLIFProbe("ptSynchronicityOFFRD2.txt",hc,locX, locY, locF, "SynchronicityOFF:");
                                SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRD2);



    return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {

	PVParams * params = hc->parameters();

    int status = 0;

    fprintf(stdout,"================================================\n");

    fprintf(stdout,"This run used the following strength parameters:\n");

    float strength = params->value("Image to Cone", "strength", strength);
    float sigma = params->value("Image to Cone", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","Image to Cone" ,strength,sigma);
    strength = params->value("ConeSigmoid to Horizontal", "strength", strength);
    sigma    = params->value("ConeSigmoid to Horizontal", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "ConeSigmoid to Horizontal",strength,sigma);
    strength = params->value("HoriGap to Horizontal", "strength", strength);
    sigma    = params->value("HoriGap to Horizontal", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","HoriGap to Horizontal" ,strength,sigma);
    strength = params->value("HoriSigmoid to Cone", "strength", strength);
    sigma = params->value("HoriSigmoid to Cone", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","HoriSigmoid to Cone" ,strength,sigma);
    strength = params->value("ConeSigmoid to BipolarON", "strength", strength);
    sigma    = params->value("ConeSigmoid to BipolarON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","ConeSigmoid to BipolarON" ,strength,sigma);
    strength = params->value("BipolarSigmoidON to GanglionON", "strength", strength);
    sigma    = params->value("BipolarSigmoidON to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidON to GanglionON" ,strength,sigma);
    strength = params->value("BipolarSigmoidON to AmacrineON", "strength", strength);
    sigma    = params->value("BipolarSigmoidON to AmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "BipolarSigmoidON to AmacrineON",strength,sigma);
    strength = params->value("GangliGapON to AmacrineON", "strength", strength);
    sigma    = params->value("GangliGapON to AmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","GangliGapON to AmacrineON" ,strength,sigma);
    strength = params->value("AmaGapON to GanglionON", "strength", strength);
    sigma    = params->value("AmaGapON to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","AmaGapON to GanglionON" ,strength,sigma);
    strength = params->value("AmaGapON to AmacrineON", "strength", strength);
    sigma    = params->value("AmaGapON to AmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","AmaGapON to AmacrineON" ,strength,sigma);
    strength = params->value("AmacrineON to GanglionON", "strength", strength);
    sigma = params->value("AmacrineON to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f  \n","AmacrineON to GanglionON" ,strength,sigma);
    strength = params->value("ConeSigmoid to BipolarOFF", "strength", strength);
    sigma    = params->value("ConeSigmoid to BipolarOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","ConeSigmoid to BipolarOFF" ,strength,sigma);
    strength = params->value("BipolarSigmoidOFF to GanglionOFF", "strength", strength);
    sigma    = params->value("BipolarSigmoidOFF to GanglionOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidOFF to GanglionOFF" ,strength,sigma);
    strength = params->value("BipolarSigmoidOFF to AmacrineOFF", "strength", strength);
    sigma    = params->value("BipolarSigmoidOFF to AmacrineOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "BipolarSigmoidOFF to AmacrineOFF",strength,sigma);
    strength = params->value("GangliGapOFF to AmacrineOFF", "strength", strength);
    sigma    = params->value("GangliGapOFF to AmacrineOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","GangliGapOFF to AmacrineOFF" ,strength,sigma);
    strength = params->value("AmaGapOFF to GanglionOFF", "strength", strength);
    sigma    = params->value("AmaGapOFF to GanglionOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","AmaGapOFF to GanglionOFF" ,strength,sigma);
    strength = params->value("AmaGapOFF to AmacrineOFF", "strength", strength);
    sigma    = params->value("AmaGapOFF to AmacrineOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","AmaGapOFF to AmacrineOFF" ,strength,sigma);
    strength = params->value("AmacrineOFF to GanglionOFF", "strength", strength);
    sigma = params->value("AmacrineOFF to GanglionOFF", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f  \n","AmacrineOFF to GanglionOFF" ,strength,sigma);



    fprintf(stdout,"================================================ \a \a \a \a \a \a \a \n");
    fprintf(stderr,"================================================ \a \a \a \a \a \a \a \n");


    return status;
}

	#endif // MAIN_USES_ADDCUSTOM

