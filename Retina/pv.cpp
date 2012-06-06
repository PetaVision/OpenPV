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

    HyPerLayer * Cone          = hc->getLayerFromName("Cone");
    if (Cone == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};

    HyPerLayer * BipolarON       = hc->getLayerFromName("BipolarON");
    if (BipolarON == NULL) {fprintf(stdout,"Can't find BipolarON pointer"); exit(-1);};
    HyPerLayer * Horizontal    = hc->getLayerFromName("Horizontal");
    if (Horizontal == NULL) {fprintf(stdout,"Can't find Horizontal pointer"); exit(-1);};
    HyPerLayer * GanglionON      = hc->getLayerFromName("GanglionON");
    if (GanglionON == NULL) {fprintf(stdout,"Can't find GanglionON pointer"); exit(-1);};
    HyPerLayer * AmacrineON      = hc->getLayerFromName("AmacrineON");
    if (AmacrineON == NULL) {fprintf(stdout,"Can't find AmacrineON pointer"); exit(-1);};
    HyPerLayer * SynchronicityON = hc->getLayerFromName("SynchronicityON"); // for analysis
    if (SynchronicityON == NULL) {fprintf(stdout,"Can't find SynchronicityON pointer"); exit(-1);};
    HyPerLayer * RetinaON      = hc->getLayerFromName("RetinaON");
    if (RetinaON == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};


    HyPerLayer * BipolarOFF       = hc->getLayerFromName("BipolarOFF");
    if (BipolarOFF == NULL) {fprintf(stdout,"Can't find BipolarOFF pointer"); exit(-1);};
    HyPerLayer * GanglionOFF      = hc->getLayerFromName("GanglionOFF");
    if (GanglionOFF == NULL) {fprintf(stdout,"Can't find GanglionOFF pointer"); exit(-1);};
    HyPerLayer * AmacrineOFF      = hc->getLayerFromName("AmacrineOFF");
    if (AmacrineOFF == NULL) {fprintf(stdout,"Can't find AmacrineOFF pointer"); exit(-1);};
    HyPerLayer * SynchronicityOFF = hc->getLayerFromName("SynchronicityOFF"); // for analysis
    if (SynchronicityOFF == NULL) {fprintf(stdout,"Can't find SynchronicityOFF pointer"); exit(-1);};
    HyPerLayer * RetinaOFF      = hc->getLayerFromName("RetinaOFF");
    if (RetinaOFF == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};


    int locX = 128;
    int locY = 128;      // probing the center
    int locF = 0;        // feature 0

    // remember to delete the probes at the end

    PointLIFProbe * ptprobeCone = new PointLIFProbe("ptCone.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeCone);

    PointLIFProbe * ptprobeConeScr = new PointLIFProbe(Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeScr);

    StatsProbe * statsCone = new StatsProbe("statsCone.txt",Cone,BufActivity,"Cone:");
    //Cone->insertProbe(statsCone);
 //----------------------------------------------------------------------
    PointLIFProbe * ptprobeBipolarON = new PointLIFProbe("ptBipolarON.txt",BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarON);
    PointLIFProbe * ptprobeBipolarONSrc = new PointLIFProbe(BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarONSrc);

    StatsProbe * statsBipolarON = new StatsProbe("statsBipolarON.txt",BipolarON,BufActivity,"BipolarON:");
    //BipolarON->insertProbe(statsBipolarON);
 //-----------------------------------------------------------------------
    PointLIFProbe * ptprobeHorizontal = new PointLIFProbe("ptHorizontal.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    //Horizontal->insertProbe(ptprobeHorizontal);

    StatsProbe * statsHorizontal = new StatsProbe("statsHorizontal.txt",Horizontal,BufActivity,"Horizontal:");
    //Horizontal->insertProbe(statsHorizontal);
 //----------------------------------------------------------------------
    PointLIFProbe * ptprobeGanglionON = new PointLIFProbe("ptGanglionON.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionON);
    //LayerProbe * ptprobeGanglionONSrc = new PointLIFProbe(locX, locY, locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONSrc);

    StatsProbe * statsGanglionON = new StatsProbe("statsGanglionON.txt",GanglionON,BufActivity,"GanglionON:");
    //GanglionON->insertProbe(statsGanglionON);
    StatsProbe * statsGanglionONScr = new StatsProbe(GanglionON,BufActivity,"GanglionON:");
    //GanglionON->insertProbe(statsGanglionONScr);

 //----------------------------------------------------------------------
    PointLIFProbe * ptprobeAmacrineON = new PointLIFProbe("ptAmacrineON.txt",AmacrineON,locX/4., locY/4., locF, "AmacrineON:");
    //AmacrineON->insertProbe(ptprobeAmacrineON);

    StatsProbe * statsAmacrineON = new StatsProbe("statsAmacrineON.txt",AmacrineON,BufActivity,"AmacrineON:");
    //AmacrineON->insertProbe(statsAmacrineON);

    //----------------------------------------------------------------------

    StatsProbe * statsSynchronicityON = new StatsProbe(SynchronicityON, BufActivity,"SynchronicityON:");
    //SynchronicityON->insertProbe(statsSynchronicityON);
    //----------------------------------------------------------------------


    PointProbe * ptprobeRetinaON = new PointProbe("ptRetinaON.txt",RetinaON,locX, locY, locF, "RetinaON:");
    //RetinaON->insertProbe(ptprobeRetinaON);

    StatsProbe * statsRetinaON = new StatsProbe("statsRetinaON.txt",RetinaON,BufActivity,"RetinaON:");
    //RetinaON->insertProbe(statsRetinaON);

    StatsProbe * statsRetinaONSrc = new StatsProbe(RetinaON,BufActivity,"RetinaON:");
    //RetinaON->insertProbe(statsRetinaONSrc);

    //----------------------------------------------------------------------
        PointLIFProbe * ptprobeBipolarOFF = new PointLIFProbe("ptBipolarOFF.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
        //BipolarOFF->insertProbe(ptprobeBipolarOFF);
        PointLIFProbe * ptprobeBipolarOFFSrc = new PointLIFProbe(BipolarOFF,locX, locY, locF, "BipolarOFF:");
        //BipolarOFF->insertProbe(ptprobeBipolarOFFSrc);

        StatsProbe * statsBipolarOFF = new StatsProbe("statsBipolarOFF.txt",BipolarOFF,BufActivity,"BipolarOFF:");
        BipolarOFF->insertProbe(statsBipolarOFF);
     //-----------------------------------------------------------------------
        PointLIFProbe * ptprobeGanglionOFF = new PointLIFProbe("ptGanglionOFF.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
        //GanglionOFF->insertProbe(ptprobeGanglionOFF);
        //LayerProbe * ptprobeGanglionOFFSrc = new PointLIFProbe(locX, locY, locF, "GanglionOFF:");
        //GanglionOFF->insertProbe(ptprobeGanglionOFFSrc);

        StatsProbe * statsGanglionOFF = new StatsProbe("statsGanglionOFF.txt",GanglionOFF,BufActivity,"GanglionOFF:");
        //GanglionOFF->insertProbe(statsGanglionOFF);
        StatsProbe * statsGanglionOFFScr = new StatsProbe(GanglionOFF,BufActivity,"GanglionOFF:");
        //GanglionOFF->insertProbe(statsGanglionOFFScr);

     //----------------------------------------------------------------------
        PointLIFProbe * ptprobeAmacrineOFF = new PointLIFProbe("ptAmacrineOFF.txt",AmacrineOFF,locX/4., locY/4., locF, "AmacrineOFF:");
        //AmacrineOFF->insertProbe(ptprobeAmacrineOFF);

        StatsProbe * statsAmacrineOFF = new StatsProbe("statsAmacrineOFF.txt",AmacrineOFF,BufActivity,"AmacrineOFF:");
        //AmacrineOFF->insertProbe(statsAmacrineOFF);

        //----------------------------------------------------------------------

        StatsProbe * statsSynchronicityOFF = new StatsProbe(SynchronicityOFF,BufActivity,"SynchronicityOFF:");
        //SynchronicityOFF->insertProbe(statsSynchronicityOFF);
        //----------------------------------------------------------------------


        PointProbe * ptprobeRetinaOFF = new PointProbe("ptRetinaOFF.txt",RetinaOFF,locX, locY, locF, "RetinaOFF:");
        //RetinaOFF->insertProbe(ptprobeRetinaOFF);

        StatsProbe * statsRetinaOFF = new StatsProbe("statsRetinaOFF.txt",RetinaOFF,BufActivity,"RetinaOFF:");
        //RetinaOFF->insertProbe(statsRetinaOFF);

        StatsProbe * statsRetinaOFFSrc = new StatsProbe(RetinaOFF,BufActivity,"RetinaOFF:");
        //RetinaOFF->insertProbe(statsRetinaOFFSrc);


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
    PointLIFProbe * ptprobeConeB = new PointLIFProbe("ptConeB.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeB);

    locX = pt10;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone10 = new PointLIFProbe("ptCone10.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeCone10);

    locX = pt08;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone08 = new PointLIFProbe("ptCone08.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeCone08);

    locX = pt06;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone06 = new PointLIFProbe("ptCone06.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeCone06);

    locX = pt04;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone04 = new PointLIFProbe("ptCone04.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeCone04);

    locX = pt02;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone02 = new PointLIFProbe("ptCone02.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeCone02);

  //--------------------------------------------------------------------------------------------


    locX = 128;
    locY = 128;   // probing the center DOT !
    PointLIFProbe * ptprobeConeP1 = new PointLIFProbe("ptConeP1.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeP1);
    PointLIFProbe * ptprobeHorizontalP1 = new PointLIFProbe("ptHorizontalP1.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    //Horizontal->insertProbe(ptprobeHorizontalP1);
    PointLIFProbe * ptprobeBipolarONP1 = new PointLIFProbe("ptBipolarONP1.txt",BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarONP1);
    PointLIFProbe * ptprobeGanglionONP1 = new PointLIFProbe("ptGanglionONP1.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONP1);
    PointLIFProbe * ptprobeAmacrineONP1 = new PointLIFProbe("ptAmacrineONP1.txt",AmacrineON,locX/4., locY/4., locF, "AmacrineON:");
    //AmacrineON->insertProbe(ptprobeAmacrineONP1);

    PointLIFProbe * ptprobeBipolarOFFP1 = new PointLIFProbe("ptBipolarOFFP1.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    //BipolarOFF->insertProbe(ptprobeBipolarOFFP1);
    PointLIFProbe * ptprobeGanglionOFFP1 = new PointLIFProbe("ptGanglionOFFP1.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    //GanglionOFF->insertProbe(ptprobeGanglionOFFP1);
    PointLIFProbe * ptprobeAmacrineOFFP1 = new PointLIFProbe("ptAmacrineOFFP1.txt",AmacrineOFF,locX/4., locY/4., locF, "AmacrineOFF:");
    //AmacrineOFF->insertProbe(ptprobeAmacrineOFFP1);


    locX = 128-15.;
    locY = 128+15.;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP3 = new PointLIFProbe("ptConeP3.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeP3);
    PointLIFProbe * ptprobeHorizontalP3 = new PointLIFProbe("ptHorizontalP3.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    //Horizontal->insertProbe(ptprobeHorizontalP3);
    PointLIFProbe * ptprobeBipolarONP3 = new PointLIFProbe("ptBipolarONP3.txt",BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarONP3);
    PointLIFProbe * ptprobeGanglionONP3 = new PointLIFProbe("ptGanglionONP3.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONP3);
    PointLIFProbe * ptprobeAmacrineONP3 = new PointLIFProbe("ptAmacrineONP3.txt",AmacrineON,locX/4., locY/4., locF, "AmacrineON:");
    //AmacrineON->insertProbe(ptprobeAmacrineONP3);

    PointLIFProbe * ptprobeBipolarOFFP3 = new PointLIFProbe("ptBipolarOFFP3.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    //BipolarOFF->insertProbe(ptprobeBipolarOFFP3);
    PointLIFProbe * ptprobeGanglionOFFP3 = new PointLIFProbe("ptGanglionOFFP3.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    //GanglionOFF->insertProbe(ptprobeGanglionOFFP3);
    PointLIFProbe * ptprobeAmacrineOFFP3 = new PointLIFProbe("ptAmacrineOFFP3.txt",AmacrineOFF,locX/4., locY/4., locF, "AmacrineOFF:");
    //AmacrineOFF->insertProbe(ptprobeAmacrineOFFP3);


    locX = 128-15;
    locY = 128-15;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP5 = new PointLIFProbe("ptConeP5.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeP5);
    PointLIFProbe * ptprobeHorizontalP5 = new PointLIFProbe("ptHorizontalP5.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    //Horizontal->insertProbe(ptprobeHorizontalP5);
    PointLIFProbe * ptprobeBipolarONP5 = new PointLIFProbe("ptBipolarONP5.txt",BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarONP5);
    PointLIFProbe * ptprobeGanglionONP5 = new PointLIFProbe("ptGanglionONP5.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONP5);
    PointLIFProbe * ptprobeAmacrineONP5 = new PointLIFProbe("ptAmacrineONP5.txt",AmacrineON,locX/4., locY/4., locF, "AmacrineON:");
    //AmacrineON->insertProbe(ptprobeAmacrineONP5);

    PointLIFProbe * ptprobeBipolarOFFP5 = new PointLIFProbe("ptBipolarOFFP5.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
     //BipolarOFF->insertProbe(ptprobeBipolarOFFP5);
     PointLIFProbe * ptprobeGanglionOFFP5 = new PointLIFProbe("ptGanglionOFFP5.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
     //GanglionOFF->insertProbe(ptprobeGanglionOFFP5);
     PointLIFProbe * ptprobeAmacrineOFFP5 = new PointLIFProbe("ptAmacrineOFFP5.txt",AmacrineOFF,locX/4., locY/4., locF, "AmacrineOFF:");
     //AmacrineOFF->insertProbe(ptprobeAmacrineOFFP5);


    locX = 128+15;
    locY = 128+15;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP7 = new PointLIFProbe("ptConeP7.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeP7);
    PointLIFProbe * ptprobeHorizontalP7 = new PointLIFProbe("ptHorizontalP7.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    //Horizontal->insertProbe(ptprobeHorizontalP7);
    PointLIFProbe * ptprobeBipolarONP7 = new PointLIFProbe("ptBipolarONP7.txt",BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarONP7);
    PointLIFProbe * ptprobeGanglionONP7 = new PointLIFProbe("ptGanglionONP7.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONP7);
    PointLIFProbe * ptprobeAmacrineONP7 = new PointLIFProbe("ptAmacrineONP7.txt",AmacrineON,locX/4., locY/4., locF, "AmacrineON:");
    //AmacrineON->insertProbe(ptprobeAmacrineONP7);

    PointLIFProbe * ptprobeBipolarOFFP7 = new PointLIFProbe("ptBipolarOFFP7.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    //BipolarOFF->insertProbe(ptprobeBipolarOFFP7);
    PointLIFProbe * ptprobeGanglionOFFP7 = new PointLIFProbe("ptGanglionOFFP7.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    //GanglionOFF->insertProbe(ptprobeGanglionOFFP7);
    PointLIFProbe * ptprobeAmacrineOFFP7 = new PointLIFProbe("ptAmacrineOFFP7.txt",AmacrineOFF,locX/4., locY/4., locF, "AmacrineOFF:");
    //AmacrineOFF->insertProbe(ptprobeAmacrineOFFP7);


    locX = 128+15;
    locY = 128-15;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP9 = new PointLIFProbe("ptConeP9.txt",Cone,locX, locY, locF, "Cone:");
    //Cone->insertProbe(ptprobeConeP9);
    PointLIFProbe * ptprobeHorizontalP9 = new PointLIFProbe("ptHorizontalP9.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    //Horizontal->insertProbe(ptprobeHorizontalP9);
    PointLIFProbe * ptprobeBipolarONP9 = new PointLIFProbe("ptBipolarONP9.txt",BipolarON,locX, locY, locF, "BipolarON:");
    //BipolarON->insertProbe(ptprobeBipolarONP9);
    PointLIFProbe * ptprobeGanglionONP9 = new PointLIFProbe("ptGanglionONP9.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    //GanglionON->insertProbe(ptprobeGanglionONP9);
    PointLIFProbe * ptprobeAmacrineONP9 = new PointLIFProbe("ptAmacrineONP9.txt",AmacrineON,locX/4., locY/4., locF, "AmacrineON:");
    //AmacrineON->insertProbe(ptprobeAmacrineONP9);

    PointLIFProbe * ptprobeBipolarOFFP9 = new PointLIFProbe("ptBipolarOFFP9.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    //BipolarOFF->insertProbe(ptprobeBipolarOFFP9);
    PointLIFProbe * ptprobeGanglionOFFP9 = new PointLIFProbe("ptGanglionOFFP9.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    //GanglionOFF->insertProbe(ptprobeGanglionOFFP9);
    PointLIFProbe * ptprobeAmacrineOFFP9 = new PointLIFProbe("ptAmacrineOFFP9.txt",AmacrineOFF,locX/4., locY/4., locF, "AmacrineOFF:");
    //AmacrineOFF->insertProbe(ptprobeAmacrineOFFP9);


    //



  // ---- calibration probes ON

       locY = 225;      // probing the bottom of the bar
       locF = 0;        // feature 0

       locX = ptb;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarONB = new PointLIFProbe("ptBipolarONB.txt",BipolarON,locX, locY, locF, "BipolarON:");
       //BipolarON->insertProbe(ptprobeBipolarONB);

       locX = pt10;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON10 = new PointLIFProbe("ptBipolarON10.txt",BipolarON,locX, locY, locF, "BipolarON:");
       //BipolarON->insertProbe(ptprobeBipolarON10);

       locX = pt08;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON08 = new PointLIFProbe("ptBipolarON08.txt",BipolarON,locX, locY, locF, "BipolarON:");
       //BipolarON->insertProbe(ptprobeBipolarON08);

       locX = pt06;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON06 = new PointLIFProbe("ptBipolarON06.txt",BipolarON,locX, locY, locF, "BipolarON:");
       //BipolarON->insertProbe(ptprobeBipolarON06);

       locX = pt04;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON04 = new PointLIFProbe("ptBipolarON04.txt",BipolarON,locX, locY, locF, "BipolarON:");
       //BipolarON->insertProbe(ptprobeBipolarON04);

       locX = pt02;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON02 = new PointLIFProbe("ptBipolarON02.txt",BipolarON,locX, locY, locF, "BipolarON:");
       //BipolarON->insertProbe(ptprobeBipolarON02);

     //

       // ---- calibration probes NSCALE IS 2

          locY = 225/2.;      // probing the middle of the bar
          locF = 0;        // feature 0

          locX = ptb/2.;    // mid between the first two bars
          PointLIFProbe * ptprobeHorizontalB = new PointLIFProbe("ptHorizontalB.txt",Horizontal,locX, locY, locF, "Horizontal:");
          //Horizontal->insertProbe(ptprobeHorizontalB);

          locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal10 = new PointLIFProbe("ptHorizontal10.txt",Horizontal,locX, locY, locF, "Horizontal:");
          //Horizontal->insertProbe(ptprobeHorizontal10);

          locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal08 = new PointLIFProbe("ptHorizontal08.txt",Horizontal,locX, locY, locF, "Horizontal:");
          //Horizontal->insertProbe(ptprobeHorizontal08);

          locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal06 = new PointLIFProbe("ptHorizontal06.txt",Horizontal,locX, locY, locF, "Horizontal:");
          //Horizontal->insertProbe(ptprobeHorizontal06);

          locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal04 = new PointLIFProbe("ptHorizontal04.txt",Horizontal,locX, locY, locF, "Horizontal:");
          //Horizontal->insertProbe(ptprobeHorizontal04);

          locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal02 = new PointLIFProbe("ptHorizontal02.txt",Horizontal,locX, locY, locF, "Horizontal:");
          //Horizontal->insertProbe(ptprobeHorizontal02);

        //


          // ---- calibration probes NSCALE IS 2

             locY = 225/2.;      // probing the middle of the bar
             locF = 0;        // feature 0

             locX = ptb/2.;    // mid between the first two bars
             PointLIFProbe * ptprobeGanglionONB = new PointLIFProbe("ptGanglionONB.txt",GanglionON,locX, locY, locF, "GanglionON:");
             //GanglionON->insertProbe(ptprobeGanglionONB);

             locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON10 = new PointLIFProbe("ptGanglionON10.txt",GanglionON,locX, locY, locF, "GanglionON:");
             //GanglionON->insertProbe(ptprobeGanglionON10);

             locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON08 = new PointLIFProbe("ptGanglionON08.txt",GanglionON,locX, locY, locF, "GanglionON:");
             GanglionON->insertProbe(ptprobeGanglionON08);

             locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON06 = new PointLIFProbe("ptGanglionON06.txt",GanglionON,locX, locY, locF, "GanglionON:");
             //GanglionON->insertProbe(ptprobeGanglionON06);

             locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON04 = new PointLIFProbe("ptGanglionON04.txt",GanglionON,locX, locY, locF, "GanglionON:");
             //GanglionON->insertProbe(ptprobeGanglionON04);

             locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON02 = new PointLIFProbe("ptGanglionON02.txt",GanglionON,locX, locY, locF, "GanglionON:");
             //GanglionON->insertProbe(ptprobeGanglionON02);

           //

             // ---- calibration probes NSCALE IS 4 --- 0.25

                locY = 225/4.;      // probing the middle of the bar
                locF = 0;        // feature 0

                locX = ptb/4.;    // mid between the first two bars
                PointLIFProbe * ptprobeAmacrineONB = new PointLIFProbe("ptAmacrineONB.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineONB);

                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeAmacrineON10 = new PointLIFProbe("ptAmacrineON10.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineON10);

                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeAmacrineON08 = new PointLIFProbe("ptAmacrineON08.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineON08);

                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeAmacrineON06 = new PointLIFProbe("ptAmacrineON06.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineON06);

                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeAmacrineON04 = new PointLIFProbe("ptAmacrineON04.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineON04);

                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeAmacrineON02 = new PointLIFProbe("ptAmacrineON02.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineON02);

                // ---- calibration probes NSCALE IS 4

                locX = 	60/4.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeAmacrineONRU1 = new PointLIFProbe("ptAmacrineONRU1.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobeAmacrineONRD1 = new PointLIFProbe("ptAmacrineONRD1.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineONRD1);


                locY =  75;      // probing
                PointLIFProbe * ptprobeAmacrineONRU2 = new PointLIFProbe("ptAmacrineONRU2.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobeAmacrineONRD2 = new PointLIFProbe("ptAmacrineONRD2.txt",AmacrineON,locX, locY, locF, "AmacrineON:");
                //AmacrineON->insertProbe(ptprobeAmacrineONRD2);




                // ---- calibration probes NSCALE IS 2

                locX = 	60/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeGanglionONRU1 = new PointLIFProbe("ptGanglionONRU1.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobeGanglionONRD1 = new PointLIFProbe("ptGanglionONRD1.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRD1);


                locY =  75;      // probing
                PointLIFProbe * ptprobeGanglionONRU2 = new PointLIFProbe("ptGanglionONRU2.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobeGanglionONRD2 = new PointLIFProbe("ptGanglionONRD2.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRD2);


                locX = 	58/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeGanglionONRU1l = new PointLIFProbe("ptGanglionONRU1l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRU1l);

                locY =  44;      // probing
                PointLIFProbe * ptprobeGanglionONRD1l = new PointLIFProbe("ptGanglionONRD1l.txt",GanglionON,locX, locY, locF, "GanglionON:");
               //GanglionON->insertProbe(ptprobeGanglionONRD1l);


                locY =  75;      // probing
                PointLIFProbe * ptprobeGanglionONRU2l = new PointLIFProbe("ptGanglionONRU2l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRU2l);

                locY =  104;      // probing
                PointLIFProbe * ptprobeGanglionONRD2l = new PointLIFProbe("ptGanglionONRD2l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRD2l);



                locX = 	62/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeGanglionONRU1r = new PointLIFProbe("ptGanglionONRU1r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRU1r);

                locY =  44;      // probing
                PointLIFProbe * ptprobeGanglionONRD1r = new PointLIFProbe("ptGanglionONRD1r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRD1r);


                locY =  75;      // probing
                PointLIFProbe * ptprobeGanglionONRU2r = new PointLIFProbe("ptGanglionONRU2r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRU2r);

                locY =  104;      // probing
                PointLIFProbe * ptprobeGanglionONRD2r = new PointLIFProbe("ptGanglionONRD2r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                //GanglionON->insertProbe(ptprobeGanglionONRD2r);


                //------- SynchronicityON layer
                locX = 	60/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeSynchronicityONRU1 = new PointLIFProbe("ptSynchronicityONRU1.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                //SynchronicityON->insertProbe(ptprobeSynchronicityONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobeSynchronicityONRD1 = new PointLIFProbe("ptSynchronicityONRD1.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                //SynchronicityON->insertProbe(ptprobeSynchronicityONRD1);


                locY =  75;      // probing
                PointLIFProbe * ptprobeSynchronicityONRU2 = new PointLIFProbe("ptSynchronicityONRU2.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                //SynchronicityON->insertProbe(ptprobeSynchronicityONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobeSynchronicityONRD2 = new PointLIFProbe("ptSynchronicityONRD2.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                //SynchronicityON->insertProbe(ptprobeSynchronicityONRD2);

                // ---- calibration probes

                       locY = 225;      // probing the bottom of the bar
                       locF = 0;        // feature 0

                       locX = ptb;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFFB = new PointLIFProbe("ptBipolarOFFB.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       //BipolarOFF->insertProbe(ptprobeBipolarOFFB);

                       locX = pt10;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF10 = new PointLIFProbe("ptBipolarOFF10.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       //BipolarOFF->insertProbe(ptprobeBipolarOFF10);

                       locX = pt08;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF08 = new PointLIFProbe("ptBipolarOFF08.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       //BipolarOFF->insertProbe(ptprobeBipolarOFF08);

                       locX = pt06;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF06 = new PointLIFProbe("ptBipolarOFF06.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       //BipolarOFF->insertProbe(ptprobeBipolarOFF06);

                       locX = pt04;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF04 = new PointLIFProbe("ptBipolarOFF04.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       //BipolarOFF->insertProbe(ptprobeBipolarOFF04);

                       locX = pt02;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF02 = new PointLIFProbe("ptBipolarOFF02.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       //BipolarOFF->insertProbe(ptprobeBipolarOFF02);

                     //

                       // ---- calibration probes NSCALE IS 2
                        // ---- calibration probes NSCALE IS 2

                             locY = 225/2.;      // probing the middle of the bar
                             locF = 0;        // feature 0

                             locX = ptb/2.;    // mid between the first two bars
                             PointLIFProbe * ptprobeGanglionOFFB = new PointLIFProbe("ptGanglionOFFB.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             //GanglionOFF->insertProbe(ptprobeGanglionOFFB);

                             locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF10 = new PointLIFProbe("ptGanglionOFF10.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             //GanglionOFF->insertProbe(ptprobeGanglionOFF10);

                             locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF08 = new PointLIFProbe("ptGanglionOFF08.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             //GanglionOFF->insertProbe(ptprobeGanglionOFF08);

                             locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF06 = new PointLIFProbe("ptGanglionOFF06.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             //GanglionOFF->insertProbe(ptprobeGanglionOFF06);

                             locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF04 = new PointLIFProbe("ptGanglionOFF04.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             //GanglionOFF->insertProbe(ptprobeGanglionOFF04);

                             locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF02 = new PointLIFProbe("ptGanglionOFF02.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             //GanglionOFF->insertProbe(ptprobeGanglionOFF02);

                           //

                             // ---- calibration probes NSCALE IS 4 --- 0.25

                                locY = 225/4.;      // probing the middle of the bar
                                locF = 0;        // feature 0

                                locX = ptb/4.;    // mid between the first two bars
                                PointLIFProbe * ptprobeAmacrineOFFB = new PointLIFProbe("ptAmacrineOFFB.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFFB);

                                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeAmacrineOFF10 = new PointLIFProbe("ptAmacrineOFF10.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFF10);

                                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeAmacrineOFF08 = new PointLIFProbe("ptAmacrineOFF08.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFF08);

                                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeAmacrineOFF06 = new PointLIFProbe("ptAmacrineOFF06.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFF06);

                                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeAmacrineOFF04 = new PointLIFProbe("ptAmacrineOFF04.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFF04);

                                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeAmacrineOFF02 = new PointLIFProbe("ptAmacrineOFF02.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFF02);

                                // ---- calibration probes NSCALE IS 4

                                locX = 	60/4.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeAmacrineOFFRU1 = new PointLIFProbe("ptAmacrineOFFRU1.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeAmacrineOFFRD1 = new PointLIFProbe("ptAmacrineOFFRD1.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeAmacrineOFFRU2 = new PointLIFProbe("ptAmacrineOFFRU2.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeAmacrineOFFRD2 = new PointLIFProbe("ptAmacrineOFFRD2.txt",AmacrineOFF,locX, locY, locF, "AmacrineOFF:");
                                //AmacrineOFF->insertProbe(ptprobeAmacrineOFFRD2);




                                // ---- calibration probes NSCALE IS 2

                                locX = 	60/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU1 = new PointLIFProbe("ptGanglionOFFRU1.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD1 = new PointLIFProbe("ptGanglionOFFRD1.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU2 = new PointLIFProbe("ptGanglionOFFRU2.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD2 = new PointLIFProbe("ptGanglionOFFRD2.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRD2);


                                locX = 	58/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU1l = new PointLIFProbe("ptGanglionOFFRU1l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRU1l);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD1l = new PointLIFProbe("ptGanglionOFFRD1l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRD1l);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU2l = new PointLIFProbe("ptGanglionOFFRU2l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRU2l);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD2l = new PointLIFProbe("ptGanglionOFFRD2l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRD2l);



                                locX = 	62/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU1r = new PointLIFProbe("ptGanglionOFFRU1r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRU1r);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD1r = new PointLIFProbe("ptGanglionOFFRD1r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRD1r);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU2r = new PointLIFProbe("ptGanglionOFFRU2r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRU2r);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD2r = new PointLIFProbe("ptGanglionOFFRD2r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                //GanglionOFF->insertProbe(ptprobeGanglionOFFRD2r);


                                //------- SynchronicityOFF layer
                                locX = 	60/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRU1 = new PointLIFProbe("ptSynchronicityOFFRU1.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                //SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRD1 = new PointLIFProbe("ptSynchronicityOFFRD1.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                //SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRU2 = new PointLIFProbe("ptSynchronicityOFFRU2.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                //SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRD2 = new PointLIFProbe("ptSynchronicityOFFRD2.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                //SynchronicityOFF->insertProbe(ptprobeSynchronicityOFFRD2);



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

