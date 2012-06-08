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
    HyPerLayer * WFAmacrineON      = hc->getLayerFromName("WFAmacrineON");
     if (WFAmacrineON == NULL) {fprintf(stdout,"Can't find WFAmacrineON pointer"); exit(-1);};
     HyPerLayer * PAAmacrineON      = hc->getLayerFromName("PAAmacrineON");
      if (PAAmacrineON == NULL) {fprintf(stdout,"Can't find PAAmacrineON pointer"); exit(-1);};
       HyPerLayer * SynchronicityON = hc->getLayerFromName("SynchronicityON"); // for analysis
    if (SynchronicityON == NULL) {fprintf(stdout,"Can't find SynchronicityON pointer"); exit(-1);};
    HyPerLayer * RetinaON      = hc->getLayerFromName("RetinaON");
    if (RetinaON == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};

    HyPerLayer * SFAmacrine      = hc->getLayerFromName("SFAmacrine");
     if (SFAmacrine == NULL) {fprintf(stdout,"Can't find SFAmacrine pointer"); exit(-1);};



    HyPerLayer * BipolarOFF       = hc->getLayerFromName("BipolarOFF");
    if (BipolarOFF == NULL) {fprintf(stdout,"Can't find BipolarOFF pointer"); exit(-1);};
    HyPerLayer * GanglionOFF      = hc->getLayerFromName("GanglionOFF");
    if (GanglionOFF == NULL) {fprintf(stdout,"Can't find GanglionOFF pointer"); exit(-1);};
    HyPerLayer * WFAmacrineOFF      = hc->getLayerFromName("WFAmacrineOFF");
    if (WFAmacrineOFF == NULL) {fprintf(stdout,"Can't find WFAmacrineOFF pointer"); exit(-1);};
    HyPerLayer * PAAmacrineOFF      = hc->getLayerFromName("PAAmacrineOFF");
    if (PAAmacrineOFF == NULL) {fprintf(stdout,"Can't find PAAmacrineOFF pointer"); exit(-1);};

    HyPerLayer * SynchronicityOFF = hc->getLayerFromName("SynchronicityOFF"); // for analysis
     if (SynchronicityOFF == NULL) {fprintf(stdout,"Can't find SynchronicityOFF pointer"); exit(-1);};
    HyPerLayer * RetinaOFF      = hc->getLayerFromName("RetinaOFF");
    if (RetinaOFF == NULL) {fprintf(stdout,"Can't find Cone pointer"); exit(-1);};


    int locX = 128;
    int locY = 128;      // probing the center
    int locF = 0;        // feature 0

    PointLIFProbe * ptprobeCone = new PointLIFProbe("ptCone.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeCone);

    PointLIFProbe * ptprobeConeScr = new PointLIFProbe(Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeScr);

    StatsProbe * statsCone = new StatsProbe("statsCone.txt",Cone,BufActivity,"Cone:");
    assert(statsCone);
 //----------------------------------------------------------------------
    PointLIFProbe * ptprobeBipolarON = new PointLIFProbe("ptBipolarON.txt",BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarON);
    PointLIFProbe * ptprobeBipolarONSrc = new PointLIFProbe(BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarONSrc);

    StatsProbe * statsBipolarON = new StatsProbe("statsBipolarON.txt",BipolarON,BufActivity,"BipolarON:");
    assert(statsBipolarON);
 //-----------------------------------------------------------------------
    PointLIFProbe * ptprobeHorizontal = new PointLIFProbe("ptHorizontal.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    assert(ptprobeHorizontal);

    StatsProbe * statsHorizontal = new StatsProbe("statsHorizontal.txt",Horizontal,BufActivity,"Horizontal:");
    assert(statsHorizontal);
 //----------------------------------------------------------------------
    PointLIFProbe * ptprobeGanglionON = new PointLIFProbe("ptGanglionON.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionON);
    PointLIFProbe * ptprobeGanglionONSrc = new PointLIFProbe(GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionONSrc);

    StatsProbe * statsGanglionON = new StatsProbe("statsGanglionON.txt",GanglionON,BufActivity,"GanglionON:");
    assert(statsGanglionON);
    StatsProbe * statsGanglionONScr = new StatsProbe(GanglionON,BufActivity,"GanglionON:");
    assert(statsGanglionONScr);

 //----------------------------------------------------------------------
    PointLIFProbe * ptprobeWFAmacrineON = new PointLIFProbe("ptWFAmacrineON.txt",WFAmacrineON,locX/4., locY/4., locF, "WFAmacrineON:");
    assert(ptprobeWFAmacrineON);
     StatsProbe * statsWFAmacrineON = new StatsProbe("statsWFAmacrineON.txt",WFAmacrineON,BufActivity,"WFAmacrineON:");
     assert(statsWFAmacrineON);
     PointLIFProbe * ptprobePAAmacrineON = new PointLIFProbe("ptPAAmacrineON.txt",PAAmacrineON,locX/4., locY/4., locF, "PAAmacrineON:");
     assert(ptprobePAAmacrineON);
      StatsProbe * statsPAAmacrineON = new StatsProbe("statPAAmacrineON.txt",PAAmacrineON,BufActivity,"PAAmacrineON:");
      assert(statsPAAmacrineON);
      PointLIFProbe * ptprobeSFAmacrine = new PointLIFProbe("ptSFAmacrine.txt",SFAmacrine,locX/4., locY/4., locF, "SFAmacrine:");
      assert(ptprobeSFAmacrine);
       StatsProbe * statsSFAmacrine = new StatsProbe("statsSFAmacrine.txt",SFAmacrine,BufActivity,"SFAmacrine:");
       assert(statsSFAmacrine);
       //----------------------------------------------------------------------

    StatsProbe * statsSynchronicityON = new StatsProbe(SynchronicityON, BufActivity,"SynchronicityON:");
    assert(statsSynchronicityON);
    //----------------------------------------------------------------------


    PointProbe * ptprobeRetinaON = new PointProbe("ptRetinaON.txt",RetinaON,locX, locY, locF, "RetinaON:");
    assert(ptprobeRetinaON);

    StatsProbe * statsRetinaON = new StatsProbe("statsRetinaON.txt",RetinaON,BufActivity,"RetinaON:");
    assert(statsRetinaON);

    StatsProbe * statsRetinaONSrc = new StatsProbe(RetinaON,BufActivity,"RetinaON:");
    assert(statsRetinaONSrc);

    //----------------------------------------------------------------------
        PointLIFProbe * ptprobeBipolarOFF = new PointLIFProbe("ptBipolarOFF.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
        assert(ptprobeBipolarOFF);
        PointLIFProbe * ptprobeBipolarOFFSrc = new PointLIFProbe(BipolarOFF,locX, locY, locF, "BipolarOFF:");
        assert(ptprobeBipolarOFFSrc);

        StatsProbe * statsBipolarOFF = new StatsProbe("statsBipolarOFF.txt",BipolarOFF,BufActivity,"BipolarOFF:");
        assert(statsBipolarOFF);
     //-----------------------------------------------------------------------
        PointLIFProbe * ptprobeGanglionOFF = new PointLIFProbe("ptGanglionOFF.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
        assert(ptprobeGanglionOFF);
        PointLIFProbe * ptprobeGanglionOFFSrc = new PointLIFProbe(GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
        assert(ptprobeGanglionOFFSrc);

        StatsProbe * statsGanglionOFF = new StatsProbe("statsGanglionOFF.txt",GanglionOFF,BufActivity,"GanglionOFF:");
        assert(statsGanglionOFF);
        StatsProbe * statsGanglionOFFScr = new StatsProbe(GanglionOFF,BufActivity,"GanglionOFF:");
        assert(statsGanglionOFFScr);

     //----------------------------------------------------------------------
        PointLIFProbe * ptprobeWFAmacrineOFF = new PointLIFProbe("ptWFAmacrineOFF.txt",WFAmacrineOFF,locX/4., locY/4., locF, "WFAmacrineOFF:");
        assert(ptprobeWFAmacrineOFF);
        StatsProbe * statsWFAmacrineOFF = new StatsProbe("statsWFAmacrineOFF.txt",WFAmacrineOFF,BufActivity,"WFAmacrineOFF:");
        assert(statsWFAmacrineOFF);
        PointLIFProbe * ptprobePAAmacrineOFF = new PointLIFProbe("ptPAAmacrineOFF.txt",PAAmacrineOFF,locX/4., locY/4., locF, "PAAmacrineOFF:");
        assert(ptprobePAAmacrineOFF);
        StatsProbe * statsPAAmacrineOFF = new StatsProbe("statsPAAmacrineOFF.txt",PAAmacrineOFF,BufActivity,"PAAmacrineOFF:");
        assert(statsPAAmacrineOFF);

        //----------------------------------------------------------------------

        StatsProbe * statsSynchronicityOFF = new StatsProbe(SynchronicityOFF,BufActivity,"SynchronicityOFF:");
        assert(statsSynchronicityOFF);
        //----------------------------------------------------------------------


        PointProbe * ptprobeRetinaOFF = new PointProbe("ptRetinaOFF.txt",RetinaOFF,locX, locY, locF, "RetinaOFF:");
        assert(ptprobeRetinaOFF);

        StatsProbe * statsRetinaOFF = new StatsProbe("statsRetinaOFF.txt",RetinaOFF,BufActivity,"RetinaOFF:");
        assert(statsRetinaOFF);

        StatsProbe * statsRetinaOFFSrc = new StatsProbe(RetinaOFF,BufActivity,"RetinaOFF:");
        assert(statsRetinaOFFSrc);


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
    //const int pt00 = 244;  // 00 %


    locX = ptb;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeConeB = new PointLIFProbe("ptConeB.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeB);

    locX = pt10;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone10 = new PointLIFProbe("ptCone10.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeCone10);

    locX = pt08;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone08 = new PointLIFProbe("ptCone08.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeCone08);

    locX = pt06;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone06 = new PointLIFProbe("ptCone06.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeCone06);

    locX = pt04;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone04 = new PointLIFProbe("ptCone04.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeCone04);

    locX = pt02;    // 3 + 12.8 /2 + n * 25.6
    PointLIFProbe * ptprobeCone02 = new PointLIFProbe("ptCone02.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeCone02);

  //--------------------------------------------------------------------------------------------


    locX = 128;
    locY = 128;   // probing the center DOT !
    PointLIFProbe * ptprobeConeP1 = new PointLIFProbe("ptConeP1.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeP1);
    PointLIFProbe * ptprobeHorizontalP1 = new PointLIFProbe("ptHorizontalP1.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    assert(ptprobeHorizontalP1);
    PointLIFProbe * ptprobeBipolarONP1 = new PointLIFProbe("ptBipolarONP1.txt",BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarONP1);
    PointLIFProbe * ptprobeGanglionONP1 = new PointLIFProbe("ptGanglionONP1.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionONP1);
    PointLIFProbe * ptprobeWFAmacrineONP1 = new PointLIFProbe("ptWFAmacrineONP1.txt",WFAmacrineON,locX/4., locY/4., locF, "WFAmacrineON:");
    assert(ptprobeWFAmacrineONP1);
    PointLIFProbe * ptprobePAAmacrineONP1 = new PointLIFProbe("ptPAAmacrineONP1.txt",PAAmacrineON,locX/4., locY/4., locF, "PAAmacrineON:");
    assert(ptprobePAAmacrineONP1);
    PointLIFProbe * ptprobeSFAmacrineP1 = new PointLIFProbe("ptSFAmacrineP1.txt",SFAmacrine,locX/4., locY/4., locF, "SFAmacrine:");
    assert(ptprobeSFAmacrineP1);

    PointLIFProbe * ptprobeBipolarOFFP1 = new PointLIFProbe("ptBipolarOFFP1.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    assert(ptprobeBipolarOFFP1);
    PointLIFProbe * ptprobeGanglionOFFP1 = new PointLIFProbe("ptGanglionOFFP1.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    assert(ptprobeGanglionOFFP1);
    PointLIFProbe * ptprobeWFAmacrineOFFP1 = new PointLIFProbe("ptWFAmacrineOFFP1.txt",WFAmacrineOFF,locX/4., locY/4., locF, "WFAmacrineOFF:");
    assert(ptprobeWFAmacrineOFFP1);
    PointLIFProbe * ptprobePAAmacrineOFFP1 = new PointLIFProbe("ptPAAmacrineOFFP1.txt",PAAmacrineOFF,locX/4., locY/4., locF, "PAAmacrineOFF:");
    assert(ptprobePAAmacrineOFFP1);



    locX = 128-15.;
    locY = 128+15.;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP3 = new PointLIFProbe("ptConeP3.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeP3);
    PointLIFProbe * ptprobeHorizontalP3 = new PointLIFProbe("ptHorizontalP3.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    assert(ptprobeHorizontalP3);
    PointLIFProbe * ptprobeBipolarONP3 = new PointLIFProbe("ptBipolarONP3.txt",BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarONP3);
    PointLIFProbe * ptprobeGanglionONP3 = new PointLIFProbe("ptGanglionONP3.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionONP3);
    PointLIFProbe * ptprobeWFAmacrineONP3 = new PointLIFProbe("ptWFAmacrineONP3.txt",WFAmacrineON,locX/4., locY/4., locF, "WFAmacrineON:");
    assert(ptprobeWFAmacrineONP3);
    PointLIFProbe * ptprobePAAmacrineONP3 = new PointLIFProbe("ptPAAmacrineONP3.txt",PAAmacrineON,locX/4., locY/4., locF, "PAAmacrineON:");
    assert(ptprobePAAmacrineONP3);
    PointLIFProbe * ptprobeSFAmacrineP3 = new PointLIFProbe("ptSFAmacrineP3.txt",SFAmacrine,locX/4., locY/4., locF, "SFAmacrine:");
    assert(ptprobeSFAmacrineP3);
    PointLIFProbe * ptprobeBipolarOFFP3 = new PointLIFProbe("ptBipolarOFFP3.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    assert(ptprobeBipolarOFFP3);
    PointLIFProbe * ptprobeGanglionOFFP3 = new PointLIFProbe("ptGanglionOFFP3.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    assert(ptprobeGanglionOFFP3);
    PointLIFProbe * ptprobeWFAmacrineOFFP3 = new PointLIFProbe("ptWFAmacrineOFFP3.txt",WFAmacrineOFF,locX/4., locY/4., locF, "WFAmacrineOFF:");
    assert(ptprobeWFAmacrineOFFP3);
    PointLIFProbe * ptprobePAAmacrineOFFP3 = new PointLIFProbe("ptPAAmacrineOFFP3.txt",PAAmacrineOFF,locX/4., locY/4., locF, "PAAmacrineOFF:");
    assert(ptprobePAAmacrineOFFP3);


    locX = 128-15;
    locY = 128-15;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP5 = new PointLIFProbe("ptConeP5.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeP5);
    PointLIFProbe * ptprobeHorizontalP5 = new PointLIFProbe("ptHorizontalP5.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    assert(ptprobeHorizontalP5);
    PointLIFProbe * ptprobeBipolarONP5 = new PointLIFProbe("ptBipolarONP5.txt",BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarONP5);
    PointLIFProbe * ptprobeGanglionONP5 = new PointLIFProbe("ptGanglionONP5.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionONP5);
    PointLIFProbe * ptprobeWFAmacrineONP5 = new PointLIFProbe("ptWFAmacrineONP5.txt",WFAmacrineON,locX/4., locY/4., locF, "WFAmacrineON:");
    assert(ptprobeWFAmacrineONP5);
    PointLIFProbe * ptprobePAAmacrineONP5 = new PointLIFProbe("ptPAAmacrineONP5.txt",PAAmacrineON,locX/4., locY/4., locF, "PAAmacrineON:");
    assert(ptprobePAAmacrineONP5);
    PointLIFProbe * ptprobeSFAmacrineP5 = new PointLIFProbe("ptSFAmacrineP5.txt",SFAmacrine,locX/4., locY/4., locF, "SFAmacrine:");
    assert(ptprobeSFAmacrineP5);

     PointLIFProbe * ptprobeBipolarOFFP5 = new PointLIFProbe("ptBipolarOFFP5.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
     assert(ptprobeBipolarOFFP5);
     PointLIFProbe * ptprobeGanglionOFFP5 = new PointLIFProbe("ptGanglionOFFP5.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
     assert(ptprobeGanglionOFFP5);
     PointLIFProbe * ptprobeWFAmacrineOFFP5 = new PointLIFProbe("ptWFAmacrineOFFP5.txt",WFAmacrineOFF,locX/4., locY/4., locF, "WFAmacrineOFF:");
     assert(ptprobeWFAmacrineOFFP5);
     PointLIFProbe * ptprobePAAmacrineOFFP5 = new PointLIFProbe("ptPAAmacrineOFFP5.txt",PAAmacrineOFF,locX/4., locY/4., locF, "PAAmacrineOFF:");
     assert(ptprobePAAmacrineOFFP5);



    locX = 128+15;
    locY = 128+15;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP7 = new PointLIFProbe("ptConeP7.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeP7);
    PointLIFProbe * ptprobeHorizontalP7 = new PointLIFProbe("ptHorizontalP7.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    assert(ptprobeHorizontalP7);
    PointLIFProbe * ptprobeBipolarONP7 = new PointLIFProbe("ptBipolarONP7.txt",BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarONP7);
    PointLIFProbe * ptprobeGanglionONP7 = new PointLIFProbe("ptGanglionONP7.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionONP7);
    PointLIFProbe * ptprobeWFAmacrineONP7 = new PointLIFProbe("ptWFAmacrineONP7.txt",WFAmacrineON,locX/4., locY/4., locF, "WFAmacrineON:");
    assert(ptprobeWFAmacrineONP7);
    PointLIFProbe * ptprobePAAmacrineONP7 = new PointLIFProbe("ptPAAmacrineONP7.txt",PAAmacrineON,locX/4., locY/4., locF, "PAAmacrineON:");
    assert(ptprobePAAmacrineONP7);
    PointLIFProbe * ptprobeAmacrineP7 = new PointLIFProbe("ptSFAmacrineP7.txt",SFAmacrine,locX/4., locY/4., locF, "SFAmacrine:");
    assert(ptprobeAmacrineP7);
    PointLIFProbe * ptprobeBipolarOFFP7 = new PointLIFProbe("ptBipolarOFFP7.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    assert(ptprobeBipolarOFFP7);
    PointLIFProbe * ptprobeGanglionOFFP7 = new PointLIFProbe("ptGanglionOFFP7.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    assert(ptprobeGanglionOFFP7);
    PointLIFProbe * ptprobeWFAmacrineOFFP7 = new PointLIFProbe("ptWFAmacrineOFFP7.txt",WFAmacrineOFF,locX/4., locY/4., locF, "WFAmacrineOFF:");
    assert(ptprobeWFAmacrineOFFP7);
    PointLIFProbe * ptprobePAAmacrineOFFP7 = new PointLIFProbe("ptPAAmacrineOFFP7.txt",PAAmacrineOFF,locX/4., locY/4., locF, "PAAmacrineOFF:");
    assert(ptprobePAAmacrineOFFP7);




    locX = 128+15;
    locY = 128-15;   // probing the four surround patches
    PointLIFProbe * ptprobeConeP9 = new PointLIFProbe("ptConeP9.txt",Cone,locX, locY, locF, "Cone:");
    assert(ptprobeConeP9);
    PointLIFProbe * ptprobeHorizontalP9 = new PointLIFProbe("ptHorizontalP9.txt",Horizontal,locX/2., locY/2., locF, "Horizontal:");
    assert(ptprobeHorizontalP9);
    PointLIFProbe * ptprobeBipolarONP9 = new PointLIFProbe("ptBipolarONP9.txt",BipolarON,locX, locY, locF, "BipolarON:");
    assert(ptprobeBipolarONP9);
    PointLIFProbe * ptprobeGanglionONP9 = new PointLIFProbe("ptGanglionONP9.txt",GanglionON,locX/2., locY/2., locF, "GanglionON:");
    assert(ptprobeGanglionONP9);
    PointLIFProbe * ptprobeWFAmacrineONP9 = new PointLIFProbe("ptWFAmacrineONP9.txt",WFAmacrineON,locX/4., locY/4., locF, "WFAmacrineON:");
    assert(ptprobeWFAmacrineONP9);
    PointLIFProbe * ptprobePAAmacrineONP9 = new PointLIFProbe("ptPAAmacrineONP9.txt",PAAmacrineON,locX/4., locY/4., locF, "PAAmacrineON:");
    assert(ptprobePAAmacrineONP9);
    PointLIFProbe * ptprobeSFAmacrineP9 = new PointLIFProbe("ptSFAmacrineP9.txt",SFAmacrine,locX/4., locY/4., locF, "SFAmacrine:");
    assert(ptprobeSFAmacrineP9);

    PointLIFProbe * ptprobeBipolarOFFP9 = new PointLIFProbe("ptBipolarOFFP9.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
    assert(ptprobeBipolarOFFP9);
    PointLIFProbe * ptprobeGanglionOFFP9 = new PointLIFProbe("ptGanglionOFFP9.txt",GanglionOFF,locX/2., locY/2., locF, "GanglionOFF:");
    assert(ptprobeGanglionOFFP9);
    PointLIFProbe * ptprobeWFAmacrineOFFP9 = new PointLIFProbe("ptWFAmacrineOFFP9.txt",WFAmacrineOFF,locX/4., locY/4., locF, "WFAmacrineOFF:");
    assert(ptprobeWFAmacrineOFFP9);
    PointLIFProbe * ptprobePAAmacrineOFFP9 = new PointLIFProbe("ptPAAmacrineOFFP9.txt",PAAmacrineOFF,locX/4., locY/4., locF, "PAAmacrineOFF:");
    assert(ptprobePAAmacrineOFFP9);




    //



  // ---- calibration probes ON

       locY = 225;      // probing the bottom of the bar
       locF = 0;        // feature 0

       locX = ptb;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarONB = new PointLIFProbe("ptBipolarONB.txt",BipolarON,locX, locY, locF, "BipolarON:");
       assert(ptprobeBipolarONB);

       locX = pt10;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON10 = new PointLIFProbe("ptBipolarON10.txt",BipolarON,locX, locY, locF, "BipolarON:");
       assert(ptprobeBipolarON10);

       locX = pt08;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON08 = new PointLIFProbe("ptBipolarON08.txt",BipolarON,locX, locY, locF, "BipolarON:");
       assert(ptprobeBipolarON08);

       locX = pt06;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON06 = new PointLIFProbe("ptBipolarON06.txt",BipolarON,locX, locY, locF, "BipolarON:");
       assert(ptprobeBipolarON06);

       locX = pt04;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON04 = new PointLIFProbe("ptBipolarON04.txt",BipolarON,locX, locY, locF, "BipolarON:");
       assert(ptprobeBipolarON04);

       locX = pt02;    // 3 + 12.8 /2 + n * 25.6
       PointLIFProbe * ptprobeBipolarON02 = new PointLIFProbe("ptBipolarON02.txt",BipolarON,locX, locY, locF, "BipolarON:");
       assert(ptprobeBipolarON02);

     //

       // ---- calibration probes NSCALE IS 2

          locY = 225/2.;      // probing the middle of the bar
          locF = 0;        // feature 0

          locX = ptb/2.;    // mid between the first two bars
          PointLIFProbe * ptprobeHorizontalB = new PointLIFProbe("ptHorizontalB.txt",Horizontal,locX, locY, locF, "Horizontal:");
          assert(ptprobeHorizontalB);

          locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal10 = new PointLIFProbe("ptHorizontal10.txt",Horizontal,locX, locY, locF, "Horizontal:");
          assert(ptprobeHorizontal10);

          locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal08 = new PointLIFProbe("ptHorizontal08.txt",Horizontal,locX, locY, locF, "Horizontal:");
          assert(ptprobeHorizontal08);

          locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal06 = new PointLIFProbe("ptHorizontal06.txt",Horizontal,locX, locY, locF, "Horizontal:");
          assert(ptprobeHorizontal06);

          locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal04 = new PointLIFProbe("ptHorizontal04.txt",Horizontal,locX, locY, locF, "Horizontal:");
          assert(ptprobeHorizontal04);

          locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
          PointLIFProbe * ptprobeHorizontal02 = new PointLIFProbe("ptHorizontal02.txt",Horizontal,locX, locY, locF, "Horizontal:");
          assert(ptprobeHorizontal02);

        //


          // ---- calibration probes NSCALE IS 2

             locY = 225/2.;      // probing the middle of the bar
             locF = 0;        // feature 0

             locX = ptb/2.;    // mid between the first two bars
             PointLIFProbe * ptprobeGanglionONB = new PointLIFProbe("ptGanglionONB.txt",GanglionON,locX, locY, locF, "GanglionON:");
             assert(ptprobeGanglionONB);

             locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON10 = new PointLIFProbe("ptGanglionON10.txt",GanglionON,locX, locY, locF, "GanglionON:");
             assert(ptprobeGanglionON10);

             locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON08 = new PointLIFProbe("ptGanglionON08.txt",GanglionON,locX, locY, locF, "GanglionON:");
             assert(ptprobeGanglionON08);

             locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON06 = new PointLIFProbe("ptGanglionON06.txt",GanglionON,locX, locY, locF, "GanglionON:");
             assert(ptprobeGanglionON06);

             locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON04 = new PointLIFProbe("ptGanglionON04.txt",GanglionON,locX, locY, locF, "GanglionON:");
             assert(ptprobeGanglionON04);

             locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
             PointLIFProbe * ptprobeGanglionON02 = new PointLIFProbe("ptGanglionON02.txt",GanglionON,locX, locY, locF, "GanglionON:");
             assert(ptprobeGanglionON02);

           //

             // ---- calibration probes NSCALE IS 4 --- 0.25

                locY = 225/4.;      // probing the middle of the bar
                locF = 0;        // feature 0

                locX = ptb/4.;    // mid between the first two bars
                PointLIFProbe * ptprobeWFAmacrineONB = new PointLIFProbe("ptWFAmacrineONB.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineONB);

                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeWFAmacrineON10 = new PointLIFProbe("ptWFAmacrineON10.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineON10);

                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeWFAmacrineON08 = new PointLIFProbe("ptWFAmacrineON08.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineON08);

                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeWFAmacrineON06 = new PointLIFProbe("ptWFAmacrineON06.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineON06);

                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeWFAmacrineON04 = new PointLIFProbe("ptWFAmacrineON04.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineON04);

                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeWFAmacrineON02 = new PointLIFProbe("ptWFAmacrineON02.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineON02);

                // ---- calibration probes NSCALE IS 4

                locX = 	60/4.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeWFAmacrineONRU1 = new PointLIFProbe("ptWFAmacrineONRU1.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobeWFAmacrineONRD1 = new PointLIFProbe("ptWFAmacrineONRD1.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineONRD1);


                locY =  75;      // probing
                PointLIFProbe * ptprobeWFAmacrineONRU2 = new PointLIFProbe("ptWFAmacrineONRU2.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobeWFAmacrineONRD2 = new PointLIFProbe("ptWFAmacrineONRD2.txt",WFAmacrineON,locX, locY, locF, "WFAmacrineON:");
                assert(ptprobeWFAmacrineONRD2);



                PointLIFProbe * ptprobePAAmacrineONB = new PointLIFProbe("ptPAAmacrineONB.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineONB);

                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobePAAmacrineON10 = new PointLIFProbe("ptPAAmacrineON10.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineON10);
                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobePAAmacrineON08 = new PointLIFProbe("ptPAAmacrineON08.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineON08);

                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobePAAmacrineON06 = new PointLIFProbe("ptPAAmacrineON06.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineON06);

                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobePAAmacrineON04 = new PointLIFProbe("ptPAAmacrineON04.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineON04);

                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobePAAmacrineON02 = new PointLIFProbe("ptPAAmacrineON02.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineON02);
                // ---- calibration probes NSCALE IS 4

                locX = 	60/4.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobePAAmacrineONRU1 = new PointLIFProbe("ptPAAmacrineONRU1.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobePAAmacrineONRD1 = new PointLIFProbe("ptPAAmacrineONRD1.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineONRD1);

                locY =  75;      // probing
                PointLIFProbe * ptprobePAAmacrineONRU2 = new PointLIFProbe("ptPAAmacrineONRU2.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobePAAmacrineONRD2 = new PointLIFProbe("ptPAAmacrineONRD2.txt",PAAmacrineON,locX, locY, locF, "PAAmacrineON:");
                assert(ptprobePAAmacrineONRD2);

                // ---- calibration probes NSCALE IS 4 --- 0.25

                locY = 225/4.;      // probing the middle of the bar
                locF = 0;        // feature 0

                locX = ptb/4.;    // mid between the first two bars
                PointLIFProbe * ptprobeSFAmacrineB = new PointLIFProbe("ptSFAmacrineB.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrineB);

                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeSFAmacrine10 = new PointLIFProbe("ptSFAmacrine10.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrine10);
                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeSFAmacrine08 = new PointLIFProbe("ptSFAmacrine08.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrine08);

                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeSFAmacrine06 = new PointLIFProbe("ptSFAmacrine06.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrine06);
                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeSFAmacrine04 = new PointLIFProbe("ptSFAmacrine04.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrine04);

                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                PointLIFProbe * ptprobeSFAmacrine02 = new PointLIFProbe("ptSFAmacrine02.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrine02);
                // ---- calibration probes NSCALE IS 4

                locX = 	60/4.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeSFAmacrineRU1 = new PointLIFProbe("ptSFAmacrineRU1.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrineRU1);
                locY =  44;      // probing
                PointLIFProbe * ptprobeSFAmacrineRD1 = new PointLIFProbe("ptSFAmacrineRD1.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrineRD1);

                locY =  75;      // probing
                PointLIFProbe * ptprobeSFAmacrineRU2 = new PointLIFProbe("ptSFAmacrineRU2.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrineRU2);
                locY =  104;      // probing
                PointLIFProbe * ptprobeSFAmacrineRD2 = new PointLIFProbe("ptSFAmacrineRD2.txt",SFAmacrine,locX, locY, locF, "SFAmacrine:");
                assert(ptprobeSFAmacrineRD2);




                // ---- calibration probes NSCALE IS 2

                locX = 	60/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeGanglionONRU1 = new PointLIFProbe("ptGanglionONRU1.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobeGanglionONRD1 = new PointLIFProbe("ptGanglionONRD1.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRD1);


                locY =  75;      // probing
                PointLIFProbe * ptprobeGanglionONRU2 = new PointLIFProbe("ptGanglionONRU2.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobeGanglionONRD2 = new PointLIFProbe("ptGanglionONRD2.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRD2);


                locX = 	58/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeGanglionONRU1l = new PointLIFProbe("ptGanglionONRU1l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRU1l);

                locY =  44;      // probing
                PointLIFProbe * ptprobeGanglionONRD1l = new PointLIFProbe("ptGanglionONRD1l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRD1l);


                locY =  75;      // probing
                PointLIFProbe * ptprobeGanglionONRU2l = new PointLIFProbe("ptGanglionONRU2l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRU2l);

                locY =  104;      // probing
                PointLIFProbe * ptprobeGanglionONRD2l = new PointLIFProbe("ptGanglionONRD2l.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRD2l);



                locX = 	62/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeGanglionONRU1r = new PointLIFProbe("ptGanglionONRU1r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRU1r);

                locY =  44;      // probing
                PointLIFProbe * ptprobeGanglionONRD1r = new PointLIFProbe("ptGanglionONRD1r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRD1r);


                locY =  75;      // probing
                PointLIFProbe * ptprobeGanglionONRU2r = new PointLIFProbe("ptGanglionONRU2r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRU2r);

                locY =  104;      // probing
                PointLIFProbe * ptprobeGanglionONRD2r = new PointLIFProbe("ptGanglionONRD2r.txt",GanglionON,locX, locY, locF, "GanglionON:");
                assert(ptprobeGanglionONRD2r);


                //------- SynchronicityON layer
                locX = 	60/2.;
                locY =  15.;      // probing
                PointLIFProbe * ptprobeSynchronicityONRU1 = new PointLIFProbe("ptSynchronicityONRU1.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                assert(ptprobeSynchronicityONRU1);

                locY =  44;      // probing
                PointLIFProbe * ptprobeSynchronicityONRD1 = new PointLIFProbe("ptSynchronicityONRD1.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                assert(ptprobeSynchronicityONRD1);


                locY =  75;      // probing
                PointLIFProbe * ptprobeSynchronicityONRU2 = new PointLIFProbe("ptSynchronicityONRU2.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                assert(ptprobeSynchronicityONRU2);

                locY =  104;      // probing
                PointLIFProbe * ptprobeSynchronicityONRD2 = new PointLIFProbe("ptSynchronicityONRD2.txt",SynchronicityON,locX, locY, locF, "SynchronicityON:");
                assert(ptprobeSynchronicityONRD2);

                // ---- calibration probes

                       locY = 225;      // probing the bottom of the bar
                       locF = 0;        // feature 0

                       locX = ptb;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFFB = new PointLIFProbe("ptBipolarOFFB.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       assert(ptprobeBipolarOFFB);

                       locX = pt10;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF10 = new PointLIFProbe("ptBipolarOFF10.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       assert(ptprobeBipolarOFF10);

                       locX = pt08;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF08 = new PointLIFProbe("ptBipolarOFF08.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       assert(ptprobeBipolarOFF08);

                       locX = pt06;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF06 = new PointLIFProbe("ptBipolarOFF06.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       assert(ptprobeBipolarOFF06);

                       locX = pt04;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF04 = new PointLIFProbe("ptBipolarOFF04.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       assert(ptprobeBipolarOFF04);

                       locX = pt02;    // 3 + 12.8 /2 + n * 25.6
                       PointLIFProbe * ptprobeBipolarOFF02 = new PointLIFProbe("ptBipolarOFF02.txt",BipolarOFF,locX, locY, locF, "BipolarOFF:");
                       assert(ptprobeBipolarOFF02);

                     //

                       // ---- calibration probes NSCALE IS 2
                        // ---- calibration probes NSCALE IS 2

                             locY = 225/2.;      // probing the middle of the bar
                             locF = 0;        // feature 0

                             locX = ptb/2.;    // mid between the first two bars
                             PointLIFProbe * ptprobeGanglionOFFB = new PointLIFProbe("ptGanglionOFFB.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             assert(ptprobeGanglionOFFB);

                             locX = pt10/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF10 = new PointLIFProbe("ptGanglionOFF10.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             assert(ptprobeGanglionOFF10);

                             locX = pt08/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF08 = new PointLIFProbe("ptGanglionOFF08.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             assert(ptprobeGanglionOFF08);

                             locX = pt06/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF06 = new PointLIFProbe("ptGanglionOFF06.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             assert(ptprobeGanglionOFF06);

                             locX = pt04/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF04 = new PointLIFProbe("ptGanglionOFF04.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             assert(ptprobeGanglionOFF04);

                             locX = pt02/2.;    // 3 + 12.8 /2 + n * 25.6
                             PointLIFProbe * ptprobeGanglionOFF02 = new PointLIFProbe("ptGanglionOFF02.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                             assert(ptprobeGanglionOFF02);

                           //

                             // ---- calibration probes NSCALE IS 4 --- 0.25

                                locY = 225/4.;      // probing the middle of the bar
                                locF = 0;        // feature 0

                                locX = ptb/4.;    // mid between the first two bars
                                PointLIFProbe * ptprobeWFAmacrineOFFB = new PointLIFProbe("ptWFAmacrineOFFB.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFFB);

                                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeWFAmacrineOFF10 = new PointLIFProbe("ptWFAmacrineOFF10.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFF10);

                                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeWFAmacrineOFF08 = new PointLIFProbe("ptWFAmacrineOFF08.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFF08);

                                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeWFAmacrineOFF06 = new PointLIFProbe("ptWFAmacrineOFF06.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFF06);

                                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeWFAmacrineOFF04 = new PointLIFProbe("ptWFAmacrineOFF04.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFF04);

                                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobeWFAmacrineOFF02 = new PointLIFProbe("ptWFAmacrineOFF02.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFF02);

                                // ---- calibration probes NSCALE IS 4

                                locX = 	60/4.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeWFAmacrineOFFRU1 = new PointLIFProbe("ptWFAmacrineOFFRU1.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeWFAmacrineOFFRD1 = new PointLIFProbe("ptWFAmacrineOFFRD1.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeWFAmacrineOFFRU2 = new PointLIFProbe("ptWFAmacrineOFFRU2.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeWFAmacrineOFFRD2 = new PointLIFProbe("ptWFAmacrineOFFRD2.txt",WFAmacrineOFF,locX, locY, locF, "WFAmacrineOFF:");
                                assert(ptprobeWFAmacrineOFFRD2);



                                // ---- calibration probes NSCALE IS 4 --- 0.25

                                locY = 225/4.;      // probing the middle of the bar
                                locF = 0;        // feature 0

                                locX = ptb/4.;    // mid between the first two bars
                                PointLIFProbe * ptprobePAAmacrineOFFB = new PointLIFProbe("ptPAAmacrineOFFB.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFFB);
                                locX = pt10/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobePAAmacrineOFF10 = new PointLIFProbe("ptPAAmacrineOFF10.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFF10);

                                locX = pt08/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobePAAmacrineOFF08 = new PointLIFProbe("ptPAAmacrineOFF08.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFF08);

                                locX = pt06/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobePAAmacrineOFF06 = new PointLIFProbe("ptPAAmacrineOFF06.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFF06);

                                locX = pt04/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobePAAmacrineOFF04 = new PointLIFProbe("ptPAAmacrineOFF04.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFF04);

                                locX = pt02/4.;    // 3 + 12.8 /2 + n * 25.6
                                PointLIFProbe * ptprobePAAmacrineOFF02 = new PointLIFProbe("ptPAAmacrineOFF02.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFF02);

                                // ---- calibration probes NSCALE IS 4

                                locX = 	60/4.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobePAAmacrineOFFRU1 = new PointLIFProbe("ptPAAmacrineOFFRU1.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobePAAmacrineOFFRD1 = new PointLIFProbe("ptPAAmacrineOFFRD1.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobePAAmacrineOFFRU2 = new PointLIFProbe("ptPAAmacrineOFFRU2.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobePAAmacrineOFFRD2 = new PointLIFProbe("ptPAAmacrineOFFRD2.txt",PAAmacrineOFF,locX, locY, locF, "PAAmacrineOFF:");
                                assert(ptprobePAAmacrineOFFRD2);


                                // ---- calibration probes NSCALE IS 2

                                locX = 	60/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU1 = new PointLIFProbe("ptGanglionOFFRU1.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD1 = new PointLIFProbe("ptGanglionOFFRD1.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU2 = new PointLIFProbe("ptGanglionOFFRU2.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD2 = new PointLIFProbe("ptGanglionOFFRD2.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRD2);


                                locX = 	58/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU1l = new PointLIFProbe("ptGanglionOFFRU1l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRU1l);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD1l = new PointLIFProbe("ptGanglionOFFRD1l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRD1l);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU2l = new PointLIFProbe("ptGanglionOFFRU2l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                               assert(ptprobeGanglionOFFRU2l);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD2l = new PointLIFProbe("ptGanglionOFFRD2l.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRD2l);



                                locX = 	62/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU1r = new PointLIFProbe("ptGanglionOFFRU1r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRU1r);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD1r = new PointLIFProbe("ptGanglionOFFRD1r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRD1r);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRU2r = new PointLIFProbe("ptGanglionOFFRU2r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRU2r);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeGanglionOFFRD2r = new PointLIFProbe("ptGanglionOFFRD2r.txt",GanglionOFF,locX, locY, locF, "GanglionOFF:");
                                assert(ptprobeGanglionOFFRD2r);


                                //------- SynchronicityOFF layer
                                locX = 	60/2.;
                                locY =  15.;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRU1 = new PointLIFProbe("ptSynchronicityOFFRU1.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                assert(ptprobeSynchronicityOFFRU1);

                                locY =  44;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRD1 = new PointLIFProbe("ptSynchronicityOFFRD1.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                assert(ptprobeSynchronicityOFFRD1);


                                locY =  75;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRU2 = new PointLIFProbe("ptSynchronicityOFFRU2.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                assert(ptprobeSynchronicityOFFRU2);

                                locY =  104;      // probing
                                PointLIFProbe * ptprobeSynchronicityOFFRD2 = new PointLIFProbe("ptSynchronicityOFFRD2.txt",SynchronicityOFF,locX, locY, locF, "SynchronicityOFF:");
                                assert(ptprobeSynchronicityOFFRD2);



    return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {

	PVParams * params = hc->parameters();

    int status = 0;

    fprintf(stdout,"================================================\n");

    fprintf(stdout,"This run used the following strength parameters:\n");

    float strength = 0;
    float sigma = 0;

    strength = params->value("Image to Cone", "strength", strength);
    sigma = params->value("Image to Cone", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","Image to Cone" ,strength,sigma);

    strength = params->value("ConeSigmoidON to Horizontal", "strength", strength);
    sigma    = params->value("ConeSigmoidON to Horizontal", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "ConeSigmoidON to Horizontal",strength,sigma);

    strength = params->value("HoriGap to Horizontal", "strength", strength);
    sigma    = params->value("HoriGap to Horizontal", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","HoriGap to Horizontal" ,strength,sigma);

    strength = params->value("HoriSigmoid to Cone", "strength", strength);
    sigma = params->value("HoriSigmoid to Cone", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","HoriSigmoid to Cone" ,strength,sigma);

    strength = params->value("ConeSigmoidON to BipolarON", "strength", strength);
    sigma    = params->value("ConeSigmoidON to BipolarON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","ConeSigmoidON to BipolarON" ,strength,sigma);

    strength = params->value("BipolarSigmoidON to GanglionON", "strength", strength);
    sigma    = params->value("BipolarSigmoidON to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidON to GanglionON" ,strength,sigma);

    strength = params->value("BipolarSigmoidON to WFAmacrineON", "strength", strength);
    sigma    = params->value("BipolarSigmoidON to WFAmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "BipolarSigmoidON to WFAmacrineON",strength,sigma);

    strength = params->value("WFAmacrineSigmoidON to BipolarON", "strength", strength);
       sigma    = params->value("WFAmacrineSigmoidON to BipolarON", "sigma", sigma);
       fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineSigmoidON to BipolarON",strength,sigma);

    strength = params->value("WFAmacrineGapON to WFAmacrineON", "strength", strength);
       sigma    = params->value("WFAmacrineGapON to WFAmacrineON", "sigma", sigma);
       fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineGapON to WFAmacrineON",strength,sigma);

    strength = params->value("GangliGapON to PAAmacrineON", "strength", strength);
    sigma    = params->value("GangliGapON to PAAmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","GangliGapON to PAAmacrineON" ,strength,sigma);

    strength = params->value("PAAmaGapON to GanglionON", "strength", strength);
    sigma    = params->value("PAAmaGapON to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapON to GanglionON" ,strength,sigma);

    strength = params->value("PAAmaGapON to PAAmacrineON", "strength", strength);
    sigma    = params->value("PAAmaGapON to PAAmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapON to PAAmacrineON" ,strength,sigma);

    strength = params->value("BipolarSigmoidON to SFAmacrine", "strength", strength);
    sigma    = params->value("BipolarSigmoidON to SFAmacrine", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidON to SFAmacrine" ,strength,sigma);

    strength = params->value("WFAmacrineSigmoidON to SFAmacrine", "strength", strength);
    sigma    = params->value("WFAmacrineSigmoidON to SFAmacrine", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","WFAmacrineSigmoidON to SFAmacrine" ,strength,sigma);

    strength = params->value("PAAmacrineON to WFAmacrineON", "strength", strength);
    sigma    = params->value("PAAmacrineON to WFAmacrineON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineON to WFAmacrineON" ,strength,sigma);

    strength = params->value("PAAmacrineON to GanglionON", "strength", strength);
    sigma    = params->value("PAAmacrineON to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineON to GanglionON" ,strength,sigma);

    strength = params->value("PAAmacrineON to PAAmacrineON", "strength", strength);
        sigma    = params->value("PAAmacrineON to PAAmacrineON", "sigma", sigma);
        fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","self shutdown PAAmacrineON to PAAmacrineON" ,strength,sigma);


    strength = params->value("SFAmacrineSigmoid to GanglionON", "strength", strength);
    sigma    = params->value("SFAmacrineSigmoid to GanglionON", "sigma", sigma);
    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","SFAmacrineSigmoid to GanglionON" ,strength,sigma);



    // OFF connections


    strength = params->value("ConeSigmoidOFF to BipolarOFF", "strength", strength);
     sigma    = params->value("ConeSigmoidOFF to BipolarOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","ConeSigmoidOFF to BipolarOFF" ,strength,sigma);

     strength = params->value("BipolarSigmoidOFF to GanglionOFF", "strength", strength);
     sigma    = params->value("BipolarSigmoidOFF to GanglionOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidOFF to GanglionOFF" ,strength,sigma);

     strength = params->value("BipolarSigmoidOFF to WFAmacrineOFF", "strength", strength);
     sigma    = params->value("BipolarSigmoidOFF to WFAmacrineOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "BipolarSigmoidOFF to WFAmacrineOFF",strength,sigma);

     strength = params->value("WFAmacrineSigmoidOFF to BipolarOFF", "strength", strength);
        sigma    = params->value("WFAmacrineSigmoidOFF to BipolarOFF", "sigma", sigma);
        fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineSigmoidOFF to BipolarOFF",strength,sigma);

     strength = params->value("WFAmacrineGapOFF to WFAmacrineOFF", "strength", strength);
        sigma    = params->value("WFAmacrineGapOFF to WFAmacrineOFF", "sigma", sigma);
        fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineGapOFF to WFAmacrineOFF",strength,sigma);

     strength = params->value("GangliGapOFF to PAAmacrineOFF", "strength", strength);
     sigma    = params->value("GangliGapOFF to PAAmacrineOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","GangliGapOFF to PAAmacrineOFF" ,strength,sigma);

     strength = params->value("PAAmaGapOFF to GanglionOFF", "strength", strength);
     sigma    = params->value("PAAmaGapOFF to GanglionOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapOFF to GanglionOFF" ,strength,sigma);

     strength = params->value("PAAmaGapOFF to PAAmacrineOFF", "strength", strength);
     sigma    = params->value("PAAmaGapOFF to PAAmacrineOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapOFF to PAAmacrineOFF" ,strength,sigma);

     strength = params->value("BipolarSigmoidOFF to SFAmacrine", "strength", strength);
     sigma    = params->value("BipolarSigmoidOFF to SFAmacrine", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidOFF to SFAmacrine" ,strength,sigma);

     strength = params->value("WFAmacrineSigmoidOFF to SFAmacrine", "strength", strength);
     sigma    = params->value("WFAmacrineSigmoidOFF to SFAmacrine", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","WFSigmoidAmacrineOFF to SFAmacrine" ,strength,sigma);

     strength = params->value("PAAmacrineOFF to WFAmacrineOFF", "strength", strength);
     sigma    = params->value("PAAmacrineOFF to WFAmacrineOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineOFF to WFAmacrineOFF" ,strength,sigma);

     strength = params->value("PAAmacrineOFF to GanglionOFF", "strength", strength);
     sigma    = params->value("PAAmacrineOFF to GanglionOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineOFF to GanglionOFF" ,strength,sigma);

     strength = params->value("PAAmacrineOFF to PAAmacrineOFF", "strength", strength);
         sigma    = params->value("PAAmacrineOFF to PAAmacrineOFF", "sigma", sigma);
         fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","self shutdown PAAmacrineOFF to PAAmacrineOFF" ,strength,sigma);

     strength = params->value("SFAmacrineSigmoid to GanglionOFF", "strength", strength);
     sigma    = params->value("SFAmacrineSigmoid to GanglionOFF", "sigma", sigma);
     fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","SFAmacrineSigmoid to GanglionOFF" ,strength,sigma);

    fprintf(stdout,"================================================ \a \a \a \a \a \a \a \n");
    fprintf(stderr,"================================================ \a \a \a \a \a \a \a \n");


    return status;
}

	#endif // MAIN_USES_ADDCUSTOM

