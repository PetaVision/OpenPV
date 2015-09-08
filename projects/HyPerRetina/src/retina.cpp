/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>

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

    return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {

    //PVParams * params = hc->parameters();

    int status = 0;

    //fprintf(stdout,"================================================\n");

    //fprintf(stdout,"This run used the following strength parameters:\n");

    //float strength = 0;
    //float sigma = 0;

    //HyPerConn * ImageToCone = hc->getConnFromName("Image to Cone");
    //if (ImageToCone != NULL) {
    //    strength = params->value("Image to Cone", "strength", strength);
    //    sigma = params->value("Image to Cone", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","Image to Cone" ,strength,sigma);
    //}

    //HyPerConn * ConeSigmoidONToHorizontal = hc->getConnFromName("ConeSigmoidON to Horizontal");
    //if (ConeSigmoidONToHorizontal != NULL) {
    //    strength = params->value("ConeSigmoidON to Horizontal", "strength", strength);
    //    sigma    = params->value("ConeSigmoidON to Horizontal", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "ConeSigmoidON to Horizontal",strength,sigma);
    //}

    //HyPerConn * HoriGapToHorizontal = hc->getConnFromName("HoriGap to Horizontal");
    //if (HoriGapToHorizontal != NULL) {
    //    strength = params->value("HoriGap to Horizontal", "strength", strength);
    //    sigma    = params->value("HoriGap to Horizontal", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","HoriGap to Horizontal" ,strength,sigma);
    //}

    //HyPerConn * HoriSigmoidToCone = hc->getConnFromName("HoriSigmoid to Cone");
    //if (HoriSigmoidToCone != NULL) {
    //    strength = params->value("HoriSigmoid to Cone", "strength", strength);
    //    sigma = params->value("HoriSigmoid to Cone", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","HoriSigmoid to Cone" ,strength,sigma);
    //}

    //HyPerConn * ConeSigmoidONToBipolarON = hc->getConnFromName("ConeSigmoidON to BipolarON");
    //if (ConeSigmoidONToBipolarON != NULL) {
    //    strength = params->value("ConeSigmoidON to BipolarON", "strength", strength);
    //    sigma    = params->value("ConeSigmoidON to BipolarON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","ConeSigmoidON to BipolarON" ,strength,sigma);
    //}

    //HyPerConn * BipolarSigmoidONToGanglionON = hc->getConnFromName("BipolarSigmoidON to GanglionON");
    //if (BipolarSigmoidONToGanglionON != NULL) {
    //    strength = params->value("BipolarSigmoidON to GanglionON", "strength", strength);
    //    sigma    = params->value("BipolarSigmoidON to GanglionON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidON to GanglionON" ,strength,sigma);
    //}

    //HyPerConn * BipolarSigmoidONToWFAmacrineON = hc->getConnFromName("BipolarSigmoidON to WFAmacrineON");
    //if (BipolarSigmoidONToWFAmacrineON != NULL) {
    //    strength = params->value("BipolarSigmoidON to WFAmacrineON", "strength", strength);
    //    sigma    = params->value("BipolarSigmoidON to WFAmacrineON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "BipolarSigmoidON to WFAmacrineON",strength,sigma);
    //}

    //HyPerConn * WFAmacrineSigmoidONToBipolarON = hc->getConnFromName("WFAmacrineSigmoidON to BipolarON");
    //if (WFAmacrineSigmoidONToBipolarON != NULL) {
    //    strength = params->value("WFAmacrineSigmoidON to BipolarON", "strength", strength);
    //    sigma    = params->value("WFAmacrineSigmoidON to BipolarON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineSigmoidON to BipolarON",strength,sigma);
    //}

    //HyPerConn * WFAmacrineGapONToWFAmacrineON = hc->getConnFromName("WFAmacrineGapON to WFAmacrineON");
    //if (WFAmacrineGapONToWFAmacrineON != NULL) {
    //    strength = params->value("WFAmacrineGapON to WFAmacrineON", "strength", strength);
    //    sigma    = params->value("WFAmacrineGapON to WFAmacrineON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineGapON to WFAmacrineON",strength,sigma);
    //}

    //HyPerConn * GangliGapONToPAAmacrineON = hc->getConnFromName("GangliGapON to PAAmacrineON");
    //if (GangliGapONToPAAmacrineON != NULL) {
    //    strength = params->value("GangliGapON to PAAmacrineON", "strength", strength);
    //    sigma    = params->value("GangliGapON to PAAmacrineON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","GangliGapON to PAAmacrineON" ,strength,sigma);
    //}

    //HyPerConn * PAAmaGapONToGanglionON = hc->getConnFromName("PAAmaGapON to GanglionON");
    //if (PAAmaGapONToGanglionON != NULL) {
    //    strength = params->value("PAAmaGapON to GanglionON", "strength", strength);
    //    sigma    = params->value("PAAmaGapON to GanglionON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapON to GanglionON" ,strength,sigma);
    //}

    //HyPerConn * PAAmaGapONToPAAmacrineON = hc->getConnFromName("PAAmaGapON to PAAmacrineON");
    //if (PAAmaGapONToPAAmacrineON != NULL) {
    //    strength = params->value("PAAmaGapON to PAAmacrineON", "strength", strength);
    //    sigma    = params->value("PAAmaGapON to PAAmacrineON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapON to PAAmacrineON" ,strength,sigma);
    //}

    //HyPerConn * BipolarSigmoidONToSFAmacrine = hc->getConnFromName("BipolarSigmoidON to SFAmacrine");
    //if (BipolarSigmoidONToSFAmacrine != NULL) {
    //    strength = params->value("BipolarSigmoidON to SFAmacrine", "strength", strength);
    //    sigma    = params->value("BipolarSigmoidON to SFAmacrine", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidON to SFAmacrine" ,strength,sigma);
    //}

    //HyPerConn * WFAmacrineSigmoidONToSFAmacrine = hc->getConnFromName("WFAmacrineSigmoidON to SFAmacrine");
    //if (WFAmacrineSigmoidONToSFAmacrine != NULL) {
    //    strength = params->value("WFAmacrineSigmoidON to SFAmacrine", "strength", strength);
    //    sigma    = params->value("WFAmacrineSigmoidON to SFAmacrine", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","WFAmacrineSigmoidON to SFAmacrine" ,strength,sigma);
    //}

    //HyPerConn * PAAmacrineONToWFAmacrineON = hc->getConnFromName("PAAmacrineON to WFAmacrineON");
    //if (PAAmacrineONToWFAmacrineON != NULL) {
    //    strength = params->value("PAAmacrineON to WFAmacrineON", "strength", strength);
    //    sigma    = params->value("PAAmacrineON to WFAmacrineON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineON to WFAmacrineON" ,strength,sigma);
    //}

    //HyPerConn * PAAmacrineONToGanglionON = hc->getConnFromName("PAAmacrineON to GanglionON");
    //if (PAAmacrineONToGanglionON != NULL) {
    //    strength = params->value("PAAmacrineON to GanglionON", "strength", strength);
    //    sigma    = params->value("PAAmacrineON to GanglionON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineON to GanglionON" ,strength,sigma);
    //}

    //HyPerConn * PAAmacrineONToPAAmacrineON = hc->getConnFromName("PAAmacrineON to PAAmacrineON");
    //if (PAAmacrineONToPAAmacrineON != NULL) {
    //    strength = params->value("PAAmacrineON to PAAmacrineON", "strength", strength);
    //    sigma    = params->value("PAAmacrineON to PAAmacrineON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","self shutdown PAAmacrineON to PAAmacrineON" ,strength,sigma);
    //}

    //HyPerConn * SFAmacrineSigmoidToGanglionON = hc->getConnFromName("SFAmacrineSigmoid to GanglionON");
    //if (SFAmacrineSigmoidToGanglionON != NULL) {
    //    strength = params->value("SFAmacrineSigmoid to GanglionON", "strength", strength);
    //    sigma    = params->value("SFAmacrineSigmoid to GanglionON", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","SFAmacrineSigmoid to GanglionON" ,strength,sigma);
    //}


    //// OFF connections

    //HyPerConn * ConeSigmoidOFFToBipolarOFF = hc->getConnFromName("ConeSigmoidOFF to BipolarOFF");
    //if (ConeSigmoidOFFToBipolarOFF != NULL) {
    //    strength = params->value("ConeSigmoidOFF to BipolarOFF", "strength", strength);
    //    sigma    = params->value("ConeSigmoidOFF to BipolarOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","ConeSigmoidOFF to BipolarOFF" ,strength,sigma);
    //}

    //HyPerConn * BipolarSigmoidOFFToGanglionOFF = hc->getConnFromName("BipolarSigmoidOFF to GanglionOFF");
    //if (BipolarSigmoidOFFToGanglionOFF != NULL) {
    //    strength = params->value("BipolarSigmoidOFF to GanglionOFF", "strength", strength);
    //    sigma    = params->value("BipolarSigmoidOFF to GanglionOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidOFF to GanglionOFF" ,strength,sigma);
    //}

    //HyPerConn * BipolarSigmoidOFFToWFAmacrineOFF = hc->getConnFromName("BipolarSigmoidOFF to WFAmacrineOFF");
    //if (BipolarSigmoidOFFToWFAmacrineOFF != NULL) {
    //    strength = params->value("BipolarSigmoidOFF to WFAmacrineOFF", "strength", strength);
    //    sigma    = params->value("BipolarSigmoidOFF to WFAmacrineOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "BipolarSigmoidOFF to WFAmacrineOFF",strength,sigma);
    //}

    //HyPerConn * WFAmacrineSigmoidOFFToBipolarOFF = hc->getConnFromName("WFAmacrineSigmoidOFF to BipolarOFF");
    //if (WFAmacrineSigmoidOFFToBipolarOFF != NULL) {
    //    strength = params->value("WFAmacrineSigmoidOFF to BipolarOFF", "strength", strength);
    //    sigma    = params->value("WFAmacrineSigmoidOFF to BipolarOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineSigmoidOFF to BipolarOFF",strength,sigma);
    //}

    //HyPerConn * WFAmacrineGapOFFToWFAmacrineOFF = hc->getConnFromName("WFAmacrineGapOFF to WFAmacrineOFF");
    //if (WFAmacrineGapOFFToWFAmacrineOFF != NULL) {
    //    strength = params->value("WFAmacrineGapOFF to WFAmacrineOFF", "strength", strength);
    //    sigma    = params->value("WFAmacrineGapOFF to WFAmacrineOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n", "WFAmacrineGapOFF to WFAmacrineOFF",strength,sigma);
    //}

    //HyPerConn * GangliGapOFFToPAAmacrineOFF = hc->getConnFromName("GangliGapOFF to PAAmacrineOFF");
    //if (GangliGapOFFToPAAmacrineOFF != NULL) {
    //    strength = params->value("GangliGapOFF to PAAmacrineOFF", "strength", strength);
    //    sigma    = params->value("GangliGapOFF to PAAmacrineOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","GangliGapOFF to PAAmacrineOFF" ,strength,sigma);
    //}

    //HyPerConn * PAAmaGapOFFToGanglionOFF = hc->getConnFromName("PAAmaGapOFF to GanglionOFF");
    //if (PAAmaGapOFFToGanglionOFF != NULL) {
    //    strength = params->value("PAAmaGapOFF to GanglionOFF", "strength", strength);
    //    sigma    = params->value("PAAmaGapOFF to GanglionOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapOFF to GanglionOFF" ,strength,sigma);
    //}

    //HyPerConn * PAAmaGapOFFToPAAmacrineOFF = hc->getConnFromName("PAAmaGapOFF to PAAmacrineOFF");
    //if (PAAmaGapOFFToPAAmacrineOFF != NULL) {
    //    strength = params->value("PAAmaGapOFF to PAAmacrineOFF", "strength", strength);
    //    sigma    = params->value("PAAmaGapOFF to PAAmacrineOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmaGapOFF to PAAmacrineOFF" ,strength,sigma);
    //}

    //HyPerConn * BipolarSigmoidOFFToSFAmacrine = hc->getConnFromName("BipolarSigmoidOFF to SFAmacrine");
    //if (BipolarSigmoidOFFToSFAmacrine != NULL) {
    //    strength = params->value("BipolarSigmoidOFF to SFAmacrine", "strength", strength);
    //    sigma    = params->value("BipolarSigmoidOFF to SFAmacrine", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","BipolarSigmoidOFF to SFAmacrine" ,strength,sigma);
    //}

    //HyPerConn * WFAmacrineSigmoidOFFToSFAmacrine = hc->getConnFromName("WFAmacrineSigmoidOFF to SFAmacrine");
    //if (WFAmacrineSigmoidOFFToSFAmacrine != NULL) {
    //    strength = params->value("WFAmacrineSigmoidOFF to SFAmacrine", "strength", strength);
    //    sigma    = params->value("WFAmacrineSigmoidOFF to SFAmacrine", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","WFSigmoidAmacrineOFF to SFAmacrine" ,strength,sigma);
    //}

    //HyPerConn * PAAmacrineOFFToWFAmacrineOFF = hc->getConnFromName("PAAmacrineOFF to WFAmacrineOFF");
    //if (PAAmacrineOFFToWFAmacrineOFF != NULL) {
    //    strength = params->value("PAAmacrineOFF to WFAmacrineOFF", "strength", strength);
    //    sigma    = params->value("PAAmacrineOFF to WFAmacrineOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineOFF to WFAmacrineOFF" ,strength,sigma);
    //}

    //HyPerConn * PAAmacrineOFFToGanglionOFF = hc->getConnFromName("PAAmacrineOFF to GanglionOFF");
    //if (PAAmacrineOFFToGanglionOFF != NULL) {
    //    strength = params->value("PAAmacrineOFF to GanglionOFF", "strength", strength);
    //    sigma    = params->value("PAAmacrineOFF to GanglionOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","PAAmacrineOFF to GanglionOFF" ,strength,sigma);
    //}

    //HyPerConn * PAAmacrineOFFToPAAmacrineOFF = hc->getConnFromName("PAAmacrineOFF to PAAmacrineOFF");
    //if (PAAmacrineOFFToPAAmacrineOFF != NULL) {
    //    strength = params->value("PAAmacrineOFF to PAAmacrineOFF", "strength", strength);
    //    sigma    = params->value("PAAmacrineOFF to PAAmacrineOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","self shutdown PAAmacrineOFF to PAAmacrineOFF" ,strength,sigma);
    //}

    //HyPerConn * SFAmacrineSigmoidToGanglionOFF = hc->getConnFromName("SFAmacrineSigmoid to GanglionOFF");
    //if (SFAmacrineSigmoidToGanglionOFF != NULL) {
    //    strength = params->value("SFAmacrineSigmoid to GanglionOFF", "strength", strength);
    //    sigma    = params->value("SFAmacrineSigmoid to GanglionOFF", "sigma", sigma);
    //    fprintf(stdout,"%32s has a strength of  %f and a sigma of %f \n","SFAmacrineSigmoid to GanglionOFF" ,strength,sigma);
    //}

    //fprintf(stdout,"================================================ \a \a \a \a \a \a \a \n");
    //fprintf(stderr,"================================================ \a \a \a \a \a \a \a \n");


    return status;
}

#endif // MAIN_USES_ADDCUSTOM

