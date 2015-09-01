/*
 * KernelActivationTest.cpp
 *
 * Tests kernel activations.  If called with no parameter file specified, will run
 * with params file input/KernelActivationTest-MirrorBCOff.params, and then
 * with params file input/KernelActivationTest-MirrorBCOn.params.
 *
 * If called with a parameter file specified, it will run with that parameter.
 *
 * For connections testing full data, since the pre/post values are .5,
 * weight update should make all the weights .5^2 with a pixel roundoff error
 *
 * For connection testing masked data in both pre and post layers, since
 * all the pre/post is constant, and the kernel normalization takes into account how many
 * pre/post pairs are actually being calculated, all the weights should be identical to
 * the full activations.
 */

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include <arch/mpi/mpi.h>

int runKernelActivationTest(int argc, char * argv[], PV_Init* initObj);
int dumpweights(HyPerCol * hc, int argc, char * argv[]);
int dumponeweight(HyPerConn * conn);

int main(int argc, char * argv[]) {
   int status;
   PV_Init * initObj = new PV_Init(&argc, &argv);
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL/*sVal*/, NULL/*paramusage*/);
   int num_cl_args;
   char ** cl_args;
   if( paramfileabsent ) {
      num_cl_args = argc + 2;
      cl_args = (char **) malloc((num_cl_args+1)*sizeof(char *));
      cl_args[0] = argv[0];
      cl_args[1] = strdup("-p");
      cl_args[2] = strdup("input/KernelActivationTest-fullData.params");
      for( int k=1; k<argc; k++) {
         cl_args[k+2] = strdup(argv[k]);
      }
      cl_args[num_cl_args] = NULL;
      status = runKernelActivationTest(num_cl_args, cl_args, initObj);
      if( status == PV_SUCCESS ) {
         free(cl_args[2]);
         cl_args[2] = strdup("input/KernelActivationTest-maskData.params");
         status = runKernelActivationTest(num_cl_args, cl_args, initObj);
      }
      free(cl_args[1]); cl_args[1] = NULL;
      free(cl_args[2]); cl_args[2] = NULL;
      free(cl_args); cl_args = NULL;
   }
   else {
      status = runKernelActivationTest(argc, argv, initObj);
   }
   delete initObj;
   return status;
}

int runKernelActivationTest(int argc, char * argv[], PV_Init* initObj) {
   int status = rebuildandrun(argc, argv, initObj, NULL, &dumpweights);
   return status;
}

int dumpweights(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   bool existsgenconn = false;
   for( int k=0; k < hc->numberOfConnections(); k++ ) {
      HyPerConn * conn = dynamic_cast<HyPerConn *>(hc->getConnection(k));
      //Only test plastic conns
      if( conn != NULL) {
         if(conn->getPlasticityFlag()){
            existsgenconn = true;
            int status1 = dumponeweight(conn);
            if( status == PV_SUCCESS ) status = status1;
         }
      }
   }
   if( existsgenconn && status != PV_SUCCESS ) {
      for( int k=0; k<72; k++ ) { fprintf(stdout, "="); } fprintf(stdout,"\n");
   }
   int rank = hc->icCommunicator()->commRank();
   char * paramsfilename;
   pv_getopt_str(argc, argv, "-p", &paramsfilename, NULL/*paramusage*/);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "Rank %d: %s failed with return code %d.\n", rank, paramsfilename, status);
   }
   else {
      printf("Rank %d: %s succeeded.\n", rank, paramsfilename);
   }
   free(paramsfilename);
   return status;
}

int dumponeweight(HyPerConn * conn) {
   int status = PV_SUCCESS;
   bool errorfound = false;
   int rank = conn->getParent()->icCommunicator()->commRank();
   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int xcenter = (nxp-1)/2;
   int ycenter = (nyp-1)/2;
   int nxpre = conn->preSynapticLayer()->getLayerLoc()->nxGlobal;
   int nypre = conn->preSynapticLayer()->getLayerLoc()->nyGlobal;
   bool usingMirrorBCs = conn->preSynapticLayer()->useMirrorBCs();

   // If xScaleDiff > 0, it's a many-to-one connection.
   int xScaleDiff = conn->postSynapticLayer()->getXScale() - conn->preSynapticLayer()->getXScale();
   float xFalloff = powf(2,xScaleDiff);
   int yScaleDiff = conn->postSynapticLayer()->getYScale() - conn->preSynapticLayer()->getYScale();
   float yFalloff = powf(2,yScaleDiff);

   for( int p=0; p<conn->getNumDataPatches(); p++ ) {
      pvwdata_t * wgtData = conn->get_wDataHead(0,p); // conn->getKernelPatch(0,p)->data;
      for( int f=0; f<nfp; f++ ) {
         for( int x=0; x<nxp; x++ ) {
            int xoffset = abs((int) floor((x-xcenter)*xFalloff));
            for( int y=0; y<nyp; y++ ) {
            int yoffset = abs((int) floor((y-ycenter)*yFalloff));
               int idx = kIndex(x, y, f, nxp, nyp, nfp);
               //TODO-CER-2014.4.4 - weight conversion
               pvdata_t wgt = wgtData[idx];
               //pvdata_t correct = usingMirrorBCs ? 1 : (nxpre-xoffset)*(nypre-yoffset)/((pvdata_t) (nxpre*nypre));
               //New normalization takes into account if pre is not active
               //The pixel value from the input is actually 127, where we divide it by 255.
               //Not exaclty .5, a little less
               //Squared because both pre and post is grabbing it's activity from the image
               pvdata_t correct = usingMirrorBCs ? pow(float(127)/float(255),2) : (float(127)/float(255)) * .5;
               if( fabs(wgt-correct)>1.0e-5 ) {
                  if( errorfound == false ) {
                      errorfound = true;
                      for( int k=0; k<72; k++ ) { fprintf(stdout, "="); } fprintf(stdout,"\n");
                      printf("Rank %d, Connection \"%s\":\n",rank, conn->getName());
                  }
                  fprintf(stdout, "Rank %d, Patch %d, x=%d, y=%d, f=%d: weight=%f, correct=%f, off by a factor of %f\n", rank, p, x, y, f, wgt, correct, wgt/correct);
                  status = PV_FAILURE;
               }
            }
         }
      }
   }
   if( status == PV_SUCCESS ) {
      fprintf(stdout, "Rank %d, connection \"%s\": Weights are correct.\n", rank, conn->getName());
   }
   return status;
}
