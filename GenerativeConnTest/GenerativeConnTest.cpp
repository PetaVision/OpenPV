/*
 * GenerativeConnTest.cpp
 *
 * Tests GenerativeConn.  If called with no parameter file specified, will run
 * with params file input/GenerativeConnTest-MirrorBCOff.params, and then
 * with params file input/GenerativeConnTest-MirrorBCOn.params.
 *
 * If called with a parameter file specified, it will run with that parameter.
 *
 * For connections with presynaptic mirrorBCflag set to true, the program
 * expects connections to all be ones.  This will be the case if
 * the presynaptic and postsynaptic neurons are all ones for a single time step,
 * and zeros the rest of the time, as specified in the image*sequence.txt files.
 *
 * For connections with presynaptic mirrorBCflag false, the program expects
 * connections to be one at the center and to fall off as one goes away from
 * the center.  For example, for a one-to-one connection with patch size 5x5
 * and pre-synaptic layer size 16x16, the weights will be
 *         /  196 210 224 210 196  \
 *    1    |  210 225 240 225 210  |
 *   --- * |  224 240 256 240 224  |
 *   256   |  210 225 240 225 210  |
 *         \  196 210 224 210 196  /
 *
 * This is the matrix product of (14 15 16 15 14)'/16 and (14 15 16 15 14)/16.
 * (the apostrophe indicates the transpose).
 *
 * For a two-to-one connection with patch size 5x5 and pre-synaptic layer size 16x16,
 * the weights will be the matrix product of (12 14 16 14 12)'/16 and (12 14 16 14 12)/16.
 * For many-to-one connections, the change in entries in the vectors is "many"/16 instead of 2/16.
 *
 * For a one-to-two connection with patch size 5x5 and pre-synaptic layer size 16x16,
 * the weight will be the matrix product of (14 14 15 15 16 16 15 15 14 14)'/16 and the same
 * vector as a row vector.  For one-to-many connections, the duplication in the entries takes
 * place "many" times instead of twice.
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "../PetaVision/src/io/io.c"
#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif // PV_USE_MPI

int runGenerativeConnTest(int argc, char * argv[]);
int dumpweights(HyPerCol * hc, int argc, char * argv[]);
int dumponeweight(GenerativeConn * conn);

int main(int argc, char * argv[]) {
   int status;
#ifdef PV_USE_MPI
   int mpi_initialized_on_entry;
   MPI_Initialized(&mpi_initialized_on_entry);
   if( !mpi_initialized_on_entry ) MPI_Init(&argc, &argv);
#endif PV_USE_MPI
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL);
   int num_cl_args;
   char ** cl_args;
   if( paramfileabsent ) {
      num_cl_args = argc + 2;
      cl_args = (char **) malloc(num_cl_args*sizeof(char *));
      cl_args[0] = argv[0];
      cl_args[1] = strdup("-p");
      cl_args[2] = strdup("input/GenerativeConnTest-MirrorBCOff.params");
      for( int k=1; k<argc; k++) {
         cl_args[k+2] = strdup(argv[k]);
      }
      status = runGenerativeConnTest(num_cl_args, cl_args);
      if( status == PV_SUCCESS ) {
         free(cl_args[2]);
         cl_args[2] = strdup("input/GenerativeConnTest-MirrorBCOn.params");
         status = runGenerativeConnTest(num_cl_args, cl_args);
      }
      free(cl_args[1]); cl_args[1] = NULL;
      free(cl_args[2]); cl_args[2] = NULL;
      free(cl_args); cl_args = NULL;
   }
   else {
      status = runGenerativeConnTest(argc, argv);
   }
   MPI_Finalize();
   return status;
}

int runGenerativeConnTest(int argc, char * argv[]) {
   int status = buildandrun(argc, argv, NULL, &dumpweights);
   return status;
}

int dumpweights(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   bool existsgenconn = false;
   for( int k=0; k < hc->numberOfConnections(); k++ ) {
      GenerativeConn * conn = dynamic_cast<GenerativeConn *>(hc->getConnection(k));
      if( conn != NULL ) {
         existsgenconn = true;
         int status1 = dumponeweight(conn);
         if( status == PV_SUCCESS ) status = status1;
      }
   }
   if( existsgenconn && status != PV_SUCCESS ) {
      for( int k=0; k<72; k++ ) { fprintf(stdout, "="); } fprintf(stdout,"\n");
   }
   int rank = hc->icCommunicator()->commRank();
   char * paramsfilename;
   pv_getopt_str(argc, argv, "-p", &paramsfilename);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "Rank %d: %s failed with return code.\n", rank, paramsfilename);
   }
   else {
      printf("Rank %d: %s succeeded.\n", rank, paramsfilename);
   }
   return status;
}

int dumponeweight(GenerativeConn * conn) {
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

   for( int p=0; p<conn->numDataPatches(); p++ ) {
      pvdata_t * wgtData = conn->getKernelPatch(0,p)->data;
      for( int f=0; f<nfp; f++ ) {
         for( int x=0; x<nxp; x++ ) {
            int xoffset = abs((int) floor((x-xcenter)*xFalloff));
            for( int y=0; y<nyp; y++ ) {
            int yoffset = abs((int) floor((y-ycenter)*yFalloff));
               int idx = kIndex(x, y, f, nxp, nyp, nfp);
               pvdata_t wgt = wgtData[idx];
               pvdata_t correct = usingMirrorBCs ? 1 : (nxpre-xoffset)*(nypre-yoffset)/((pvdata_t) (nxpre*nypre));
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
      fprintf(stdout, "Rank %d, GenerativeConn \"%s\": Weights are correct.\n", rank, conn->getName());
   }
   return status;
}
