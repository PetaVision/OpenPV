/*
 * LayerRestartTest.cpp
 *
 * Tests the restart flag for HyPerLayer.
 * Run without arguments, it will run the following params files in sequence:
 * input/LayerRestartTest-Write.params
 * input/LayerRestartTest-Check.params.
 * input/LayerRestartTest-Read.params.
 *
 * LayerRestartTest-Write loads an image from input/F_N160050.jpg into
 * the layer "Copy" and random noise into the layer "Comparison", and
 * writes Copy_{A,V}.pvp and Comparison_{A,V}.pvp files.
 *
 * LayerRestartTest-Check restarts the Comparison layer.
 * the main() function calls buildandrun with this params file,
 * and uses the customexit hook to verify that Comparison is
 * not all zeros.
 *
 * LayerRestartTest-Read restarts the Copy and Comparison layers,
 * and loads input/F_N160050.jpg into an Image layer.
 * The comparison layer takes the difference of Image and Copy,
 * which should be all zeros.  The customexit hook now verifies
 * that Comparison is all zeros.
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "../PetaVision/src/io/io.c"
#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../PetaVision/src/include/mpi_stubs.h"
#endif // PV_USE_MPI

int checkComparisonZero(HyPerCol * hc, int argc, char * argv[]);
int checkComparisonNonzero(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   int status;
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL);
   if( !paramfileabsent ) {
      fprintf(stderr, "%s runs a number of params files in sequence.  Do not include a '-p' option when running this program.\n", argv[0]);
      exit(EXIT_FAILURE);
   }
#ifdef PV_USE_MPI
   int mpi_initialized_on_entry;
   MPI_Initialized(&mpi_initialized_on_entry);
   if( !mpi_initialized_on_entry ) MPI_Init(&argc, &argv);
#endif // PV_USE_MPI

   int num_cl_args;
   char ** cl_args;
   num_cl_args = argc + 2;
   cl_args = (char **) malloc(num_cl_args*sizeof(char *));
   cl_args[0] = argv[0];
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/LayerRestartTest-Write.params");
   for( int k=1; k<argc; k++) {
      cl_args[k+2] = strdup(argv[k]);
   }
   status = buildandrun(num_cl_args, cl_args);
   if( status == PV_SUCCESS ) {
      free(cl_args[2]);
      cl_args[2] = strdup("input/LayerRestartTest-Check.params");
      status = buildandrun(num_cl_args, cl_args, NULL, &checkComparisonNonzero);
      if( status == PV_SUCCESS ) {
         free(cl_args[2]);
         cl_args[2] = strdup("input/LayerRestartTest-Read.params");
         status = buildandrun(num_cl_args, cl_args, NULL, &checkComparisonZero);
      }
   }
   free(cl_args[1]); cl_args[1] = NULL;
   free(cl_args[2]); cl_args[2] = NULL;
   free(cl_args); cl_args = NULL;

#ifdef PV_USE_MPI
   // Output status from each process, but go through root process since we might be using MPI across several machines
   // and only have a console on the root process
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if( rank == 0 ) {
      int otherprocstatus = status;
      int commsize;
      MPI_Comm_size(MPI_COMM_WORLD, &commsize);
      for(int r=0; r<commsize; r++) {
         if( r!= 0) MPI_Recv(&otherprocstatus, 1, MPI_INT, r, 59, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         if( otherprocstatus == PV_SUCCESS ) {
            printf("%s: rank %d process succeeded.\n", argv[0], r);
         }
         else {
            fprintf(stderr, "%s: rank %d process FAILED with return code %d\n", argv[0], r, otherprocstatus);
            status = PV_FAILURE;
         }
      }
   }
   else {
      MPI_Send(&status, 1, MPI_INT, 0, 59, MPI_COMM_WORLD);
   }

   if( !mpi_initialized_on_entry ) MPI_Finalize();
#endif // PV_USE_MPI
   return status;
}

int checkComparisonNonzero(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_FAILURE;
   int numLayers = hc->numberOfLayers();
   int layerIndex;
   HyPerLayer * layer;
   for( layerIndex=0; layerIndex<numLayers; layerIndex++ ) {
      layer = hc->getLayer(layerIndex);
      if( !strcmp(hc->getLayer(layerIndex)->getName(), "Comparison") ) break;
   }
   if( layerIndex >= numLayers) {
      fprintf(stderr, "%s: couldn't find layer \"Comparison\".", argv[0]);
      return PV_FAILURE;
   }
   pvdata_t * V = layer->getV();
   for( int k=0; k<layer->getNumNeurons(); k++ ) {
      if( V[k] ) {
         status = PV_SUCCESS;
         break;
      }
   }
   return status;
}

int checkComparisonZero(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   int numLayers = hc->numberOfLayers();
   int layerIndex;
   HyPerLayer * layer;
   for( layerIndex=0; layerIndex<numLayers; layerIndex++ ) {
      layer = hc->getLayer(layerIndex);
      if( !strcmp(hc->getLayer(layerIndex)->getName(), "Comparison") ) break;
   }
   if( layerIndex >= numLayers) {
      fprintf(stderr, "%s: couldn't find layer \"Comparison\".", argv[0]);
      return PV_FAILURE;
   }
   pvdata_t * V = layer->getV();
   for( int k=0; k<layer->getNumNeurons(); k++ ) {
      if( V[k] ) {
         fprintf(stderr, "Neuron %d: discrepancy %f\n", k, V[k]);
         status = PV_FAILURE;
      }
   }
   return status;
}
