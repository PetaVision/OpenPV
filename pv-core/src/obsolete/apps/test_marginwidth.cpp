/*
 * test_marginwidth.cpp
 *
 *  Created on: Nov 8, 2011
 *      Author: pschultz
 */

#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/layers/Retina.hpp"
#include "../src/layers/ANNLayer.hpp"
#include "../src/io/io.h"

#define ARGC 3

using namespace PV;

int runonecolumn(int argc, char * argv[], int correctvalue);

int main(int argc, char * argv[]) {
   char * cl_args[3];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = NULL;
#ifdef PV_USE_MPI
   int mpi_initialized_on_entry;
   MPI_Initialized(&mpi_initialized_on_entry);
   if( !mpi_initialized_on_entry ) MPI_Init(&argc, &argv);
   // If mpi has already been initialized when a new HyPerCol is created,
   // the HyPerCol won't call MPI_Finalize when it is deleted.
   // That way we can run several marginwidth tests in the same job.
#endif // PV_USE_MPI

   int status = PV_SUCCESS;
   int correctvalue = PV_SUCCESS;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/correctsize_one_to_one.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != correctvalue ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/correctsize_many_to_one.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != correctvalue ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/correctsize_one_to_many.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != correctvalue ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/toolarge_one_to_one.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != correctvalue ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/toolarge_many_to_one.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != correctvalue ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/toolarge_one_to_many.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != correctvalue ? PV_MARGINWIDTH_FAILURE : status;

   correctvalue = PV_MARGINWIDTH_FAILURE; // The next three tests return marginwidth errors.

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/toosmall_one_to_one.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != PV_MARGINWIDTH_FAILURE ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/toosmall_many_to_one.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != PV_MARGINWIDTH_FAILURE ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/toosmall_one_to_many.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != PV_MARGINWIDTH_FAILURE ? PV_MARGINWIDTH_FAILURE : status;

   free(cl_args[2]);
   cl_args[2] = strdup("input/test_marginwidth_paramfiles/margin_larger_than_layer.params");
   status = runonecolumn(ARGC, cl_args, correctvalue) != PV_MARGINWIDTH_FAILURE ? PV_MARGINWIDTH_FAILURE : status;

   for( int k=0; k<ARGC; k++ )
   {
      free(cl_args[k]);
   }
#ifdef PV_USE_MPI
   MPI_Finalize();
#endif // PV_USE_MPI
   return status;
}

int runonecolumn(int argc, char * argv[], int correctvalue) {
   char * paramfilename;
   pv_getopt_str(argc, argv, "-p", &paramfilename);
   HyPerCol * hc = new HyPerCol("column", argc, argv);
   
   const char * preLayerName = "presynaptic layer";
   const char * postLayerName = "postsynaptic layer";
   
   HyPerLayer * pre = new Retina("presynaptic layer", hc);
   assert(pre);
   HyPerLayer * post = new ANNLayer("postsynaptic layer", hc);
   assert(post);
   HyPerConn * conn = new KernelConn("pre to post connection", hc, preLayerName, postLayerName);
   assert(conn);
   
   bool rootproc = hc->icCommunicator()->commRank()==0;
   if(rootproc) {
      fflush(stdout);
      printf("%s: Beginning test of %s.  For this test, ", argv[0], argv[2]);
      if( correctvalue == PV_MARGINWIDTH_FAILURE) {
         printf("marginwidth failures are the correct behavior.\n");
      }
      else {
         printf("there should not be any marginwidth failures.\n");
      }
      fflush(stdout);
   }
   int status = hc->run();
   delete hc;
   if( status != PV_SUCCESS && status != PV_MARGINWIDTH_FAILURE ) {
      fprintf(stderr, "test_marginwidth failed on params file \"%s\" with error %d.  Exiting.\n", argv[2], status);
      exit(status);
   }

   if(rootproc) {
      if( status == correctvalue ) {
         printf("%s: Test %s passed.\n\n", argv[0], argv[2]);
      }
      else {
         printf("%s: Test %s FAILED.\n\n", argv[0], argv[2]);
      }
      fflush(stdout);
   }
   return status;
}
