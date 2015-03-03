/*
 * pv.cpp
 *
 */

// using ShrunkenPatchTestLayer
// activity/V are initialized to the global x/y/f position
// using uniform weights with total output strength of 1,
// all post synaptic cells should receive a total weighted input
// equal to thier global position
// ShrunkenPatchProbe checks whether he above suppositions are satisfied

#include <columns/buildandrun.hpp>
#include <io/io.c>
#include "CustomGroupHandler.hpp"
#include <assert.h>

int main(int argc, char * argv[]) {

   int status;
   // If params file was not specified, add input/ShrunkenPatchTest.params to command line arguments
   int paramfileabsent = pv_getopt_str(argc, argv, "-p", NULL/*sVal*/, NULL/*paramusage*/);
   int num_cl_args;
   char ** cl_args;
   if( paramfileabsent ) {
      num_cl_args = argc + 2;
      cl_args = (char **) malloc((num_cl_args+1)*sizeof(char *));
      cl_args[0] = argv[0];
      cl_args[1] = strdup("-p");
      cl_args[2] = strdup("input/ShrunkenPatchTest.params");
      for( int k=1; k<argc; k++) {
         cl_args[k+2] = strdup(argv[k]);
      }
      cl_args[num_cl_args] = NULL;
   }
   else {
      num_cl_args = argc;
      cl_args = argv;
   }
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler;
   status = buildandrun(num_cl_args, cl_args, NULL, NULL, &customGroupHandler, 1)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   if( paramfileabsent ) {
      free(cl_args[1]);
      free(cl_args[2]);
      free(cl_args);
   }
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
