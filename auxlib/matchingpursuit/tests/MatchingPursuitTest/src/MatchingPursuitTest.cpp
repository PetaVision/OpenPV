#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include <MatchingPursuitGroupHandler.hpp>
#include "CustomGroupHandler.hpp"

int main(int argc, char * argv[]) {
   int paramfilestatus = pv_getopt_str(argc, argv, "-p", NULL/*sVal*/, NULL/*paramusage*/);
   int cl_argc = argc + (paramfilestatus!=0 ? 2 : 0);
   char ** cl_argv = (char **) malloc((size_t) (cl_argc+1) * sizeof(char *));
   assert(cl_argv!=NULL);
   for (int a=0; a<argc; a++) {
      cl_argv[a] = strdup(argv[a]);
      assert(cl_argv[a]);
   }
   if (paramfilestatus!=0) {
      cl_argv[argc] = strdup("-p");
      assert(cl_argv[argc]);
      cl_argv[argc+1] = strdup("input/MatchingPursuitTest.params");
      assert(cl_argv[argc+1]);
   }
   cl_argv[cl_argc] = NULL;

   PV::ParamGroupHandler * customGroupHandlers[2];
   customGroupHandlers[0] = new PVMatchingPursuit::MatchingPursuitGroupHandler;
   customGroupHandlers[1] = new CustomGroupHandler;
   int status = buildandrun(cl_argc, cl_argv, NULL, NULL, customGroupHandlers, 2);
   for (int a=0; a<cl_argc; a++) {
      free(cl_argv[a]); cl_argv[a]=NULL;
   }
   free(cl_argv); cl_argv = NULL;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
