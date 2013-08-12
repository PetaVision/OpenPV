#include <columns/buildandrun.hpp>
#include "MatchingPursuitProbe.hpp"

void * customgroups(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {
   char * param_file = NULL;
   int paramfilestatus = pv_getopt_str(argc, argv, "-p", &param_file);
   int cl_argc = argc + (paramfilestatus!=0 ? 2 : 0);
   char ** cl_argv = (char **) malloc((size_t) cl_argc * sizeof(char *));
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

   int status = buildandrun(cl_argc, cl_argv, NULL, NULL, &customgroups);
   for (int a=0; a<cl_argc; a++) {
      free(cl_argv[a]); cl_argv[a]=NULL;
   }
   free(cl_argv); cl_argv = NULL;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * name, HyPerCol * hc) {
   void * addedGroup = NULL;
   if (!strcmp(keyword, "MatchingPursuitProbe")) {
      addedGroup = new MatchingPursuitProbe(name, hc);
   }
   return addedGroup;
}
