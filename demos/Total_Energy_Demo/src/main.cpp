/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>

#undef MAIN_USES_CUSTOMGROUPS

#ifdef MAIN_USES_CUSTOMGROUPS
#include <io/ParamGroupHandler.hpp>
#include "CustomGroupHandler.hpp"
// CustomGroupHandler is for adding objects not supported by CoreParamGroupHandler().
#endif // MAIN_USES_CUSTOMGROUPS

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_CUSTOMGROUPS
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
   delete customGroupHandler;
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_CUSTOMGROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
