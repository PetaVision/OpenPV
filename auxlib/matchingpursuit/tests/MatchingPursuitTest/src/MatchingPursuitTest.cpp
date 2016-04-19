#include <columns/buildandrun.hpp>
#include <MatchingPursuitRegisterKeywords.hpp>
#include "MatchingPursuitProbe.hpp"

int main(int argc, char * argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.initialize();
   if (pv_initObj.getParams()==NULL) {
      char const * params_file = "input/MatchingPursuitTest.params";
      pv_initObj.setParams(params_file);
   }

   PVMatchingPursuit::MatchingPursuitRegisterKeywords(&pv_initObj);
   int status = pv_initObj.registerKeyword("MatchingPursuitProbe", createMatchingPursuitProbe);
   assert(status==PV_SUCCESS);
   
   status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
