#include <columns/buildandrun.hpp>
#include "MaskTestInputLayer.hpp"
#include "MaskTestLayer.hpp"

#define MAIN_USES_CUSTOMGROUPS

int main(int argc, char * argv[]) {

#ifdef MAIN_USES_CUSTOMGROUPS
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("MaskTestInputLayer", Factory::create<MaskTestInputLayer>);
   pv_initObj.registerKeyword("MaskTestLayer", Factory::create<MaskTestLayer>);
   int status = buildandrun(&pv_initObj, NULL, NULL);
#else // MAIN_USES_CUSTOMGROUPS
   int status = buildandrun(argc, argv, NULL, NULL);
#endif // MAIN_USES_CUSTOMGROUPS

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
