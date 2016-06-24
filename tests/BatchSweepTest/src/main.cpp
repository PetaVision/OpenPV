/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "BatchSweepTestProbe.hpp"

#define MAIN_USES_CUSTOM_GROUPS

#ifdef MAIN_USES_CUSTOM_GROUPS
#include <columns/PV_Init.hpp>
#endif // MAIN_USES_CUSTOM_GROUPS

int main(int argc, char * argv[]) {

#ifndef MAIN_USES_CUSTOM_GROUPS
   //
   // The most basic invocation of PetaVision, suitable if you are running from a params file with no custom classes.
   //
   int status = buildandrun(argc, argv);
#else // MAIN_USES_CUSTOM_GROUPS
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   //
   // The method PV_Init::registerKeyword(char const * keyword, ObjectCreatorFn creator))
   // adds a new keyword known to the buildandrun functions.  The type ObjectCreatorFn is a
   // function pointer to a function taking a C-style string (the keyword used in params files)
   // and a pointer to a HyPerCol, and returning a BaseObject pointer, created with the C++ new operator.
   // It is expected that the object will be added to the HyPerCol, or (in the case of weight initializers
   // and weight normalizers) a HyPerConn, and will therefore automatically be deleted when the HyPerCol is deleted.
   // If you create a new class that buildandrun needs to know about, you should also write a function taking
   // the name of the object and the parent HyPerCol as arguments.  See, for example, createANNLayer in <layers/ANNLayer.cpp>.
   // The function pointer to that create-function can then be passed to the registerKeyword method:
   //
   // pv_initObj.registerKeyword("CustomClass1", createCustomClass1);
   // pv_initObj.registerKeyword("CustomClass2", createCustomClass2);
   // etc.
   pv_initObj.registerKeyword("BatchSweepTestProbe", createBatchSweepTestProbe);
   int status = buildandrun(&pv_initObj, NULL, NULL);
#endif // MAIN_USES_CUSTOM_GROUPS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
