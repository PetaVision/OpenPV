/*
 * pv.cpp
 *
 */

#include "BinningTestProbe.hpp"
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

#define MAIN_USES_CUSTOM_GROUPS

#ifdef MAIN_USES_CUSTOM_GROUPS
#include <columns/PV_Init.hpp>
#endif // MAIN_USES_CUSTOM_GROUPS

int main(int argc, char *argv[]) {

#ifndef MAIN_USES_CUSTOM_GROUPS
   //
   // The most basic invocation of PetaVision, suitable if you are running from a params file with
   // no custom classes.
   //
   int status = buildandrun(argc, argv);
#else // MAIN_USES_CUSTOM_GROUPS
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   //
   // If you create a new class that buildandrun needs to know about, you need to register the
   // keyword
   // with the PV_Init object.  Generally, this can be done with the Factory::create function
   // template:
   //
   // pv_initObj.registerKeyword("CustomClass1", Factory::create<CustomClass1>);
   // pv_initObj.registerKeyword("CustomClass2", Factory::create<CustomClass2>);
   // etc.
   //
   pv_initObj.registerKeyword("BinningTestProbe", Factory::create<BinningTestProbe>);
   int status = buildandrun(&pv_initObj);
#endif // MAIN_USES_CUSTOM_GROUPS
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
