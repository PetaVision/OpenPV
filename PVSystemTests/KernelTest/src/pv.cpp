/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "KernelTestProbe.hpp"
#include "KernelTestGroupHandler.hpp"

int main(int argc, char * argv[]) {

   int status;
   KernelTestGroupHandler * groupHandlerList[1];
   groupHandlerList[0] = new KernelTestGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, (ParamGroupHandler **) groupHandlerList, 1);
   delete groupHandlerList[0];
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
