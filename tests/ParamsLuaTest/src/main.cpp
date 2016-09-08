/*
 * main.cpp for ParamsLuaTest
 *
 *  Created on: Jul 1, 2016
 *      Author: peteschultz
 */


#include "ColumnArchive.hpp"
#include <cmath>
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <columns/HyPerCol.hpp>
#include <layers/HyPerLayer.hpp>
#include <connections/HyPerConn.hpp>

int main(int argc, char * argv[]) {
   float tolerance = 1.0e-5;

   PV::PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   int rank = pv_initObj.getWorldRank();
   int status = PV_SUCCESS;
   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the params file argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt_str(argc, argv, "-c", NULL, NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt(argc, argv, "-r", NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the restart flag.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank==0) {
         pvErrorNoExit().printf("This test uses compares a hard-coded .params.lua file with a hard-coded .params file, and the results of the two runs are compared.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   std::string paramsfile("input/ParamsLuaTest.params");
   std::string paramsluafile("output/pv.params.lua");
   pv_initObj.setParams(paramsfile.c_str());
   PV::HyPerCol * hc1 = createHyPerCol(&pv_initObj);
   if (hc1==nullptr) { pvError() << "setParams(\"" << paramsfile << "\") failed.\n"; }
   status = hc1->run();
   if (status != PV_SUCCESS) { pvError() << "Running with \"" << paramsfile << "\" failed.\n"; }
   ColumnArchive columnArchive1(hc1, tolerance, tolerance); // Archive the layer and connection data since changing the params file is destructive.

   pv_initObj.setParams(paramsluafile.c_str());
   PV::HyPerCol * hc2 = createHyPerCol(&pv_initObj);
   if (hc2==nullptr) { pvError() << "setParams(\"" << paramsluafile << "\") failed.\n"; }
   status = hc2->run();
   if (status != PV_SUCCESS) { pvError() << "Running with \"" << paramsluafile << "\" failed.\n"; }
   ColumnArchive columnArchive2(hc2, tolerance, tolerance);

   return columnArchive1==columnArchive2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
