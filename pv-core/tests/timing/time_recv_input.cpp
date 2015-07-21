/*
 * time_recv_input.cpp
 *
 *  timing tests for recvSynapticInput
 *
 *  Created on: Nov 5, 2008
 *      Author: rasmussn
 */

#include <stdlib.h>
#include "clock.h"

#include "../../src/include/pv_common.h"
#include "../../src/columns/HyPerCol.hpp"

#include "../../src/layers/Example.hpp"
#include "../../src/layers/Retina.hpp"

#include "../../src/layers/fileread.h"


const char input[] = "../../src/io/input/const_one_64x64.bin";

int main(int argc, char* argv[])
{
  //   int nloops = 2;
   int nloops = 1000;

   PV::HyPerCol* hc = new PV::HyPerCol("column", argc, argv);

   // construct layers
   PV::HyPerLayer* retina  = new PV::Retina("Example Retina", hc, input);
   PV::HyPerLayer* example = new PV::Example("Example Layer", hc);

   // set layer functions and parameters
   retina->setFuncs((INIT_FN) &fileread_init, (UPDATE_FN) &pvlayer_copyUpdate);

   // connect the layers
   PV::HyPerConn * conn = new PV::HyPerConn("Retina to Example", hc, retina, example);

   // finish initialization now that everything is connected
   hc->initFinish();

   // initialize activity to 1.0
   PVLayerCube* cube = retina->clayer->activity;
   for (int i = 0; i < cube->numItems; i++) {
      cube->data[i] = 1.0;
   }

   start_clock();
   double start = MPI_Wtime();
   for (int i = 0; i < nloops; i++) {
      conn->deliver(cube, 0);
      example->updateState(hc->simulationTime());
   }

   printf("\n");
   stop_clock();
   double elapsed = MPI_Wtime() - start;

   printf("\n[0] elapsed time (MPI_Wtime) = %f\n\n", (float) elapsed);

   // clean up (HyPerCol owns the layers, so don't delete them here)
   delete hc;

   return 0;
}
