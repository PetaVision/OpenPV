/*
 * pv_example.cpp
 *
 *  Example of a PetaVision application.
 *
 */

#include <stdlib.h>

#include "include/pv_common.h"
#include "columns/HyPerCol.hpp"

#include "layers/Example.hpp"
#include "layers/Retina.hpp"

#include "layers/fileread.h"

using namespace PV;

int main(int argc, char* argv[])
{
   const char input[] = INPUT_PATH "const_one_64x64.tif";

   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // construct layers
   HyPerLayer * retina  = new Retina("Example Retina", hc, input);
   HyPerLayer * example = new Example("Example Layer", hc);

   // connect the layers
   new PV::HyPerConn("Retina to Example", hc, retina, example, CHANNEL_EXC);

   // finish initialization now that everything is connected
   hc->initFinish();

   // run the simulation for 2 time steps
   hc->run(2);

   hc->writeState();

   // clean up (HyPerCol owns the layers, so don't delete them here)
   delete hc;

   return 0;
}

