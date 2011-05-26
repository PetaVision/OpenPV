/*
 * pv_example.cpp
 *
 *  Example of a PetaVision application.
 *
 */

#include <stdlib.h>

#include "include/pv_common.h"
#include "columns/HyPerCol.hpp"

#include "layers/Image.hpp"
#include "layers/Retina.hpp"
#include "layers/LIF.hpp"

using namespace PV;

int main(int argc, char* argv[])
{
   HyPerCol * hc = new HyPerCol("column", argc, argv, ".");

   // construct layers
   Image * image = new Image("Image", hc, hc->inputFile());
   HyPerLayer * retina  = new Retina("Retina", hc);
   HyPerLayer * l1      = new LIF("L1", hc);

   // connect the layers
   new PV::HyPerConn("Image to Retina", hc, image, retina, CHANNEL_EXC);
   new PV::HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);

   // finish initialization now that everything is connected
   hc->initFinish();

   // run the simulation for 2 time steps
   hc->run(1);

   // clean up (HyPerCol owns the layers, so don't delete them here)
   delete hc;

   return 0;
}

