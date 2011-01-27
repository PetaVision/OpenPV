/*
 * strings.cpp
 *
 *  Created on: Jan 27, 2011
 *      Author: Craig Rasmussen
 */

#include <stdlib.h>

#include "StringImage.hpp"
#include <src/columns/HyPerCol.hpp>
#include <src/layers/LIF.hpp>
#include <src/connections/HyPerConn.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   HyPerLayer * strings = new StringImage("Strings", hc);
   HyPerLayer * complex = new LIF("Complex", hc);

   // connect the layers
   new HyPerConn("Strings to Complex", hc, strings, complex, CHANNEL_EXC);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
