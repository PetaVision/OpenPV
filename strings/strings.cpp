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
#include <src/io/PointProbe.hpp>
#include <src/io/PointLIFProbe.hpp>
#include <src/io/StatsProbe.hpp>
#include <src/io/PostConnProbe.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   HyPerLayer * S1  = new StringImage("S1", hc);
   HyPerLayer * C1  = new LIF("C1", hc);
   HyPerLayer * C1i = new LIF("C1i", hc);

   // connect the layers
   HyPerConn * S1_to_C1  = new HyPerConn("S1 to C1",  hc, S1, C1,  CHANNEL_EXC);
//   HyPerConn * C1_to_C1i = new HyPerConn("C1 to C1i", hc, C1, C1i, CHANNEL_EXC);
//   HyPerConn * C1i_to_C1 = new HyPerConn("C1i to C1", hc, C1i, C1, CHANNEL_INH);

   // probes
   //
   const PVLayerLoc * loc = C1->getLayerLoc();
   int x = (loc->nx + 2*loc->nb) / 2;
   x = 0;
   LayerProbe * lProbe   = new PointProbe(x, 0, 0, "simple");
   LayerProbe * lifProbe = new PointLIFProbe(x, 0, 0, "complex");
   PostConnProbe * cProbe = new PostConnProbe(x, 0, 0);
   cProbe->setOutputIndices(false);

   S1->insertProbe(lProbe);
   //C1->insertProbe(lifProbe);
   //S1_to_C1->insertProbe(cProbe);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
