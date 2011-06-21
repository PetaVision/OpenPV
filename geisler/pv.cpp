/*
 * pv_ca.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/LIF.hpp>
#include <src/connections/HyPerConn.hpp>
#include <src/connections/CocircConn.hpp>

using namespace PV;

int main_pv(int argc, char* argv[])
{
   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   HyPerLayer * retina = new Retina("Retina", hc);
   HyPerLayer * l1     = new LIF("L1", hc);

   // connect the layers
   new HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   new CocircConn("L1 to L1",    hc, l1,     l1, CHANNEL_EXC);

   int locX = 39;
   int locY = 31; // 53;
   int locF = 0;

   // add probes
   LayerProbe * probe   = new LinearActivityProbe(hc, PV::DimX, locY, locF);
   LayerProbe * ptprobe = new PointProbe(locX, locY, locF, "L1:");

   l1->insertProbe(probe);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;
   delete probe;
   delete ptprobe;

   return 0;
}
