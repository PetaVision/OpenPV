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
#include <src/layers/V1.hpp>
#include <src/connections/HyPerConn.hpp>
#include <src/connections/CocircConn.hpp>

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina = new PV::Retina("Retina", hc);
   PV::HyPerLayer * l1     = new PV::V1("L1", hc);

   // connect the layers
   new PV::HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   new PV::CocircConn("L1 to L1",    hc, l1,     l1, CHANNEL_EXC);

   int locX = 39;
   int locY = 53; //31; // 53;
   int locF = 0;

   // add probes
   PV::PVLayerProbe * probe   = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * ptprobe = new PV::PointProbe(locX, locY, locF, "L1:");

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
