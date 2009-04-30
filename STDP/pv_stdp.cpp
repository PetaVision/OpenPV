/*
 * pv_ca.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/layers/Retina.hpp>
#include "Retina1D.hpp"
#include <src/layers/V1.hpp>
#include "STDPConn.hpp"

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina1D = new PV::Retina1D("Retina1D", hc);
   PV::HyPerLayer * l1     = new PV::V1("L1", hc);

   // connect the layers
   PV::HyPerConn * conn = new PV::RandomConn("Retina1D to L1", hc, retina1D, l1, CHANNEL_EXC);

   int locX = 32;//39;  // image ON at 32 (so rule fires ON 33)
   int locY = 4;  // image ON
   int locF = 1;   // 0 OFF, 1 ON cell, ...

   int locK = 0 + locX + 64*locY;

   // add probes
   PV::PVLayerProbe * probe   = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * ptprobe = new PV::PointProbe(locX, locY, locF, "l1:");

   PV::ConnectionProbe * cprobe1 = new PV::ConnectionProbe(locK);
   PV::ConnectionProbe * cprobe2 = new PV::ConnectionProbe(locK+1);

//   retina->insertProbe(probeR);
   retina1D->insertProbe(probe);
   conn->insertProbe(cprobe1);
   conn->insertProbe(cprobe2);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;
   delete probe;
   delete ptprobe;
   delete cprobe1;
   delete cprobe2;

   return 0;
}

