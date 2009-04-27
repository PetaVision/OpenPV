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
   PV::HyPerConn * conn = new PV::STDPConn("Retina1D to L1", hc, retina1D, l1, CHANNEL_EXC);

   int locX = 32;//39;  // image ON at 32 (so rule fires ON 33)
   int locY = 4;  // image ON
   int locF = 1;   // 0 OFF, 1 ON cell, ...

   int locK = 0 + 2*locX + 2*64*locY;

   // add probes
   PV::PVLayerProbe * probe   = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * ptprobe = new PV::PointProbe(locX, locY, locF, "l1:");

   conn->writeWeights(locK);
   conn->writeWeights(locK+1);

//   retina->insertProbe(probeR);
   retina1D->insertProbe(probe);

   // run the simulation
   hc->initFinish();
   hc->run();

   conn->writeWeights(locK);
   conn->writeWeights(locK+1);

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;
   delete probe;
   delete ptprobe;

   return 0;
}

