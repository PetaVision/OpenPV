/*
 * gamma.cpp
 *
 *  Created on: June 8, 2009
 *      Author: gar
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>

#include "RetinaGrating.hpp"

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina, * v1, * v1Inh;

   retina = new PV::RetinaGrating("Retina", hc, 2);
   v1     = new PV::V1("V1",  hc);
   v1Inh  = new PV::V1("V1 Inhib",  hc);

// maybe later
//   lgn    = new PV::V1("LGN",  hc);

   // connect the layers
   PV::HyPerConn * r_v1, * r_v1Inh, * v1_v1, * v1_v1Inh, * v1Inh_v1, * v1Inh_v1Inh;

//   r_lgn       = new PV::HyPerConn("Retina to LGN",  hc, retina, lgn,   CHANNEL_EXC);
//   lgn_v1      = new PV::HyPerConn("LGN to V1",      hc, lgn,    v1,    CHANNEL_EXC);

   r_v1        = new PV::HyPerConn("Retina to V1",    hc, retina, v1,    CHANNEL_EXC);
   r_v1Inh     = new PV::HyPerConn("Retina to V1Inh", hc, retina, v1Inh, CHANNEL_EXC);
   v1_v1       = new PV::HyPerConn("V1 to V1",        hc, v1,     v1,    CHANNEL_EXC);
   v1_v1Inh    = new PV::HyPerConn("V1 to V1Inh",     hc, v1,     v1Inh, CHANNEL_EXC);
   v1Inh_v1    = new PV::HyPerConn("V1Inh to V1",     hc, v1Inh,  v1,    CHANNEL_INH);
   v1Inh_v1Inh = new PV::HyPerConn("V1Inh to V1Inh",  hc, v1Inh,  v1Inh, CHANNEL_INH);

   int locX = 0;
   int locY = 0;
   int locF = 0;

   // add probes
   PV::PVLayerProbe * probe0 = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * probeRetina = new PV::LinearActivityProbe("retina.txt", hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * probeV1 = new PV::LinearActivityProbe("v1.txt", hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * probeV1Inh = new PV::LinearActivityProbe("v1Inh.txt", hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * ptprobe = new PV::PointProbe(locX, locY, locF, "l1i:");

   v1->insertProbe(probe0);
   retina->insertProbe(probeRetina);
   v1->insertProbe(probeV1);
   v1Inh->insertProbe(probeV1Inh);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;
   delete probe0;
   delete ptprobe;

   return 0;
}
