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
#include <src/io/PostConnProbe.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina, * lgn, * v1, * v1inh;

   retina = new PV::Retina("Retina", hc);
   lgn = new PV::V1("LGN",  hc);
   v1 = new PV::V1("V1",  hc);
   v1inh = new PV::V1("V1 Inhib", hc);

   // connect the layers
   PV::HyPerConn * r_lgn, * lgn_v1, * v1_v1inh, * v1_v1, * v1inh_v1, * v1inh_v1inh;

   // connections for STDP learning
   r_lgn   = new PV::HyPerConn("Retina to LGN",     hc, retina, lgn, CHANNEL_EXC);
   lgn_v1  = new PV::HyPerConn("LGN to V1",         hc, lgn, v1, CHANNEL_EXC);
   v1_v1inh= new PV::HyPerConn("V1 to V1 Inhib",    hc, v1, v1inh, CHANNEL_EXC);
   v1_v1   = new PV::HyPerConn("V1 Lateral",        hc, v1, v1, CHANNEL_EXC);
   v1inh_v1= new PV::HyPerConn("V1 Inhib to V1",    hc, v1inh, v1, CHANNEL_INH);
//   v1inh_v1inh= new PV::HyPerConn("V1 Inhib to V1 Inhib", hc, v1inh, v1inh, CHANNEL_INH);

   int locX = 64;
   int locY = 63;  // image ON
   int locF = 0;   // 0 OFF, 1 ON cell, ...

   // add probes
   PV::PVLayerProbe * probe0  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * probe1  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+1);
   PV::PVLayerProbe * probe2  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+2);
   PV::PVLayerProbe * probe3  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+3);

   PV::PVLayerProbe * ptprobe = new PV::PointProbe(locX, locY, locF, "l1i:");

   v1->insertProbe(probe0);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}

