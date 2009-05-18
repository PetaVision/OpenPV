/*
 * pv_ca.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include "Retina1D.hpp"

#include "LinearPostConnProbe.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/connections/RandomConn.hpp>

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
//   PV::HyPerLayer * retina = new PV::Retina("Retina", hc);
   PV::HyPerLayer * retina = new PV::Retina1D("Retina", hc);
   PV::HyPerLayer * l1     = new PV::V1("L1", hc);
   PV::HyPerLayer * l1Inh  = new PV::V1("L1Inh", hc);
//   PV::HyPerLayer * l2     = new PV::V1("L2", hc);

   // connect the layers
   PV::HyPerConn * r_l1, * l1_l1Inh, * l1Inh_l1;
   r_l1     = new PV::RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   l1_l1Inh = new PV::HyPerConn( "L1 to L1Inh",  hc, l1,  l1Inh, CHANNEL_EXC);
//   l1Inh_l1 = new PV::RandomConn("L1Inh to L1",  hc, l1Inh,  l1, CHANNEL_INH);
//   new PV::PoolConn("L1 to L2",     hc, l1,     l2, CHANNEL_EXC);
//   new PV::RuleConn("L2 to L1",     hc, l2,     l1, CHANNEL_EXC);

   int locX = 39;  // image ON at 32 (so rule fires ON 33)
   int locY = 8;   // image ON
   int locF = 9;   // 0 OFF, 1 ON cell, ...

   locY = 0;
   locF = 0;   // 0 OFF, 1 ON cell, ...

   // add probes
   PV::PVLayerProbe * rProbe0  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 0);
   PV::PVLayerProbe * rProbe1  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 1);
   PV::PVLayerProbe * probe0  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 0);
   PV::PVLayerProbe * probe1  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 1);
   PV::PVLayerProbe * probe2  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 2);
   PV::PVLayerProbe * probe3  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 3);
   PV::PVLayerProbe * probe4  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 4);
   PV::PVLayerProbe * probe5  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 5);
   PV::PVLayerProbe * probe6  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 6);
   PV::PVLayerProbe * probe7  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 7);

   PV::PVLayerProbe * ptprobeI = new PV::PointProbe(61, locY, 0, "LI:x=61 f=0");

   PV::PVLayerProbe * ptprobe0 = new PV::PointProbe(61, locY, 0, "L1:x=61 f=0");
   PV::PVLayerProbe * ptprobe1 = new PV::PointProbe(61, locY, 1, "L1:x=61 f=1");
   PV::PVLayerProbe * ptprobe2 = new PV::PointProbe(61, locY, 2, "L1:x=61 f=2");
   PV::PVLayerProbe * ptprobe3 = new PV::PointProbe(61, locY, 3, "L1:x=61 f=3");
   PV::PVLayerProbe * ptprobe4 = new PV::PointProbe(61, locY, 4, "L1:x=61 f=4");
   PV::PVLayerProbe * ptprobe5 = new PV::PointProbe(61, locY, 5, "L1:x=61 f=5");
   PV::PVLayerProbe * ptprobe6 = new PV::PointProbe(61, locY, 6, "L1:x=61 f=6");
   PV::PVLayerProbe * ptprobe7 = new PV::PointProbe(61, locY, 7, "L1:x=61 f=7");

   PV::ConnectionProbe * cProbe0 = new PV::ConnectionProbe(2*15 + 0);
   PV::ConnectionProbe * cProbe1 = new PV::ConnectionProbe(2*15 + 1);

   PV::PostConnProbe * pcProbe0 = new PV::LinearPostConnProbe(PV::DimX, locY, 0);
   PV::PostConnProbe * pcProbe1 = new PV::LinearPostConnProbe(PV::DimX, locY, 1);
   PV::PostConnProbe * pcProbe2 = new PV::LinearPostConnProbe(PV::DimX, locY, 2);
   PV::PostConnProbe * pcProbe3 = new PV::LinearPostConnProbe(PV::DimX, locY, 3);
   PV::PostConnProbe * pcProbe4 = new PV::LinearPostConnProbe(PV::DimX, locY, 4);
   PV::PostConnProbe * pcProbe5 = new PV::LinearPostConnProbe(PV::DimX, locY, 5);
   PV::PostConnProbe * pcProbe6 = new PV::LinearPostConnProbe(PV::DimX, locY, 6);
   PV::PostConnProbe * pcProbe7 = new PV::LinearPostConnProbe(PV::DimX, locY, 7);

   retina->insertProbe(rProbe0);
   retina->insertProbe(rProbe1);
   //r_l1->insertProbe(cProbe);

   l1->insertProbe(probe0);
   l1->insertProbe(probe1);
   l1->insertProbe(probe2);
   l1->insertProbe(probe3);
   l1->insertProbe(probe4);
   l1->insertProbe(probe5);
   l1->insertProbe(probe6);
   l1->insertProbe(probe7);

//   l1Inh->insertProbe(ptprobeI);
//   l1Inh->insertProbe(probe0);

//   r_l1->insertProbe(cProbe0);
//   r_l1->insertProbe(cProbe1);

//   r_l1->insertProbe(pcProbe0);
//   r_l1->insertProbe(pcProbe1);
//   r_l1->insertProbe(pcProbe2);
//   r_l1->insertProbe(pcProbe3);
//   r_l1->insertProbe(pcProbe4);
//   r_l1->insertProbe(pcProbe5);
//   r_l1->insertProbe(pcProbe6);
//   r_l1->insertProbe(pcProbe7);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
