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

#include "Retina1D.hpp"
#include "BiConn.hpp"
#include "LinearPostConnProbe.hpp"

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina, * l1s, * l2s, * l2c, * l3s, * l3c, * l4s, * l4c;

   retina = new PV::Retina1D("Retina", hc);
   l1s = new PV::V1("L1 Simple",  hc);
   l2s = new PV::V1("L2 Simple",  hc);
   l2c = new PV::V1("L2 Complex", hc);
   l3s = new PV::V1("L3 Simple", hc);
   l3c = new PV::V1("L3 Complex", hc);
   l4s = new PV::V1("L4 Simple", hc);
   l4c = new PV::V1("L4 Complex", hc);     // readout layer

   // inhibitory layers
   PV::HyPerLayer * l1sInh;

   l1sInh = new PV::V1("L1 Simple Inh", hc);      // when l1s neuron fires, inhibit other features
   //   l4Inh = new PV::V1("L4Inh", hc);

   // connect the layers
   PV::HyPerConn * r_l1s, * l1s_l2s, * l2s_l2c, * l2s_l3s, * l2c_l3s, * l3s_l3c, * l3c_l4s;
   PV::HyPerConn *l4s_l4c;

   // connections for STDP learning
   r_l1s   = new PV::RandomConn("Retina to L1 Simple",     hc, retina, l1s, CHANNEL_EXC);

// r_l1s   = new PV::BiConn("Retina to L1 Simple",     hc, retina, l1s, CHANNEL_EXC, 11);
//   l1s_l2s = new PV::BiConn("L1 Simple to L2 Simple",  hc, l1s,    l2s, CHANNEL_EXC, 1121);
//   l2s_l2c = new PV::BiConn("L2 Simple to L2 Complex", hc, l2s,    l2c, CHANNEL_EXC, 2122);
//   l2c_l3s = new PV::BiConn("L2 Complex to L3 Simple", hc, l2c,    l3s, CHANNEL_EXC, 2231);
//   l3s_l3c = new PV::BiConn("L3 Simple to L3 Complex", hc, l3s,    l3c, CHANNEL_EXC, 3132);
//   l3c_l4s = new PV::BiConn("L3 Complex to L4 Simple", hc, l3c,    l4s, CHANNEL_EXC, 3241);
//   l4s_l4c = new PV::BiConn("L4 Simple to L4 Complex", hc, l4s,    l4c, CHANNEL_EXC, 4142);

   // inhibitory connections
   PV::HyPerConn * l1s_l1sInh, * l1sInh_l1s;

   l1s_l1sInh = new PV::BiConn("L1 Simple to L1 Simple Inh", hc, l1s, l1sInh, CHANNEL_EXC, 11110);
   l1sInh_l1s = new PV::BiConn("L1 Simple Inh to L1 Simple", hc, l1sInh, l1s, CHANNEL_INH, 11011);

//   l2_l3 = new PV::RandomConn("L2 to L3", hc, l2, l3, CHANNEL_EXC);
//   l3_l4 = new PV::RandomConn("L3 to L4", hc, l3, l4, CHANNEL_EXC);

//   l4_l4Inh = new PV::RandomConn("L4 to L4Inh", hc, l4, l4Inh, CHANNEL_EXC);
//   l4Inh_l4 = new PV::RandomConn("L4Inh to L4", hc, l4Inh, l4, CHANNEL_INH);

   int locX = 0;
   int locY = 0;  // image ON
   int locF = 0;   // 0 OFF, 1 ON cell, ...

   int locK = 0; // 7; // (7 for right cells)

   // add probes
   PV::PVLayerProbe * probe0  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF);
   PV::PVLayerProbe * probe1  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+1);
   PV::PVLayerProbe * probe2  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+2);
   PV::PVLayerProbe * probe3  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+3);
   PV::PVLayerProbe * probe4  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+4);
   PV::PVLayerProbe * probe5  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+5);
   PV::PVLayerProbe * probe6  = new PV::LinearActivityProbe(hc, PV::DimX, locY, locF+6);

   locK = 7;

   PV::ConnectionProbe * cprobe0 = new PV::ConnectionProbe(locK);
   PV::ConnectionProbe * cprobe1 = new PV::ConnectionProbe(locK+1);
   PV::ConnectionProbe * cprobe2 = new PV::ConnectionProbe(locK+2);
   PV::ConnectionProbe * cprobe3 = new PV::ConnectionProbe(locK+3);
   PV::ConnectionProbe * cprobe4 = new PV::ConnectionProbe(locK+4);
   PV::ConnectionProbe * cprobe5 = new PV::ConnectionProbe(locK+5);

   PV::LinearPostConnProbe * lpcprobe0 = new PV::LinearPostConnProbe(PV::DimX, locY, 0);
   PV::LinearPostConnProbe * lpcprobe1 = new PV::LinearPostConnProbe(PV::DimX, locY, 1);
   PV::LinearPostConnProbe * lpcprobe2 = new PV::LinearPostConnProbe(PV::DimX, locY, 2);
   PV::LinearPostConnProbe * lpcprobe3 = new PV::LinearPostConnProbe(PV::DimX, locY, 3);

   PV::PostConnProbe * pcprobe0 = new PV::PostConnProbe(locK);

   locF = 7;
   PV::PVLayerProbe * ptprobe = new PV::PointProbe(locX, locY, locF, "l1i:");

   retina->insertProbe(probe0);
   retina->insertProbe(probe1);

   l1s->insertProbe(probe0);
   l1s->insertProbe(probe1);
   l1s->insertProbe(probe2);
   l1s->insertProbe(probe3);
//   l1s->insertProbe(ptprobe);

//   l1sInh->insertProbe(probe0);

   /****
   l2s->insertProbe(probe0);
   l2s->insertProbe(probe1);
   l2s->insertProbe(probe2);
   l2s->insertProbe(probe3);
   l2s->insertProbe(probe4);
   l2s->insertProbe(probe5);
   l2s->insertProbe(probe6);

   l2c->insertProbe(probe0);
   l2c->insertProbe(probe1);
   l2c->insertProbe(probe2);
   l2c->insertProbe(probe3);
   l2c->insertProbe(probe4);

   l3s->insertProbe(probe0);
   l3s->insertProbe(probe1);
   l3s->insertProbe(probe2);
   l3s->insertProbe(probe3);
   l3s->insertProbe(probe4);
   l3s->insertProbe(probe5);
   l3s->insertProbe(probe6);

   l3c->insertProbe(probe0);
   l3c->insertProbe(probe1);
   l3c->insertProbe(probe2);
   l3c->insertProbe(probe3);
   l3c->insertProbe(probe4);

   l4s->insertProbe(probe0);
   l4s->insertProbe(probe1);

   *****/

//   l4c->insertProbe(probe0);

//   l1s->insertProbe(ptprobe);

   r_l1s->insertProbe(lpcprobe0);
   r_l1s->insertProbe(lpcprobe1);
   r_l1s->insertProbe(lpcprobe2);
   r_l1s->insertProbe(lpcprobe3);
   r_l1s->insertProbe(pcprobe0);
//   l2c_l3s->insertProbe(cprobe1);
//   l2c_l3s->insertProbe(cprobe2);
//   l2c_l3s->insertProbe(cprobe3);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}

