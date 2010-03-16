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

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   HyPerLayer * retina, * l1s, * l2s, * l2c, * l3s, * l3c, * l4s, * l4c;

   retina = new Retina1D("Retina", hc);
   l1s = new V1("L1 Simple",  hc);
   l2s = new V1("L2 Simple",  hc);
   l2c = new V1("L2 Complex", hc);
   l3s = new V1("L3 Simple", hc);
   l3c = new V1("L3 Complex", hc);
   l4s = new V1("L4 Simple", hc);
   l4c = new V1("L4 Complex", hc);     // readout layer

   // inhibitory layers
   HyPerLayer * l1sInh;

   l1sInh = new V1("L1 Simple Inh", hc);      // when l1s neuron fires, inhibit other features
   //   l4Inh = new V1("L4Inh", hc);

   // connect the layers
   HyPerConn * r_l1s, * l1s_l2s, * l2s_l2c, * l2s_l3s, * l2c_l3s, * l3s_l3c, * l3c_l4s;
   HyPerConn *l4s_l4c;

   // connections for STDP learning
   r_l1s   = new RandomConn("Retina to L1 Simple",     hc, retina, l1s, CHANNEL_EXC);

// r_l1s   = new BiConn("Retina to L1 Simple",     hc, retina, l1s, CHANNEL_EXC, 11);
//   l1s_l2s = new BiConn("L1 Simple to L2 Simple",  hc, l1s,    l2s, CHANNEL_EXC, 1121);
//   l2s_l2c = new BiConn("L2 Simple to L2 Complex", hc, l2s,    l2c, CHANNEL_EXC, 2122);
//   l2c_l3s = new BiConn("L2 Complex to L3 Simple", hc, l2c,    l3s, CHANNEL_EXC, 2231);
//   l3s_l3c = new BiConn("L3 Simple to L3 Complex", hc, l3s,    l3c, CHANNEL_EXC, 3132);
//   l3c_l4s = new BiConn("L3 Complex to L4 Simple", hc, l3c,    l4s, CHANNEL_EXC, 3241);
//   l4s_l4c = new BiConn("L4 Simple to L4 Complex", hc, l4s,    l4c, CHANNEL_EXC, 4142);

   // inhibitory connections
   HyPerConn * l1s_l1sInh, * l1sInh_l1s;

   l1s_l1sInh = new BiConn("L1 Simple to L1 Simple Inh", hc, l1s, l1sInh, CHANNEL_EXC, 11110);
   l1sInh_l1s = new BiConn("L1 Simple Inh to L1 Simple", hc, l1sInh, l1s, CHANNEL_INH, 11011);

//   l2_l3 = new RandomConn("L2 to L3", hc, l2, l3, CHANNEL_EXC);
//   l3_l4 = new RandomConn("L3 to L4", hc, l3, l4, CHANNEL_EXC);

//   l4_l4Inh = new RandomConn("L4 to L4Inh", hc, l4, l4Inh, CHANNEL_EXC);
//   l4Inh_l4 = new RandomConn("L4Inh to L4", hc, l4Inh, l4, CHANNEL_INH);

   int locX = 0;
   int locY = 0;  // image ON
   int locF = 0;   // 0 OFF, 1 ON cell, ...

   int locK = 0; // 7; // (7 for right cells)

   // add probes
#ifdef PROBES
   PVLayerProbe * probe0  = new LinearActivityProbe(hc, DimX, locY, locF);
   PVLayerProbe * probe1  = new LinearActivityProbe(hc, DimX, locY, locF+1);
   PVLayerProbe * probe2  = new LinearActivityProbe(hc, DimX, locY, locF+2);
   PVLayerProbe * probe3  = new LinearActivityProbe(hc, DimX, locY, locF+3);
   PVLayerProbe * probe4  = new LinearActivityProbe(hc, DimX, locY, locF+4);
   PVLayerProbe * probe5  = new LinearActivityProbe(hc, DimX, locY, locF+5);
   PVLayerProbe * probe6  = new LinearActivityProbe(hc, DimX, locY, locF+6);

   locK = 7;

   ConnectionProbe * cprobe0 = new ConnectionProbe(locK);
   ConnectionProbe * cprobe1 = new ConnectionProbe(locK+1);
   ConnectionProbe * cprobe2 = new ConnectionProbe(locK+2);
   ConnectionProbe * cprobe3 = new ConnectionProbe(locK+3);
   ConnectionProbe * cprobe4 = new ConnectionProbe(locK+4);
   ConnectionProbe * cprobe5 = new ConnectionProbe(locK+5);

   LinearPostConnProbe * lpcprobe0 = new LinearPostConnProbe(DimX, locY, 0);
   LinearPostConnProbe * lpcprobe1 = new LinearPostConnProbe(DimX, locY, 1);
   LinearPostConnProbe * lpcprobe2 = new LinearPostConnProbe(DimX, locY, 2);
   LinearPostConnProbe * lpcprobe3 = new LinearPostConnProbe(DimX, locY, 3);

   PostConnProbe * pcprobe0 = new PostConnProbe(locK);

   locF = 7;
   PVLayerProbe * ptprobe = new PointProbe(locX, locY, locF, "l1i:");

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
#endif

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}

