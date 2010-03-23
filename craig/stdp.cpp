/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: Craig Rasmussen
 */

#include <stdlib.h>

#include "LinearPostConnProbe.hpp"
#include "BiConn.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/GLDisplay.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/StatsProbe.hpp>
#include <src/layers/Gratings.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/connections/RandomConn.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   //
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the image
   //
   Image * image = new Gratings("Image", hc);

   // create the layers
   //
   HyPerLayer * retina = new Retina("Retina", hc, image);
   HyPerLayer * l1     = new V1("L1", hc);
//   HyPerLayer * l1Inh  = new V1("L1Inh", hc);

   // connect the layers
   //
   HyPerConn * r_l1, * l1_l1Inh, * l1Inh_l1;
   r_l1     = new BiConn("Retina to L1", hc, retina, l1, CHANNEL_EXC, 0);
//   r_l1     = new RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
//   l1_l1Inh = new HyPerConn( "L1 to L1Inh",  hc, l1,  l1Inh, CHANNEL_EXC);
//   l1Inh_l1 = new HyPerConn("L1Inh to L1",  hc, l1Inh,  l1, CHANNEL_INH);

#ifdef DISPLAY
   GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);
   display->setDelay(0);
   display->setImage(image);
   display->addLayer(retina);
   display->addLayer(l1);
   display->addLayer(l1Inh);
#endif

   int nfPost = 8;
   int locX = 5;
   int locY = 0;
   int locF = 0;   // 0 OFF, 1 ON cell, ...

   // add probes
   //

//   PVLayerProbe * rProbe0  = new LinearActivityProbe(hc, PV::DimX, locY, 0);
//   PVLayerProbe * ptprobe1 = new PointProbe(61, locY, 1, "L1:x=61 f=1");
//   ConnectionProbe * cProbe0 = new ConnectionProbe(2*5 + 0);
//   PostConnProbe * pcProbe0 = new LinearPostConnProbe(PV::DimX, locY, 0);

//   PostConnProbe * pcProbe = new PostConnProbe(37, 38, 0);
//   r_l1->insertProbe(pcProbe);

//   StatsProbe * sProbe = new StatsProbe(PV::BufActivity, "l1");
//   l1->insertProbe(sProbe);

   // run the simulation
   //

   if (hc->columnId() == 0) {
      printf("[0]: Running simulation ...");  fflush(stdout);
   }

   hc->run();

   if (hc->columnId() == 0) {
      printf("\n[0]: Finished\n");
   }

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
