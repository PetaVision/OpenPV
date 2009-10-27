/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include "Gratings.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/layers/Image.hpp>
#include <src/layers/ImageCreator.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/StatsProbe.hpp>

#include <src/io/imageio.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   //   ImageCreator * image = new ImageCreator("Image", hc);
   Image * image = new Gratings("Image", hc);

   HyPerLayer * retina = new Retina("Retina", hc, image);
   HyPerLayer * l1 = new V1("L1", hc);
   HyPerLayer * l1Inh = new V1("L1Inh", hc);

   HyPerConn * r_l1   = new HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   //HyPerConn * l1_inh = new HyPerConn("L1 to L1Inh", hc, l1, l1Inh, CHANNEL_EXC);

   // add probes

   HyPerLayer * displayLayer = retina;

   const int ny = displayLayer->clayer->loc.ny;
   PVLayerProbe * laProbes[ny]; // array of ny pointers to LinearActivityProbe

   for (int iy = 1; iy < ny-1; iy++) {
      laProbes[iy] = new LinearActivityProbe(hc, DimX, iy, 0);
      displayLayer->insertProbe(laProbes[iy]);
   }

   PVLayerProbe * statsR     = new StatsProbe(BufActivity, "R    :");
   PVLayerProbe * statsL1    = new StatsProbe(BufActivity, "L1   :");
   PVLayerProbe * statsL1Inh = new StatsProbe(BufActivity, "L1Inh:");

   retina->insertProbe(statsR);
   l1->insertProbe(statsL1);
//   l1Inh->insertProbe(statsL1Inh);

   PVLayerProbe * ptprobe = new PointProbe(16, 16, 0, "L1:(16,16)");
   l1->insertProbe(ptprobe);

//   ConnectionProbe * cProbe0 = new ConnectionProbe(2*5 + 0);
   ConnectionProbe * cProbe = new PostConnProbe(24, 16, 0);
   r_l1->insertProbe(cProbe);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
