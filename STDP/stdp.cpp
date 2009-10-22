/*
 * stdp.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

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
   ImageCreator * image = new ImageCreator("Image", hc);
   HyPerLayer * retina = new Retina("Retina", hc, image);
   HyPerLayer * l1 = new V1("L1", hc);

   HyPerConn * r_l1 = new HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);

   // add probes

   const int ny = retina->clayer->loc.ny;
   PVLayerProbe * laProbes[ny]; // array of ny pointers to LinearActivityProbe

   for (int iy = 1; iy < ny-1; iy++) {
	   laProbes[iy] = new PV::LinearActivityProbe(hc, DimX, iy, 0);
       l1->insertProbe(laProbes[iy]);
   }

   PVLayerProbe * statsProbe = new StatsProbe(BufActivity, "L1");
   l1->insertProbe(statsProbe);

//   PVLayerProbe * ptprobeI = new PointProbe(61, locY, 0, "LI:x=61 f=0");

//   ConnectionProbe * cProbe0 = new ConnectionProbe(2*5 + 0);
//   ConnectionProbe * cProbe0 = new PostConnProbe(0 + 5*nfPost);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
