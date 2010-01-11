/*
 * earth-stdp.cpp
 *
 *  Created on: Jan 7, 2010
 *      Author: rasmussn
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/connections/AvgConn.hpp>
#include <src/connections/RandomConn.hpp>
#include <src/layers/Gratings.hpp>
#include <src/layers/Movie.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/io/GLDisplay.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/StatsProbe.hpp>

#include <src/io/imageio.hpp>

using namespace PV;

int main(int argc, char* argv[])
{
   // create the managing hypercolumn
   //
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the image
   //
   Image * image = new Gratings("Image", hc);
   //Image * image = new Movie("Image", hc, "input/earth-files.txt", 4);

   // create a runtime data display
   //
   GLDisplay * display = new GLDisplay(&argc, argv, hc, 10);
   display->setImage(image);

   // create the layers
   //
   HyPerLayer * retina = new Retina("Retina", hc, image);
   //HyPerLayer * l1 = new V1("L1", hc);
   //HyPerLayer * av_retina = new V1("RetinaAvg", hc);

#ifdef INHIB
   HyPerLayer * l1Inh = new V1("L1Inh", hc);
#endif

   // create the connections
   //
   //HyPerConn * r_l1   = new HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   //HyPerConn * r_av   = new AvgConn("Retina to Average", hc, retina, av_retina, CHANNEL_EXC, NULL);
#ifdef INHIB
   HyPerConn * l1_inh = new HyPerConn("L1 to L1Inh", hc, l1, l1Inh, CHANNEL_EXC);
   HyPerConn * inh_l1 = new HyPerConn("L1Inh to L1", hc, l1Inh, l1, CHANNEL_INH);
#endif

   // add probes
   //

//   PVLayerProbe * statsR     = new StatsProbe(BufActivity, "R    :");
//   PVLayerProbe * statsL1    = new StatsProbe(BufActivity, "L1   :");
#ifdef INHIB
   PVLayerProbe * statsL1Inh = new StatsProbe(BufActivity, "L1Inh:");
#endif

//   retina->insertProbe(statsR);
//   l1->insertProbe(statsL1);
#ifdef INHIB
   l1Inh->insertProbe(statsL1Inh);
#endif

//   ConnectionProbe * cProbe = new ConnectionProbe(6, 6, 0);
//   r_av->insertProbe(cProbe);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
