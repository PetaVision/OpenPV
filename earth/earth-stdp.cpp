/*
 * earth-stdp.cpp
 *
 *  Created on: Jan 7, 2010
 *      Author: rasmussn
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/connections/AvgConn.hpp>
#include <src/connections/KernelConn.hpp>
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
   Image * image = new Movie("Image", hc, "input/grey-earth-files.txt", 10);

   // create a runtime data display
   //
#ifdef GL_DISPLAY
   GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);
   display->setImage(image);
#endif

   // create the layers
   //
   HyPerLayer * retinaOn  = new Retina("RetinaOn", hc);
   HyPerLayer * retinaOff = new Retina("RetinaOff", hc);
   //HyPerLayer * l1        = new V1("L1", hc);

#ifdef AVG_LAYER
   HyPerLayer * av_retina = new V1("RetinaAvg", hc);
   display->addLayer(retina);
   display->addLayer(av_retina);
#endif

#ifdef INHIB
   HyPerLayer * l1Inh = new V1("L1Inh", hc);
#endif

   // create the connections
   //
   HyPerConn * i_r1_c  = new KernelConn("Image to RetinaOn Center",   hc, image, retinaOn, CHANNEL_EXC);
   HyPerConn * i_r1_s  = new KernelConn("Image to RetinaOn Surround", hc, image, retinaOn, CHANNEL_INH);
   HyPerConn * i_r0_c  = new KernelConn("Image to RetinaOff Center", hc, image, retinaOff, CHANNEL_INH);
   HyPerConn * i_r0_s  = new KernelConn("Image to RetinaOff Surround", hc, image, retinaOff, CHANNEL_EXC);
   //HyPerConn * r1_l1   = new HyPerConn("RetinaOn to L1", hc, retinaOn, l1, CHANNEL_EXC);
   //HyPerConn * r0_l1   = new HyPerConn("RetinaOff to L1", hc, retinaOff, l1, CHANNEL_EXC);

#ifdef INHIB
   HyPerConn * l1_inh = new HyPerConn("L1 to L1Inh", hc, l1, l1Inh, CHANNEL_EXC);
   HyPerConn * inh_l1 = new HyPerConn("L1Inh to L1", hc, l1Inh, l1, CHANNEL_INH);
#endif

   // add probes
   //

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
