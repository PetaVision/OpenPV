/*
 * pv_ca.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: rasmussn
 */

#include <stdlib.h>

#include "Retina1D.hpp"
#include "Retina2D.hpp"

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
   PV::HyPerLayer * retina = new PV::Retina2D("Retina", hc);
   PV::HyPerLayer * l1     = new PV::V1("L1", hc);
   PV::HyPerLayer * l2     = new PV::V1("L2", hc);
   PV::HyPerLayer * l3     = new PV::V1("L3", hc);
   PV::HyPerLayer * l4     = new PV::V1("L4", hc);
   PV::HyPerLayer * l5     = new PV::V1("L5", hc);

   PV::HyPerConn * r_l1 =
		   new PV::RandomConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   PV::HyPerConn * l1_l2 =
		   new PV::RandomConn("L1 to L2", hc, l1, l2, CHANNEL_EXC);
   PV::HyPerConn * l2_l3 =
		  new PV::RandomConn("L2 to L3", hc, l2, l3, CHANNEL_EXC);
   PV::HyPerConn * l3_l4 =
		   new PV::RandomConn("L3 to L4", hc, l3, l4, CHANNEL_EXC);
   PV::HyPerConn * l4_l5 =
		   new PV::RandomConn("L4 to L5", hc, l4, l5, CHANNEL_EXC);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
