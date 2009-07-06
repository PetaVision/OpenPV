/*
 * darpa.cpp
 *
 *  Files to make timing runs for Darpa proposal
 *    main: darpa.cpp (this file)
 *    input: input/params.darpa
 */

#include <stdlib.h>

#include "Retina1D.hpp"

#include "LinearPostConnProbe.hpp"

#include <src/columns/HyPerCol.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/clock.h>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/connections/RandomConn.hpp>

int darpa_main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer *retina, *l1, *l2, *l3, *l4, *l5, *l6, *l7, *l8, *l9;
   retina = new PV::Retina("Retina", hc);
   l1     = new PV::V1("L1", hc);
   l2     = new PV::V1("L2", hc);
   l3     = new PV::V1("L3", hc);
   l4     = new PV::V1("L4", hc);
   l5     = new PV::V1("L5", hc);
   l6     = new PV::V1("L6", hc);
   l7     = new PV::V1("L7", hc);
   l8     = new PV::V1("L8", hc);
   l9     = new PV::V1("L9", hc);
#ifdef NODO
#endif

   // connect the layers
   PV::HyPerConn *r_l1, *l1_l2, *l2_l3, *l3_l4, *l4_l5, *l5_l6, *l6_l7, *l7_l8, *l8_l9;
   r_l1  = new PV::HyPerConn("Retina to L1", hc, retina, l1, CHANNEL_EXC);
   l1_l2 = new PV::HyPerConn("L1 to L2", hc, l1, l2, CHANNEL_EXC);
   l2_l3 = new PV::HyPerConn("L2 to L3", hc, l2, l3, CHANNEL_EXC);
   l3_l4 = new PV::HyPerConn("L3 to L4", hc, l3, l4, CHANNEL_EXC);
   l4_l5 = new PV::HyPerConn("L4 to L5", hc, l4, l5, CHANNEL_EXC);
   l5_l6 = new PV::HyPerConn("L5 to L6", hc, l5, l6, CHANNEL_EXC);
   l6_l7 = new PV::HyPerConn("L6 to L7", hc, l6, l7, CHANNEL_EXC);
   l7_l8 = new PV::HyPerConn("L7 to L8", hc, l7, l8, CHANNEL_EXC);
   l8_l9 = new PV::HyPerConn("L8 to L9", hc, l8, l9, CHANNEL_EXC);
#ifdef NODO
#endif

   int locX = 31;
   int locY = 34;
   int locF = 0;

   // add probes
//   PV::PVLayerProbe * probe0  = new PV::LinearActivityProbe(hc, PV::DimX, locY, 0);

//   PV::PVLayerProbe * ptprobe0 = new PV::PointProbe(61, locY, 0, "L1:x=61 f=0");

//   PV::ConnectionProbe * cProbe0 = new PV::ConnectionProbe(locX, locY, locF);

//   PV::PostConnProbe * pcProbe0 = new PV::LinearPostConnProbe(PV::DimX, locY, 0);

//   l9->insertProbe(probe0);

   // run the simulation
   hc->initFinish();

   printf("Starting run...\n");
   hc->run();

   /* clean up (HyPerCol owns layers and connections, don't delete them) */
   delete hc;

   return 0;
}
