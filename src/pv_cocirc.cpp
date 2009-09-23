/*
 * pv_cocirc.cpp
 *
 *  Runs a V1 layer with cocircular connections
 *
 */

#include <stdlib.h>

#include "include/pv_common.h"
#include "include/default_params.h"
#include "columns/HyPerCol.hpp"
#include "connections/GaborConn.hpp"
#include "connections/InhibConn.hpp"
#include "connections/CocircConn.hpp"
#include "connections/LongRangeConn.hpp"
#include "io/LinearActivityProbe.hpp"
#include "io/PointProbe.hpp"
#include "io/StatsProbe.hpp"

#include "layers/Retina.hpp"
#include "layers/LGN.hpp"
#include "layers/V1.hpp"

#undef HAS_MAIN

#ifdef HAS_MAIN
int pv_cocirc_main(int argc, char* argv[]);
int main(int argc, char* argv[])
{
   return pv_cocirc_main(argc, argv);
}
#endif

using namespace PV;

int pv_cocirc_main(int argc, char* argv[])
{
   // create the managing hypercolumn
   HyPerCol * hc = new HyPerCol("column", argc, argv);

   // create the layers
   HyPerLayer * retina   = new Retina("Retina", hc);
   HyPerLayer * lgn      = new V1("LGN", hc);
   HyPerLayer * lgnInh   = new V1("LGN Inh", hc);
   HyPerLayer * v1       = new V1("V1", hc);
   HyPerLayer * v1Inh    = new V1("V1 Inh", hc);

   HyPerLayer * v2       = new V1("V2", hc);
   HyPerLayer * v2Inh    = new V1("V2 Inh", hc);

   // connect the layers
   new HyPerConn("Retina to LGN",       hc, retina,     lgn,    CHANNEL_EXC);
   new HyPerConn("Retina to LGN Inh",   hc, retina,     lgnInh, CHANNEL_EXC);
   new HyPerConn("LGN Inh to LGN",      hc, lgnInh,     lgn,    CHANNEL_INH);
   new HyPerConn("LGN to V1",           hc, lgn,        v1,     CHANNEL_EXC);
   new HyPerConn("LGN to V1 Inh",       hc, lgn,        v1Inh,  CHANNEL_EXC);
   new CocircConn("V1 to V1",           hc, v1,         v1,     CHANNEL_EXC);
   new CocircConn("V1 to V1 Inh",       hc, v1,         v1Inh,  CHANNEL_EXC);
   new HyPerConn("V1 Inh to V1",        hc, v1Inh,      v1,     CHANNEL_INH);
   new HyPerConn("V1 to LGN",           hc, v1,         lgn,    CHANNEL_EXC);
   new HyPerConn("V1 to LGN Inh",       hc, v1,         lgnInh, CHANNEL_EXC);

   new CocircConn("V1 to V2",            hc, v1,         v2,     CHANNEL_EXC);
   new CocircConn("V1 to V2 Inh",        hc, v1,         v2Inh,  CHANNEL_EXC);
   new CocircConn("V2 to V2",            hc, v2,         v2,     CHANNEL_EXC);
   new CocircConn("V2 to V2 Inh",        hc, v2,         v2Inh,  CHANNEL_EXC);
   new CocircConn("V2 Inh to V2",        hc, v2Inh,      v2,     CHANNEL_INH);
   new CocircConn("V2 to V1",            hc, v2,         v1,     CHANNEL_EXC);
   new CocircConn("V2 to V1 Inh",        hc, v2,         v1Inh,  CHANNEL_EXC);

   //new LongRangeConn("V2 to V2 long range", hc, v2, v2, CHANNEL_EXC);

   // finish initialization now that everything is connected
   hc->initFinish();

   const int probeX = 0;
   const int probeY = 31;  // long line, {53,60} long/short dashed lines
   const int probeF = 0;

   PVLayerProbe * ptprobe = new PointProbe("V1Point.txt", probeX, probeY, probeF, "v1:");
   StatsProbe   * stats   = new StatsProbe("V1Stats.txt", BufActivity, "v1: a:");
   PVLayerProbe * probe1  = new LinearActivityProbe(hc, DimX, probeY, probeF);
   PVLayerProbe * probe2  = new LinearActivityProbe("V2Probe.txt", hc, DimX, probeY, probeF);
   v1->insertProbe(probe1);
   v2->insertProbe(probe2);

   // run the simulation
   hc->run();

   // clean up (HyPerCol owns the layers, so don't delete them here)
   delete hc;

   delete ptprobe;
   delete stats;
   delete probe1;
   delete probe2;

   return 0;
}
