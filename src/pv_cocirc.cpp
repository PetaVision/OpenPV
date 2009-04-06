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

int pv_cocirc_main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina   = new PV::Retina("Retina", hc);
   PV::HyPerLayer * lgn      = new PV::V1("LGN", hc);
   PV::HyPerLayer * lgnInh   = new PV::V1("LGN Inh", hc);
   PV::HyPerLayer * v1       = new PV::V1("V1", hc);
   PV::HyPerLayer * v1Inh    = new PV::V1("V1 Inh", hc);

   PV::HyPerLayer * v2       = new PV::V1("V2", hc);
   PV::HyPerLayer * v2Inh    = new PV::V1("V2 Inh", hc);

   // connect the layers
   new PV::HyPerConn("Retina to LGN",       hc, retina,     lgn,    CHANNEL_EXC);
   new PV::HyPerConn("Retina to LGN Inh",   hc, retina,     lgnInh, CHANNEL_EXC);
   new PV::HyPerConn("LGN Inh to LGN",      hc, lgnInh,     lgn,    CHANNEL_INH);
   new PV::HyPerConn("LGN to V1",           hc, lgn,        v1,     CHANNEL_EXC);
   new PV::HyPerConn("LGN to V1 Inh",       hc, lgn,        v1Inh,  CHANNEL_EXC);
   new PV::CocircConn("V1 to V1",           hc, v1,         v1,     CHANNEL_EXC);
   new PV::CocircConn("V1 to V1 Inh",       hc, v1,         v1Inh,  CHANNEL_EXC);
   new PV::HyPerConn("V1 Inh to V1",        hc, v1Inh,      v1,     CHANNEL_INH);
   new PV::HyPerConn("V1 to LGN",           hc, v1,         lgn,    CHANNEL_EXC);
   new PV::HyPerConn("V1 to LGN Inh",       hc, v1,         lgnInh, CHANNEL_EXC);

   new PV::CocircConn("V1 to V2",            hc, v1,         v2,     CHANNEL_EXC);
   new PV::CocircConn("V1 to V2 Inh",        hc, v1,         v2Inh,  CHANNEL_EXC);
   new PV::CocircConn("V2 to V2",            hc, v2,         v2,     CHANNEL_EXC);
   new PV::CocircConn("V2 to V2 Inh",        hc, v2,         v2Inh,  CHANNEL_EXC);
   new PV::CocircConn("V2 Inh to V2",        hc, v2Inh,      v2,     CHANNEL_INH);
   new PV::CocircConn("V2 to V1",            hc, v2,         v1,     CHANNEL_EXC);
   new PV::CocircConn("V2 to V1 Inh",        hc, v2,         v1Inh,  CHANNEL_EXC);

   //new PV::LongRangeConn("V2 to V2 long range", hc, v2, v2, CHANNEL_EXC);

   // finish initialization now that everything is connected
   hc->initFinish();

   const int probeX = 0;
   const int probeY = 31;  // long line, {53,60} long/short dashed lines
   const int probeF = 0;

   PV::PVLayerProbe * ptprobe = new PV::PointProbe("V1Point.txt", probeX, probeY, probeF, "v1:");
   PV::StatsProbe   * stats   = new PV::StatsProbe("V1Stats.txt", PV::BufActivity, "v1: a:");
   PV::PVLayerProbe * probe1  = new PV::LinearActivityProbe(hc, PV::DimX, probeY, probeF);
   PV::PVLayerProbe * probe2  = new PV::LinearActivityProbe("V2Probe.txt", hc, PV::DimX, probeY, probeF);
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
