/*
 * pv_synch.cpp
 *
 *  Runs a one-dimensional V1 layer to try to get a line segment to synchronize
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
int pv_synch_main(int argc, char* argv[]);
int main(int argc, char* argv[])
{
   return pv_synch_main(argc, argv);
}
#endif

int pv_synch_main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina   = new PV::Retina("Retina", hc);
   PV::HyPerLayer * lgn      = new PV::V1("LGN", hc);
   PV::HyPerLayer * lgnInh   = new PV::V1("LGN Inh", hc);
   PV::HyPerLayer * v1       = new PV::V1("V1", hc);
   PV::HyPerLayer * v1Inh    = new PV::V1("V1 Inh", hc);

   // connect the layers
   new PV::HyPerConn("Retina to LGN",       hc, retina,     lgn,    CHANNEL_EXC);
   new PV::HyPerConn("Retina to LGN Inh",   hc, retina,     lgnInh, CHANNEL_EXC);
   new PV::HyPerConn("LGN Inh to LGN",      hc, lgnInh,     lgn,    CHANNEL_INH);
   new PV::GaborConn("LGN to V1",           hc, retina,     v1,     CHANNEL_EXC);
   new PV::GaborConn("LGN to V1 Inh",       hc, retina,     v1Inh,  CHANNEL_EXC);
   new PV::CocircConn("V1 to V1",           hc, v1,         v1,     CHANNEL_EXC);
   new PV::CocircConn("V1 to V1 Inh",       hc, v1,         v1Inh,  CHANNEL_EXC);
   new PV::GaborConn("V1 Inh to V1",        hc, v1Inh,      v1,     CHANNEL_INH);
//   new PV::LongRangeConn("V1 to V1 long range", hc, v1, v1, CHANNEL_EXC);

   // finish initialization now that everything is connected
   hc->initFinish();

   const int probeX = 0;
   const int probeY = 31;
   const int probeF = 0;

   PV::PVLayerProbe * ptprobe = new PV::PointProbe(probeX, probeY, probeF, "v1:");
   PV::StatsProbe   * stats   = new PV::StatsProbe(PV::BufActivity, "v1: a:");
   PV::PVLayerProbe * probe   = new PV::LinearActivityProbe(hc, PV::DimX, probeY, probeF);
   v1->insertProbe(probe);

   // run the simulation
   hc->run();

   // clean up (HyPerCol owns the layers, so don't delete them here)
   delete hc;

   delete ptprobe;
   delete stats;
   delete probe;

   return 0;
}

