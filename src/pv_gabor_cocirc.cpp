/*
 * pv_gabor_cocirc.cpp
 *
 *  Runs a V1 layer with cocircular connections using gabor filters as input
 *
 */

#include <stdlib.h>

#include "include/pv_common.h"
#include "include/default_params.h"
#include "columns/HyPerCol.hpp"
#include "connections/GaborConn.hpp"
#include "connections/InhibConn.hpp"
#include "connections/CocircConn.hpp"

#include "layers/Retina.hpp"
#include "layers/V1.hpp"

#undef HAS_MAIN

#ifdef HAS_MAIN
int pv_gabor_cocirc_main(int argc, char* argv[]);
int main(int argc, char* argv[])
{
   return pv_gabor_cocirc_main(argc, argv);
}
#endif

int pv_gabor_cocirc_main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina = new PV::Retina("Retina", hc);
   PV::HyPerLayer * v1     = new PV::V1("V1", hc);
   PV::HyPerLayer * v1Inh  = new PV::V1("V1 Inhibit", hc);

   // set layer functions
   // TODO - copyUpdate or fileread_update?
   retina->setFuncs((INIT_FN) &fileread_init, (UPDATE_FN) &pvlayer_copyUpdate);

   // connect the layers
   new PV::GaborConn("Retina to V1",  hc, retina, v1, CHANNEL_EXC);
   new PV::GaborConn("Retina to V1 Inhibit", hc, retina, v1Inh, CHANNEL_EXC);
   new PV::InhibConn("V1 Inhibit to V1", hc, v1Inh, v1, CHANNEL_INH);
   //   new PV::CocircConn("V1 to V1", hc, v1, v1, CHANNEL_EXC);

   // finish initialization now that everything is connected
   hc->initFinish();

   // TODO - decide where this should go
   hc->loadState();

   // run the simulation
   hc->run();

   hc->writeState();

   // clean up (HyPerCol owns the layers, so don't delete them here)
   delete hc;

   return 0;
}

