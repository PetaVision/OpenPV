/*
 * pv.cpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  Main entry point for pv executable.
 *
 *  Currently, passes command-line args to HyPerCol
 *  and hard-codes parameter values for run from this file.
 */

#include <stdlib.h>

#include "columns/HyPerCol.hpp"
#include "io/LinearActivityProbe.hpp"
#include "layers/Retina.hpp"
#include "layers/V1.hpp"

#undef HAS_MAIN

#ifdef HAS_MAIN
int pv_main(int argc, char* argv[]);
int main(int argc, char* argv[])
{
   return pv_main(argc, argv);
}
#endif

int pv_main(int argc, char* argv[])
{
   // create the managing hypercolumn
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   // create the layers
   PV::HyPerLayer * retina  = new PV::Retina("Retina", hc);
   PV::HyPerLayer * lgn     = new PV::V1("LGN",        hc);
   PV::HyPerLayer * lgnInh  = new PV::V1("LGN Inh",    hc);
   PV::HyPerLayer * v1      = new PV::V1("V1",         hc);
   PV::HyPerLayer * v2      = new PV::V1("V2",         hc);

   // connect the layers
   new PV::HyPerConn("Retina to LGN",       hc, retina,     lgn,    CHANNEL_EXC);
   new PV::HyPerConn("Retina to LGN Inh",   hc, retina,     lgnInh, CHANNEL_EXC);
//   new PV::HyPerConn("LGN Inh to LGN",      hc, lgnInh,     lgn,    CHANNEL_INH);
//   new PV::HyPerConn("V1 to V2",            hc, v1,         v2,     CHANNEL_EXC);

   int locY = 60; // solid-31, dashed-53, 3-dashed-60;
   int f = 0;     // 0 on, 1 off cell

   // add probes
//   PV::PVLayerProbe * probeR = new PV::LinearActivityProbe("RetinaProbe.txt", hc, PV::DimX, locY, f);
   PV::PVLayerProbe * probe  = new PV::LinearActivityProbe(hc, PV::DimX, locY, f);

//   retina->insertProbe(probeR);
   lgn->insertProbe(probe);

   // run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers, don't delete them) */
   delete hc;

#ifdef TESTING

#define NF 8
#define NB NBORDER

   PVAxonalArborList arborList;

   PVPatch phi_patch;
   PVPatch w_patch;

   const int nxp = 2*NB;
   const int nyp = 2*NB;
   const int nf  = NF;

   float myPhi[nf*(NX+2*NB)*(NY+2*NB)];
   float myWeights[nf*nxp*nyp];

   PVAxonalArbor myArbor;
   PVAxonalArborList * tl = &arborList;
   PVAxonalArbor * tPtr   = &myArbor;

   const int sf  = 1;
   const int sx  = nf;
   const int sy  = sx*nxp;

   tl->numTasks = 1;
   tl->tasks = &tPtr;

   pvpatch_init(&w_patch,   nxp, nyp, nf, sf, sx, sy, myWeights);
   pvpatch_init(&phi_patch, NX+2*NB, NY+2*NB, nf, sf, sx, sx*(NX+2*NB), myPhi);

   tl->tasks[0]->weights = &w_patch;
   tl->tasks[0]->data    = &phi_patch;

   // unroll loop by 1
   PVAxonalArbor * arbor = tl->tasks[0];
   float* w   = arbor->weights->data;
   float* phi = arbor->data->data;

   float a = 1.0;

   // TODO - add y loop
   pvpatch_accumulate(nf*nxp, phi + 0*sy, a, w);
   pvpatch_accumulate(nf*nxp, phi + 1*sy, a, w);
   pvpatch_accumulate(nf*nxp, phi + 2*sy, a, w);
   pvpatch_accumulate(nf*nxp, phi + 3*sy, a, w);

   for (unsigned int i = 0; i < tl->numTasks; i++) {
      arbor = tl->tasks[0];
      w    = arbor->weights->data;
      phi  = arbor->data->data;

      // TODO - add y loop
      int j = 0;
      pvpatch_accumulate(nf*nxp, phi + i*sx + (j+0)*sy, a, w);
      pvpatch_accumulate(nf*nxp, phi + i*sx + (j+1)*sy, a, w);
      pvpatch_accumulate(nf*nxp, phi + i*sx + (j+2)*sy, a, w);
      pvpatch_accumulate(nf*nxp, phi + i*sx + (j+3)*sy, a, w);
    }
#endif

   return 0;
}

