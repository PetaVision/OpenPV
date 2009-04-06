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

#include "include/pv_common.h"
#include "include/default_params.h"
#include "columns/HyPerCol.hpp"

#include "layers/Retina.hpp"
#include "layers/V1.hpp"

// layer + connection handlers/callback headers
#include "layers/LIF.h"
#include "layers/LIF2.h"
#include "layers/thru.h"
#include "layers/gauss2D.h"
#include "layers/gauss2Dx.h"
#include "layers/cocirc1D.h"
#include "layers/fileread.h"
#include "layers/prob_fire.h"

#ifdef HAS_MAIN
int pv_main(int argc, char* argv[]);
int main(int argc, char* argv[])
{
   return pv_main(argc, argv);
}
#endif

#define paramLen(p) (sizeof(p)/sizeof(*p))

const char inputFilename[] = "src/io/input/amoeba2X.bin";

float V1SimpleNeuronParams[] =
{
    V_REST, V_EXC, V_INH, V_INHB,                       // V (mV)
    4*TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,                              // tau (ms)
    0.25, NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    0.25, NOISE_AMP*1.0,
    0.25, NOISE_AMP*1.0   // noise (G)
};

float V1FlankInhibNeuronParams[] =
{
    V_REST, V_EXC, V_INH, V_INHB,                       // V (mV)
    1*TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,                              // tau (ms)
    0.25, 1*NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + 1*TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    0.25, 1*NOISE_AMP*1.0,
    0.25, NOISE_AMP*1.0   // noise (G)
};

float V1SurroundInhibNeuronParams[] =
{
    V_REST, V_EXC, V_INH, V_INHB,                       // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,                              // tau (ms)
    0.25, 1*NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + 1*TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    0.25, 1*NOISE_AMP*1.0,
    0.25, NOISE_AMP*1.0   // noise (G)
};

float RetinaToV1SimpleSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    5.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float RetinaToV1FlankInhibSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    5.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float RetinaToV1SurroundInhibSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    1.0, //G_ASPECT (1/aspect ratio)
    5.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float V1FlankInhibToV1SimpleLeftSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    1.0, //G_OFFSET
    1.0 * 1.0 * DTH * DTH, //G_SIGMA_THETA2
    1.0 * DTH, // G_DTH_MAX
    1.0, //G_ASYM_FLAG
    1.0, //G_SELF_FLAG
    10.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float V1FlankInhibToV1SimpleRightSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    1.0, //G_OFFSET
    1.0 * 1.0 * DTH * DTH, //G_SIGMA_THETA2
    1.0 * DTH, // G_DTH_MAX
    -1.0, //G_ASYM_FLAG
    1.0, //G_SELF_FLAG
    10.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float V1FlankInhibToV1FlankInhibLeftSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    1.0, //G_OFFSET
    1.0 * 1.0 * DTH * DTH, //G_SIGMA_THETA2
    1.0 * DTH, // G_DTH_MAX
    1.0, //G_ASYM_FLAG
    0.0, //G_SELF_FLAG
    10.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float V1FlankInhibToV1FlankInhibRightSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    1.0, //G_OFFSET
    1.0 * 1.0 * DTH * DTH, //G_SIGMA_THETA2
    1.0 * DTH, // G_DTH_MAX
    -1.0, //G_ASYM_FLAG
    0.0, //G_SELF_FLAG
    10.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};

float V1FlankInhibGapSynapseParams[] =
{
    1.0, //G_usePBC: periodic boundary conditions
    2*RMAX_EDGE*RMAX_EDGE,//G_R2: neighborhood radius
    2 * SIGMA_EDGE * SIGMA_EDGE, //G_SIGMA2
    ASPECT_RATIO, //G_ASPECT (1/aspect ratio)
    0.0, //G_OFFSET
    1.0 * 1.0 * DTH * DTH, //G_SIGMA_THETA2
    1.0 * DTH, // G_DTH_MAX
    0.0, //G_ASYM_FLAG
    0.0, //G_SELF_FLAG
    0.0 * GLOBAL_GAIN //GAUSS_WEIGHT_SCALE
};


float V1SimpleToV1SimpleSynapseParams[] =
{
    1.0, // CCK_usePBC
    2 * RMAX_COCIRC * RMAX_COCIRC, //EXCITE_R2
    2 * SIGMA_DIST_COCIRC * SIGMA_DIST_COCIRC, // COCIRC_SIGMA_DIST2
    2.0 * DK * DK, // COCIRC_SIGMA_KURVE2
    2.0 * DTH * DTH, // COCIRC_SIGMA_COCIRC2
    0.0, // COCIRC_SELF
    50.0 * GLOBAL_GAIN // COCIRC_WEIGHT_SCALE
};


float V1SimpleToV1SimpleInhSynapseParams[] =
{
    1.0, // CCK_usePBC
    2 * RMAX_COCIRC * RMAX_COCIRC, //EXCITE_R2
    2 * SIGMA_DIST_COCIRC * SIGMA_DIST_COCIRC, // COCIRC_SIGMA_DIST2
    1.0 * 2.0 * DK * DK, // COCIRC_SIGMA_KURVE2
    -4.0 * 2.0 * DTH * DTH, // COCIRC_SIGMA_COCIRC2
    1.0, // COCIRC_SELF
    1000.0 * GLOBAL_GAIN // COCIRC_WEIGHT_SCALE
};


float V1SimpleToV1FlankInhibSynapseParams[] =
{
    1.0, // CCK_usePBC
    2 * RMAX_COCIRC * RMAX_COCIRC, //EXCITE_R2
    2 * SIGMA_DIST_COCIRC * SIGMA_DIST_COCIRC, // COCIRC_SIGMA_DIST2
    2.0 * DK * DK, // COCIRC_SIGMA_KURVE2
    2.0 * DTH * DTH, // COCIRC_SIGMA_COCIRC2
    1.0, // COCIRC_SELF
    0.0 * GLOBAL_GAIN // COCIRC_WEIGHT_SCALE
};

// NOTE - change function name to main to build pv application
int pv_main(int argc, char* argv[])
{
   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);

   PV::HyPerLayer* Retina = new PV::Retina("Retina", hc, inputFilename);
   PV::HyPerLayer* V1Simple = new PV::V1("V1Simple", hc, TypeV1Simple);
   PV::HyPerLayer* V1FlankInhib = new PV::V1("V1FlankInhib", hc, TypeV1FlankInhib);
//   PV::HyPerLayer* V1SurroundInhib = new PV::V1("V1SurroundInhib", hc, TypeV1SurroundInhib);

   V1Simple    ->setParams(paramLen(V1SimpleNeuronParams),     sizeof(V1SimpleNeuronParams), V1SimpleNeuronParams);
   V1FlankInhib->setParams(paramLen(V1FlankInhibNeuronParams), sizeof(V1FlankInhibNeuronParams), V1FlankInhibNeuronParams);
//   V1SurroundInhib->setParams(PLEN(V1SurroundInhibNeuronParams), sizeof(V1SurroundInhibNeuronParams), V1SurroundInhibNeuronParams);

   // connect the layers

/*****
   // retina to cortex
   hc->addCConnection(Retina, V1Simple, EXCITE_DELAY, VEL, RMIN, RMAX_EDGE, PHI_EXC,
                      paramLen(RetinaToV1SimpleSynapseParams),
                      RetinaToV1SimpleSynapseParams, (RECV_FN) &gauss2D_rcv,
                      (INIT_FN) &gauss2D_init);
   hc->addCConnection(Retina, V1FlankInhib, EXCITE_DELAY, VEL, RMIN, RMAX_EDGE, PHI_EXC,
                      paramLen(RetinaToV1FlankInhibSynapseParams),
                      RetinaToV1FlankInhibSynapseParams, (RECV_FN) &gauss2D_rcv,
                      (INIT_FN) &gauss2D_init);
//     hc->addCConnection(Retina, V1SurroundInhib, EXCITE_DELAY, VEL, RMIN, RMAX_EDGE, PHI_EXC,
//                     paramLen(RetinaToV1SurroundInhibSynapseParams),
//                     RetinaToV1SurroundInhibSynapseParams, (RECV_FN) &gauss2D_rcv,
//                     (INIT_FN) &gauss2D_init);


   // flanking inhibition
   hc->addCConnection(V1FlankInhib, V1Simple, INHIB_DELAY, VEL, RMIN, RMAX, PHI_INH,
                      paramLen(V1FlankInhibToV1SimpleLeftSynapseParams),
                      V1FlankInhibToV1SimpleLeftSynapseParams,
                      (RECV_FN) &gauss2Dx_rcv, (INIT_FN) &gauss2Dx_init);
   hc->addCConnection(V1FlankInhib, V1Simple, INHIB_DELAY, VEL, RMIN, RMAX_EDGE, PHI_INH,
                      paramLen(V1FlankInhibToV1SimpleRightSynapseParams),
                      V1FlankInhibToV1SimpleRightSynapseParams,
                      (RECV_FN) &gauss2Dx_rcv, (INIT_FN) &gauss2Dx_init);
   hc->addCConnection(V1FlankInhib, V1FlankInhib, INHIB_DELAY, VEL, RMIN, RMAX, PHI_INH,
                      paramLen(V1FlankInhibToV1FlankInhibLeftSynapseParams),
                      V1FlankInhibToV1FlankInhibLeftSynapseParams,
                      (RECV_FN) &gauss2Dx_rcv, (INIT_FN) &gauss2Dx_init);
   hc->addCConnection(V1FlankInhib, V1FlankInhib, INHIB_DELAY, VEL, RMIN, RMAX_EDGE, PHI_INH,
                      paramLen(V1FlankInhibToV1FlankInhibRightSynapseParams),
                      V1FlankInhibToV1FlankInhibRightSynapseParams,
                      (RECV_FN) &gauss2Dx_rcv, (INIT_FN) &gauss2Dx_init);
//     hc->addCConnection(V1FlankInhib, V1Simple, INHIB_DELAY, VEL, RMIN, RMAX_EDGE, PHI_INH,
//                     paramLen(V1FlankInhibToV1SimpleSynapseParams),
//                     V1FlankInhibToV1SimpleSynapseParams,
//                     (RECV_FN) &gauss2Dx_rcv, (INIT_FN) &gauss2Dx_init);

    // Gap Junctions
//     hc->addCConnection(V1FlankInhib, V1FlankInhib, 0, VEL, RMIN, RMAX_EDGE, PHI_EXC,
//                     paramLen(V1FlankInhibGapSynapseParams),
//                     V1FlankInhibGapSynapseParams, (RECV_FN) &gauss2Dx_rcv, (INIT_FN) &gauss2Dx_init);


   // cocircular connections
   hc->addCConnection(V1Simple, V1Simple, EXCITE_DELAY, EXCITE_VEL, RMIN, RMAX_COCIRC, PHI_EXC,
                      paramLen(V1SimpleToV1SimpleSynapseParams),
                      V1SimpleToV1SimpleSynapseParams, (RECV_FN) &cocirc1D_rcv, (INIT_FN) &cocirc1D_init);
   hc->addCConnection(V1Simple, V1Simple, INHIB_DELAY, INHIB_VEL, RMIN, RMAX_COCIRC, PHI_INH,
                      paramLen(V1SimpleToV1SimpleInhSynapseParams),
                      V1SimpleToV1SimpleInhSynapseParams, (RECV_FN) &cocirc1D_rcv, (INIT_FN) &cocirc1D_init);

//     hc->addCConnection(V1Simple, V1FlankInhib, EXCITE_DELAY, EXCITE_VEL, RMIN, RMAX_COCIRC, PHI_EXC,
//                     paramLen(V1SimpleToV1FlankInhibSynapseParams),
//                     V1SimpleToV1FlankInhibSynapseParams, (RECV_FN) &cocirc1D_rcv, (INIT_FN) &cocirc1D_init);
*****/

   // Run the simulation
   hc->initFinish();
   hc->run();

   /* clean up (HyPerCol owns layers, don't delete them) */
   delete hc;

#ifdef TESTING

#define NF 8
#define NB NBORDER

   PVSynapseBundle taskList;

   PVPatch phi_patch;
   PVPatch w_patch;

   const int nxp = 2*NB;
   const int nyp = 2*NB;
   const int nf  = NF;

   float myPhi[nf*(NX+2*NB)*(NY+2*NB)];
   float myWeights[nf*nxp*nyp];

   PVSynapseTask myTask;
   PVSynapseBundle * tl = &taskList;
   PVSynapseTask * tPtr = &myTask;

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
   PVSynapseTask * task = tl->tasks[0];
   float* w   = task->weights->data;
   float* phi = task->data->data;

   float a = 1.0;

   // TODO - add y loop
   pvpatch_accumulate(nf*nxp, phi + 0*sy, a, w);
   pvpatch_accumulate(nf*nxp, phi + 1*sy, a, w);
   pvpatch_accumulate(nf*nxp, phi + 2*sy, a, w);
   pvpatch_accumulate(nf*nxp, phi + 3*sy, a, w);

   for (unsigned int i = 0; i < tl->numTasks; i++) {
      task = tl->tasks[0];
      w    = task->weights->data;
      phi  = task->data->data;

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

