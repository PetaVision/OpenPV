/*
 * RuleConn.cpp
 *
 *  Created on: Apr 5, 2009
 *      Author: Craig Rasmussen
 */

#include "BiConn.hpp"
#include "src/io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

BiConn::BiConn(const char * name,
               HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel,
               int type)
{
   this->type = type;
   HyPerConn::initialize(name, hc, pre, post, channel);
}

int BiConn::outputState(FILE * fp, int kPre)
{
   return 0;

#ifdef DELETEME
   PVSynapseBundle * tasks = this->tasks(kPre, 0);
   PVPatch * pIncr = tasks->tasks[0]->plasticIncr;
   PVPatch * w     = tasks->tasks[0]->weights;
   size_t offset   = tasks->tasks[0]->offset;

   float * M = &pDecr->data[offset];  // STDP decrement variable

   fprintf(fp, "w%d:      M=", kPre);
   stdp_text_write_patch(fp, pIncr, M);
   fprintf(fp, "P=");
   stdp_text_write_patch(fp, pIncr, pIncr->data);  // write the P variable
   fprintf(fp, "w=");
   stdp_text_write_patch(fp, w, w->data);
   fprintf(fp, "\n");
   fflush(fp);
#endif

   return 0;
}

PVPatch ** BiConn::initializeWeights(PVPatch ** patches,
                                      int numPatches, const char * filename)
{
   PVParams * params = parent->parameters();
   int randomFlag = (int) params->value(getName(), "randomFlag", 0);

   if (filename == NULL && randomFlag == 0) {
      const float strength = params->value(name, "strength");

      const int arbor = 0;
      const int numPatches = numWeightPatches(arbor);
      assert(numPatches == pre->clayer->numNeurons);
      for (int i = 0; i < numPatches; i++) {
         int kPre = i;
         ruleWeights(wPatches[arbor][i], kPre, strength);
      }
      return wPatches[arbor];
   }

   fprintf(stderr, "Initializing weights not using rules RuleConn\n");
   return HyPerConn::initializeWeights(patches, numPatches, filename);
}

int BiConn::ruleWeights(PVPatch * wp, int kPre, float strength)
{
   pvdata_t * w = wp->data;

   const int nx = (int) pre->clayer->loc.nx;
   const int ny = (int) pre->clayer->loc.ny;
   const int nf = (int) pre->clayer->numFeatures;
   const int kx = (int) kxPos(kPre, nx, ny, nf);
   const int kfPre = featureIndex(kPre, nx, ny, nf);

   const int left = ! (kx % 2);

   const int nxp = (int) wp->nx;
   const int nyp = (int) wp->ny;
   const int nfp = (int) wp->nf;

   // strides
   const int sx = (int) wp->sx;  assert(sx == nfp);
   const int sy = (int) wp->sy;  assert(sy == nfp*nxp);
   const int sf = (int) wp->sf;  assert(sf == 1);

   assert(nyp == 1);

   // loop over all post synaptic neurons in patch

   // initialize connections to 0
   for (int f = 0; f < nfp; f++) {
      for (int j = 0; j < nyp; j++) {
         for (int i = 0; i < nxp; i++) {
            w[i*sx + j*sy + f*sf] = 0;
         }
      }
   }

   if (type == 11) {
      assert(nfp == 4);

      // L1 simple cells
      // f=0 : off(n) : OFF OFF ;
      // f=1 : on(n)  : ON  ON  ;
      // f=2 : ls     : OFF ON  ;
      // f=3 : rs     : ON  OFF ;

      // now set the actual pattern

      if (kfPre == 0) { // OFF cell
         if (left) {
            w[0] = 1;
            w[2] = 1;
         }
         else {
            w[0] = 1;
            w[3] = 1;
         }
      }
      else { // ON cell
         if (left) {
            w[1] = 1;
            w[3] = 1;
         }
         else {
            w[1] = 1;
            w[2] = 1;
         }
      }
   }
   // "L1 Simple to L1 Simple Inh"
   else if (type == 11110) {
      assert(nfp == 1);

      // L1 simple cells
      // f=0 : off(n) : OFF OFF ;
      // f=1 : on(n)  : ON  ON  ;
      // f=2 : ls     : OFF ON  ;
      // f=3 : rs     : ON  OFF ;

      // now set the actual pattern
      w[0] = 1;
   }
   // "L1 Simple Inh to L1 Simple"
   else if (type == 11011) {
      assert(nfp == 4);

      // L1 simple cells
      // f=0 : off(n) : OFF OFF ;
      // f=1 : on(n)  : ON  ON  ;
      // f=2 : ls     : OFF ON  ;
      // f=3 : rs     : ON  OFF ;

      // now set the actual pattern
      w[0] = 1;
      w[1] = 1;
      w[2] = 1;
      w[3] = 1;
   }
   else if (type == 1121) {
      assert(nfp == 7);

      // L2 simple cells
      // f=0 : off4  : off2 off2 ;
      // f=1 : on4   : on2  on2  ;
      // f=2 : ls_21 : off2 ls_1 ;
      // f=3 : ls_22 : ls_1 on2  ;
      // f=4 : rs_21 : on2  rs_1 ;
      // f=5 : rs_22 : rs_1 off2 ;
      // f=6 : ms_2  : ls_1 rs_1 ;

      // now set the actual pattern

      if (kfPre == 0) { // off2 cell
         if (left) {
            w[0] = 1;
            w[2] = 1;
         }
         else {
            w[0] = 1;
            w[5] = 1;
         }
      }
      else if (kfPre == 1) { // on2 cell
         if (left) {
            w[1] = 1;
            w[4] = 1;
         }
         else {
            w[1] = 1;
            w[3] = 1;
         }
      }
      else if (kfPre == 2) { // l1s cell
         if (left) {
            w[3] = 1;
            w[6] = 1;
         }
         else {
            w[2] = 1;
         }
      }
      else if (kfPre == 3) { // r1s cell
         if (left) {
            w[5] = 1;
         }
         else {
            w[4] = 1;
            w[6] = 1;
         }
      }
   }
   else if (type == 2122) {
      assert(nfp == 5);

      // complex cells

      // f=0 : offc_2 : offc_2
      // f=1 : on_c_2 : on_c_2
      // f=2 : lc_2   : ls_21 | ls_22 ;
      // f=3 : rc_2   : rs_21 | rs_22 ;
      // f=4 : mc_2   : ms_2

      // f=0 : off4  : off2 off2 ;
      // f=1 : on4   : on2  on2  ;
      // f=2 : ls_21 : off2 ls_1 ;
      // f=3 : ls_22 : ls_1 on2  ;
      // f=4 : rs_21 : on2  rs_1 ;
      // f=5 : rs_22 : rs_1 off2 ;

      // now set the actual pattern

      if (kfPre == 0) { // off_s_2 cell
         w[0] = 1;
      }
      else if (kfPre == 1) { // on_s_2 cell
         w[1] = 1;
      }
      else if (kfPre == 2) { // ls_21 cell
//         if (left) {
            w[2] = 1;
//         }
      }
      else if (kfPre == 3) { // ls_22 cell
//         if (!left) {
            w[2] = 1;
//         }
      }
      else if (kfPre == 4) { // rs_21 cell
//         if (left) {
            w[3] = 1;
//         }
      }
      else if (kfPre == 5) { // rs_22 cell
//         if (!left) {
            w[3] = 1;
//         }
      }
      else if (kfPre == 6) { // ms_2 cell
         w[4] = 1;
      }
   }
   else if (type == 2231) {
      assert(nfp == 7);

      // L3 simple cells
      // f=0 : off8  : off4 off4 ;
      // f=1 : on8   : on4  on4  ;
      // f=2 : ls_31 : off4 lc_2 ;
      // f=3 : ls_32 : lc_2 on4  ;
      // f=4 : rs_31 : on4  rc_2 ;
      // f=5 : rs_32 : rc_2 off4 ;
      // f=6 : ms_3  : lc_2 rc_2 ;

      // now set the actual pattern

      if (kfPre == 0) { // off_c_2 cell
         if (left) {
            w[0] = 1;
            w[2] = 1;
         }
         else {
            w[0] = 1;
            w[5] = 1;
         }
      }
      else if (kfPre == 1) { // on_c_2 cell
         if (left) {
            w[1] = 1;
            w[4] = 1;
         }
         else {
            w[1] = 1;
            w[3] = 1;
         }
      }
      else if (kfPre == 2) { // lc_2 cell
         if (left) {
            w[3] = 1;
            w[6] = 1;
         }
         else {
            w[2] = 1;
         }
      }
      else if (kfPre == 3) { // rc_2 cell
         if (left) {
            w[5] = 1;
         }
         else {
            w[4] = 1;
            w[6] = 1;
         }
      }
      else if (kfPre == 4) { // mc_2 cell
         // TODO - mc_s1 and mc_s2?
      }
   }
   else if (type == 3132) {
      assert(nfp == 5);

      // complex cells

      // f=0 : off4  : off4
      // f=1 : on4   : on4
      // f=2 : lc_2  : ls_21 | ls_22 ;
      // f=3 : rc_2  : rs_21 | rs_22 ;
      // f=4 : mc_2   : ms_2

      // f=0 : off4  : off2 off2 ;
      // f=1 : on4   : on2  on2  ;
      // f=2 : ls_21 : off2 ls_1 ;
      // f=3 : ls_22 : ls_1 on2  ;
      // f=4 : rs_21 : on2  rs_1 ;
      // f=5 : rs_22 : rs_1 off2 ;

      // now set the actual pattern

      if (kfPre == 0) { // off_s_2 cell
         w[0] = 1;
      }
      else if (kfPre == 1) { // on_s_2 cell
         w[1] = 1;
      }
      else if (kfPre == 2) { // ls_21 cell
//         if (left) {
            w[2] = 1;
//         }
      }
      else if (kfPre == 3) { // ls_22 cell
//         if (!left) {
            w[2] = 1;
//         }
      }
      else if (kfPre == 4) { // rs_21 cell
//         if (left) {
            w[3] = 1;
//         }
      }
      else if (kfPre == 5) { // rs_22 cell
//         if (!left) {
            w[3] = 1;
//         }
      }
      else if (kfPre == 6) { // m_s2 cell
         w[4] = 1;
      }
   }
   else if (type == 3241) {
      assert(nfp == 3);

      // L4 simple cells
      // f=0 : ms_41  : off_c_3 mc_3;
      // f=0 : ms_42  : mc_3 off_c_3;
      // f=0 : ms_43  : lc_3 rc_3 ;

      // now set the actual pattern

      if (kfPre == 0) { // off_c_3 cell
         if (left) {
            w[0] = 1;
         }
         else {
            w[1] = 1;
         }
      }
      else if (kfPre == 1) { // on_c_3 cell
         // TODO - error
      }
      else if (kfPre == 2) { // lc_3 cell
         if (left) {
            w[2] = 1;
         }
         else {
            // TODO - error
         }
      }
      else if (kfPre == 3) { // rc_3 cell
         if (left) {
            // TODO - error
         }
         else {
            w[2] = 1;
         }
      }
      else if (kfPre == 4) { // mc_3 cell
         if (left) {
            w[1] = 1;
         }
         else {
            w[0] = 1;
         }
      }
   }
   else if (type == 4142) {
      assert(nfp == 1);

      // L4 complex cells
      // f=0 : mc_4 : mc_41 | mc_42 | mc_43;

      // now set the actual pattern

      if (kfPre == 0) { // mc_41 cell
         w[0] = 1;
      }
      else if (kfPre == 1) { // mc_42 cell
         w[0] = 1;
      }
      else if (kfPre == 2) { // mc_43 cell
         w[0] = 1;
      }
   }
   else {
      assert(0);
   }

   for (int f = 0; f < nfp; f++) {
      float factor = strength;
      for (int i = 0; i < nxp*nyp; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

} // namespace PV
