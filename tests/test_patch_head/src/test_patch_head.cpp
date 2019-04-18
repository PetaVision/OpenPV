#include "utils/PVLog.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <utils/conversions.h>

/*
 * The preferred patch size is even for a > 1 and odd for a <= 1
 */

// not used, zPatchHead called directly instead
int test_PatchHead(int kzPre, int nzPatch, int zScaleLog2Pre, int zScaleLog2Post) {
   int shift;

   float a = powf(2.0f, (float)(zScaleLog2Pre - zScaleLog2Post));

   if ((int)a == 1) {
      shift = -(int)(0.5f * (float)nzPatch);
      return shift + kzPre; // nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post);
   }

   shift = 1 - (int)(0.5f * (float)nzPatch);

   if (nzPatch % 2 == 0 && a < 1) {
      // density increases in post-synaptic layer

      // extra shift subtracted if kzPre is in right half of the
      // set of presynaptic indices that are between postsynaptic
      //

      int kpos = (kzPre < 0) ? -(1 + kzPre) : kzPre;
      int l    = (int)(2 * a * kpos) % 2;
      shift -= (kzPre < 0) ? l == 1 : l == 0;
   }
   else if (nzPatch % 2 == 1 && a < 1) {
      // density decreases in post-synaptic layer
      shift = -(int)(0.5f * (float)nzPatch);
      return shift + nearby_neighbor(kzPre, zScaleLog2Post - zScaleLog2Pre);
   }

   int neighbor = nearby_neighbor(kzPre, zScaleLog2Post - zScaleLog2Pre);

   // added if nzPatch == 1
   if (nzPatch == 1) {
      return neighbor;
   }

   return shift + neighbor;
}

/*
 *
 *
 */

int main(int argc, char *argv[]) {
   float a;
   int scaleLog2Pre, scaleLog2Post, ans;
   int kpre, kh, kBack, nPatch, test;

   // keep pre-synaptic scale fixed
   //

   scaleLog2Pre = 0;

   // common usage tests, nPatch odd, relative scale==0
   //

   scaleLog2Post = 0;

   nPatch = 1;
   test   = 1;
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch, scaleLog2Pre - scaleLog2Post) + nPatch - 1;
      if (kh != kpre - nPatch / 2 || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
   }

   nPatch = 7;
   test   = 2;
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch, scaleLog2Pre - scaleLog2Post) + nPatch - 1;
      if (kh != kpre - nPatch / 2 || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
   }

   nPatch = 27;
   test   = 3;
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch, scaleLog2Pre - scaleLog2Post) + nPatch - 1;
      if (kh != kpre - nPatch / 2 || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
   }

   // common usage tests, nPatch ODD, relative scale==2 (post less dense than pre, many to one)
   //
   int ainv;
   int kmod;

   a             = 0.5;
   ainv          = 2;
   scaleLog2Post = 1;
   nPatch        = 3;
   test          = 4;
   ans           = -5 - 1; // head starts at -6, increases every other kpre
   for (kpre = -9; kpre < 9; kpre++) {
      if (kpre >= 0) {
         kmod = kpre % ainv;
      }
      else {
         kmod = ainv - 1 - ((-1 - kpre) % ainv);
      }
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch * ainv, scaleLog2Pre - scaleLog2Post) + nPatch * ainv - ainv
              + kmod;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d kmod=%d\n",
               test,
               kpre,
               kh,
               kBack,
               kmod);
      }
      ans += (kmod == (ainv - 1));
   }

   a             = 0.5;
   ainv          = 2;
   scaleLog2Post = 1;
   nPatch        = 9;
   test          = 5;
   ans           = -5 - 4; // head starts at -9, increases every other kpre
   for (kpre = -9; kpre < 9; kpre++) {
      if (kpre >= 0) {
         kmod = kpre % ainv;
      }
      else {
         kmod = ainv - 1 - ((-1 - kpre) % ainv);
      }
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch * ainv, scaleLog2Pre - scaleLog2Post) + nPatch * ainv - ainv
              + kmod;
      if (kh != ans || kBack != kpre) {
         InfoLog().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d kmod=%d\n",
               test,
               kpre,
               kh,
               kBack,
               kmod);
      }
      ans += (kmod == (ainv - 1));
   }

   // common usage tests, nPatch ODD, relative scale==4 (less dense)
   //

   a             = 0.25;
   ainv          = 4;
   scaleLog2Post = 2;
   nPatch        = 3;
   test          = 6;
   ans           = -3 - 1; // head starts at -4, increases every fourth kpre
   for (kpre = -9; kpre < 9; kpre++) {
      if (kpre >= 0) {
         kmod = kpre % ainv;
      }
      else {
         kmod = ainv - 1 - ((-1 - kpre) % ainv);
      }
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch * ainv, scaleLog2Pre - scaleLog2Post) + nPatch * ainv - ainv
              + kmod;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d kmod=%d\n",
               test,
               kpre,
               kh,
               kBack,
               kmod);
      }
      ans += (kmod == (ainv - 1));
   }

   a             = 0.25;
   ainv          = 4;
   scaleLog2Post = 2;
   nPatch        = 9;
   test          = 7;
   ans           = -3 - 4; // head starts at -7, increases every fourth kpre
   for (kpre = -9; kpre < 9; kpre++) {
      if (kpre >= 0) {
         kmod = kpre % ainv;
      }
      else {
         kmod = ainv - 1 - ((-1 - kpre) % ainv);
      }
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch * ainv, scaleLog2Pre - scaleLog2Post) + nPatch * ainv - ainv
              + kmod;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d kmod=%d\n",
               test,
               kpre,
               kh,
               kBack,
               kmod);
      }
      ans += (kmod == (ainv - 1));
   }

   // common usage tests, nPatch even, relative scale==-1 (more dense)
   //

   a             = 2;
   scaleLog2Post = -1;

   nPatch = 2;
   test   = 8;
   ans    = -18; // head starts at -18, increases by 2
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch / a, scaleLog2Pre - scaleLog2Post) + nPatch / a - 2
              + (kpre % 2 == 0);
      kBack = kpre;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
      ans += 2;
   }

   a             = 2;
   scaleLog2Post = -1;

   nPatch = 4;
   test   = 9;
   ans    = -18 - 1; // head starts at -19, increases by 2
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch / a, scaleLog2Pre - scaleLog2Post) + nPatch / a - 2
              + (kpre % 2 == 0);
      kBack = kpre;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
      ans += 2;
   }

   a             = 2;
   scaleLog2Post = -1;

   nPatch = 8;
   test   = 10;
   ans    = -18 - 3; // head starts at -21, increases by 2
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch / a, scaleLog2Pre - scaleLog2Post) + nPatch / a - 2
              + (kpre % 2 == 0);
      kBack = kpre;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
      ans += 2;
   }

   // common usage tests, nPatch even, relative scale==-2 (more dense)
   //

   a             = 4;
   scaleLog2Post = -2;

   nPatch = 2;
   test   = 11;
   ans    = -35; // head starts at -35, increases by 4
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch / a, scaleLog2Pre - scaleLog2Post) + nPatch / a - 2
              + (kpre % 2 == 0);
      kBack = kpre;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
      ans += 4;
   }

   nPatch = 4;
   test   = 12;
   ans    = -35 - 1; // head starts at -36, increases by 4
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch / a, scaleLog2Pre - scaleLog2Post) + nPatch / a - 2
              + (kpre % 2 == 0);
      kBack = kpre;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
      ans += 4;
   }

   nPatch = 8;
   test   = 13;
   ans    = -35 - 3; // head starts at -38, increases by 4
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Post - scaleLog2Pre);
      kBack = zPatchHead(kh, nPatch / a, scaleLog2Pre - scaleLog2Post) + nPatch / a - 2
              + (kpre % 2 == 0);
      kBack = kpre;
      if (kh != ans || kBack != kpre) {
         Fatal().printf(
               "FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n",
               test,
               kpre,
               kh,
               kBack);
      }
      ans += 4;
   }

   return 0;
}
