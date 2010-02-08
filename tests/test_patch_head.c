#include "../src/utils/conversions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * The preferred patch size is even for a != 1 and odd for a == 1
 */

int test_PatchHead(int kzPre, int nzPatch, int zScaleLog2Pre, int zScaleLog2Post)
{
   int shift;

   float a = powf(2.0f, (float) (zScaleLog2Pre - zScaleLog2Post));

   if ((int) a == 1) {
      shift = - (int) (0.5f * (float) nzPatch);
      return shift + nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post);
   }

   shift = 1 - (int) (0.5f * (float) nzPatch);

   if (nzPatch % 2 == 0 && a < 1) {
      // density increases in post-synaptic layer

      // extra shift subtracted if kzPre is in right half of the
      // set of presynaptic indices that are between postsynaptic
      //

      int kpos = (kzPre < 0) ? -(1+kzPre) : kzPre;
      int l = (int) (2*a*kpos) % 2;
      shift -= (kzPre < 0) ? l == 1 : l == 0;
   }

   int neighbor = nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post);
   return shift + neighbor;
}

/*
 * 
 * 
 */

int main(int argc, char* argv[])
{
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
   test = 1;
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre,  scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch, scaleLog2Post, scaleLog2Pre ) + nPatch - 1;
      if (kh != kpre-nPatch/2  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
   }

   nPatch = 7;
   test = 2;
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre,  scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch, scaleLog2Post, scaleLog2Pre ) + nPatch - 1;
      if (kh != kpre-nPatch/2  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
   }

   nPatch = 27;
   test = 3;
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre,  scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch, scaleLog2Post, scaleLog2Pre ) + nPatch - 1;
      if (kh != kpre-nPatch/2  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
   }

   // common usage tests, nPatch even, relative scale==1 (less dense)
   //

   a = 0.5;
   scaleLog2Post = 1;

   nPatch = 2;
   test = 4;
   ans = -5;   // head starts at -5, increases every other kpre
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d addi==%d\n", test, nPatch, kpre, kh, ans, kBack, (kpre+9)%2);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += (kpre + 9) % 2;
   }

   a = 0.5;
   scaleLog2Post = 1;

   nPatch = 8;
   test = 5;
   ans = -5 - 3;   // head starts at -8, increases every other kpre
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d addi==%d\n", test, nPatch, kpre, kh, ans, kBack, (kpre+9)%2);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += (kpre + 9) % 2;
   }

   // common usage tests, nPatch even, relative scale==2 (less dense)
   //

   a = 0.25;
   scaleLog2Post = 2;

   nPatch = 2;
   test = 6;
   ans = -3;   // head starts at -3, increases every fourth kpre
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch, scaleLog2Post, scaleLog2Pre ) + nPatch/a + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d addi=%d\n", test, nPatch, kpre, kh, ans, kBack, (kpre+9)%4==1);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += ((kpre + 9) % 4) == 2;
   }

   a = 0.25;
   scaleLog2Post = 2;

   nPatch = 8;
   test = 7;
   ans = -3 - 3;   // head starts at -6, increases every fourth kpre
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch, scaleLog2Post, scaleLog2Pre ) + nPatch + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += ((kpre + 9) % 4) == 2;
   }

   // common usage tests, nPatch even, relative scale==-1 (more dense)
   //

   a = 2;
   scaleLog2Post = -1;

   nPatch = 2;
   test = 7;
   ans = -18;   // head starts at -18, increases by 2
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += 2;
   }

   a = 2;
   scaleLog2Post = -1;

   nPatch = 4;
   test = 8;
   ans = -18 - 1;   // head starts at -19, increases by 2
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += 2;
   }

   a = 2;
   scaleLog2Post = -1;

   nPatch = 8;
   test = 9;
   ans = -18 - 3;   // head starts at -21, increases by 2
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += 2;
   }

   // common usage tests, nPatch even, relative scale==-2 (more dense)
   //

   a = 4;
   scaleLog2Post = -2;

   nPatch = 2;
   test = 10;
   ans = -35;   // head starts at -35, increases by 4
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += 4;
   }

   nPatch = 4;
   test = 10;
   ans = -35 - 1;   // head starts at -36, increases by 4
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += 4;
   }

   nPatch = 8;
   test = 10;
   ans = -35 - 3;   // head starts at -38, increases by 4
   for (kpre = -9; kpre < 9; kpre++) {
      kh    = zPatchHead(kpre, nPatch, scaleLog2Pre, scaleLog2Post);
      kBack = zPatchHead(kh  , nPatch/a, scaleLog2Post, scaleLog2Pre) + nPatch/a - 2 + (kpre%2 == 0);
      kBack = kpre;
      //printf("test==%d nPatch==%d kpre==%d kh==%d ans==%d kBack==%d\n", test, nPatch, kpre, kh, ans, kBack);
      if (kh != ans  ||  kBack != kpre) {
         printf("FAILED:TEST_PATCH_HEAD: test==%d kpre==%d kh==%d kBack==%d\n", test, kpre, kh, kBack);
         exit(1);
      }
      ans += 4;
   }

   return 0;
}
