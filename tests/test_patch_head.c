#include "../src/layers/PVLayer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float pvlayer_patchHead(float kxPre, float kxPost0Left, int xScale, float nxPatch);

/*
 * Only tests even X even patches (odd numbers may not be valid for the algorithm)
 * Not true anymore, odd patches work better (at least for scale >= 0)
 */

int main(int argc, char* argv[])
{
   int kh, scale;
   float kpre, k0l, nPatch;

   // common usage tests for nPatch odd, scale >=0 (feed-forward)

   scale = 0;
   nPatch = 3;
   k0l = 0;
   kpre = 0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 3;
   k0l = 0;
   kpre = 1;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 0) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 3;
   k0l = 0;
   kpre = 2;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 3;
   k0l = 1;
   kpre = 2;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 3;
   k0l = 0;
   kpre = 0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 3;
   k0l = 0;
   kpre = 1;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 3;
   k0l = 0;
   kpre = 2;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 0) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 3;
   k0l = 0;
   kpre = 4;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 3;
   k0l = -5;
   kpre = 7;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -3) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 3;
   k0l = 0;
   kpre = 0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 3;
   k0l = 0;
   kpre = 1;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 3;
   k0l = 0;
   kpre = 2;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 3;
   k0l = 0;
   kpre = 3;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 3;
   k0l = 0;
   kpre = 4;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 0) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // common usage tests for scale <= 0 feedback

   scale = 0;
   nPatch = 6;
   k0l = 0;
   kpre = 0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -3) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 6;
   k0l = 0;
   kpre = 1;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 6;
   k0l = 0;
   kpre = 2;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 6;
   k0l = 0;
   kpre = 3;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 1) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = -1;
   nPatch = 6;
   k0l = 0;
   kpre = 0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = -1;
   nPatch = 6;
   k0l = 0;
   kpre = 1;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 0) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = -1;
   nPatch = 6;
   k0l = 0;
   kpre = 2;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = -1;
   nPatch = 6;
   k0l = 0;
   kpre = 3;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 4) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 0;
   nPatch = 4;
   k0l = 0;
   kpre = 7;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 6) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = -1;
   nPatch = 8;
   k0l = -1;
   kpre = 6;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 8) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 8;
   k0l = -1;
   kpre = 5;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -2) {  // was -1
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 8;
   k0l = -1;
   kpre = 6;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != -2) {  // was -1
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 4;
   k0l = -1;
   kpre = 5;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 0) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 1;
   nPatch = 4;
   k0l = -1;
   kpre = 6;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 0) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   printf("Finshed with known results\n");

   scale = 2;
   nPatch = 2;
   k0l = -1;
   kpre = 10;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 2;
   k0l = -1;
   kpre = 11;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 2;
   k0l = -1;
   kpre = 12;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   scale = 2;
   nPatch = 2;
   k0l = -1;
   kpre = 13;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != 2) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // scale = 0, layers have equal number of neurons
   scale = 0;

   // one direction of 2x2 patch
   nPatch = 2;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -1;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 2;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 4x4 patch
   nPatch = 4;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l-1)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 3;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l-1)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -2;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l-1)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 8x8 patch
   nPatch = 8;

   k0l = 4;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -3;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != (kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 0;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != (kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // scale = -1, post layer has 2 times the number of neurons of the pre layer
   scale = -1;

   // one direction of 4x4 patch (minimum #)
   nPatch = 4;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-1)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -1;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-1)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 1;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-1)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 8x8 patch
   nPatch = 8;

   k0l = 1;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 0;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -5;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 7;
   kpre = 3.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(2*kpre+k0l-3)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // scale = 1, post layer has 1/2 the number of neurons of the pre layer
   scale = 1;

   // one direction of 2x2 patch
   nPatch = 2;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != (int)k0l) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -1;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre+1)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 1;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre+1)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 4x4 patch
   nPatch = 4;

   k0l = 1;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-1)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 0;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-1)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -5;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-1)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 7;
   kpre = 3.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-1)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 8x8 patch
   nPatch = 8;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-5)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -1;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-5)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 70;
   kpre = 3.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-5)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -20;
   kpre = 5000;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-5)/2)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // scale = 2, post layer has 1/4 the number of neurons of the pre layer
   scale = 2;

   // one direction of 2x2 patch
   nPatch = 2;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre+2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -1;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre+2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 1;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre+2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 4x4 patch
   nPatch = 4;

   k0l = 1;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 0;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -5;
   kpre = 2.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 7;
   kpre = 3.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-2)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   // one direction of 8x8 patch
   nPatch = 8;

   k0l = 0;
   kpre = 0.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-10)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -1;
   kpre = 1.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-10)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = 70;
   kpre = 3.0;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-10)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   k0l = -20;
   kpre = 5000;
   kh = pvlayer_patchHead(kpre, k0l, scale, nPatch);
   if (kh != floor(k0l+(kpre-10)/4)) {
      printf("FAILED:TEST_PATCH_HEAD: kh = %d\n", kh);
      exit(1);
   }

   return 0;
}
