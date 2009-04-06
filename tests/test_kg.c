#include "../src/layers/elementals.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
   PVLayerLoc loc;
   int kl, kg;
   int kx, ky, kf, kxg, kyg, kfg;
#ifdef FEATURES_LAST
   int ij;
#endif

   int nf = 3;

   float nx = loc.nx = 63.0;
   float ny = loc.ny = 127.0;

   loc.kx0 = 0.0;
   loc.ky0 = 0.0;

   loc.nxGlobal = nx;
   loc.nyGlobal = ny;

   for (kl = 0; kl < nx*ny*nf; kl++) {
      kg = globalIndexFromLocal(kl, loc, nf);

      if (kg != kl) {
         printf("FAILED:TEST_KG: (kl,kg) = (%d,%d)\n", kl, kg);
         exit(1);
      }
   }

  // divide in halve by x, take right

  nf = 2;
  nx = loc.nx = 32.0;
  ny = loc.ny = 128.0;

  loc.kx0 = 32.0;
  loc.ky0 = 0.0;

  loc.nxGlobal = 2.0*nx;
  loc.nyGlobal = ny;

#ifdef FEATURES_LAST
  for (kf = 0; kf < nf; kf++) {
     for (ij = 0; ij < nx*ny; ij++) {
        kl = ij + nx*ny*kf;
        kx = kxPos(kl, loc.nx, loc.ny, nf);
        ky = kyPos(kl, loc.nx, loc.ny, nf);

        kg  = globalIndexFromLocal(kl, loc, nf);
        kxg = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
        kyg = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
        kfg = featureIndex(kg, loc.nxGlobal, loc.nyGlobal, nf);

        if ((kg-kl) != loc.kx0 + (loc.ky0 + kyg)*loc.nx + kf*nx*ny) {
           printf("FAILED:TEST_KG: right (kl,kg) = (%d,%d)\n", kl, kg);
           exit(1);
        }
     }
  }
#else
  for (kl = 0; kl < nx*ny*nf; kl++) {
     kx = kxPos(kl, loc.nx, loc.ny, nf);
     ky = kyPos(kl, loc.nx, loc.ny, nf);
     kf = featureIndex(kl, loc.nx, loc.ny, nf);

     kg = globalIndexFromLocal(kl, loc, nf);
     kxg = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kyg = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kfg = featureIndex(kg, loc.nxGlobal, loc.nyGlobal, nf);

     assert(loc.kx0+kx == kxg);
     assert(ky == kyg);
     assert(kf == kfg);

     if ((kg-kl) != loc.kx0*nf*(1+ky)) {
        printf("FAILED:TEST_KG: right (kl,kg) = (%d,%d)\n", kl, kg);
        exit(1);
     }
  }
#endif

  // divide in halve by y, take bottom

  nf = 5;
  nx = loc.nx = 32.0;
  ny = loc.ny = 128.0;

  loc.kx0 = 0.0;
  loc.ky0 = 64.0;

  loc.nxGlobal = nx;
  loc.nyGlobal = 2.0*ny;

#ifdef FEATURES_LAST
  for (kf = 0; kf < nf; kf++) {
     for (ij = 0; ij < nx*ny; ij++) {
        int kl = ij + nx*ny*kf;
        int kx = kxPos(kl, loc.nx, loc.ny, nf);
        int ky = kyPos(kl, loc.nx, loc.ny, nf);
        kg = globalIndexFromLocal(kl, loc, nf);
        kx = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
        ky = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);

        // kg = ky0*nxGlobal + kf*nxGlobal*nyGlobal
        // kl = kf*nx*ny
        if ((kg-kl) != nx*loc.ky0 + kf*nx*(loc.nyGlobal - ny)) {
           printf("FAILED:TEST_KG: bottom (kl,kg) = (%d,%d)\n", kl, kg);
           exit(1);
        }
     }
  }
#else
  for (kl = 0; kl < nx*ny*nf; kl++) {
     kx = kxPos(kl, loc.nx, loc.ny, nf);
     ky = kyPos(kl, loc.nx, loc.ny, nf);
     kf = featureIndex(kl, loc.nx, loc.ny, nf);

     kg = globalIndexFromLocal(kl, loc, nf);
     kxg = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kyg = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kfg = featureIndex(kg, loc.nxGlobal, loc.nyGlobal, nf);

     assert(loc.kx0+kx == kxg);
     assert(loc.ky0+ky == kyg);
     assert(kf == kfg);

     if ((kg-kl) != loc.ky0*nf*nx) {
        printf("FAILED:TEST_KG: bottom (kl,kg) = (%d,%d)\n", kl, kg);
        exit(1);
     }
  }
#endif

  nf = 1;
  nx = loc.nx = 4096.0;
  ny = loc.ny = 4096.0+1.0;  // this should fail
  ny = loc.ny = 4096.0;

  loc.kx0 = 0.0;
  loc.ky0 = 0.0;

  loc.nxGlobal = nx;
  loc.nyGlobal = ny;

  for (kl = 0; kl < nx*ny*nf; kl++) {
     kg = globalIndexFromLocal(kl, loc, nf);

     if (kg != kl) {
        printf("FAILED:TEST_KG: max ny (kl,kg) = (%d,%d)\n", kl, kg);
        exit(1);
     }
  }

  return 0;
}
