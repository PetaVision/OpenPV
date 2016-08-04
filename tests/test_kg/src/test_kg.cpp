#include <utils/conversions.h>
#include <utils/PVLog.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "utils/PVLog.hpp"

//
// A replacement for globalIndexFromLocal from conversions.h.
// WARNING - any changes in conversions.h should be reflected here.
static inline int globalIndexFromLocal_nompi(int kl, PVLayerLoc loc)
{
   int kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, loc.nf);
   int kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, loc.nf);
   int  kf = featureIndex(kl, loc.nx, loc.ny, loc.nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
}


int main(int argc, char* argv[])
{
   PVLayerLoc loc;
   int kl, kg;
   int kx, ky, kf, kxg, kyg, kfg;
#ifdef FEATURES_LAST
   int ij;
#endif

   int nf = loc.nf = 3;
   int nx = loc.nx = 63;
   int ny = loc.ny = 127;

   loc.kx0 = 0;
   loc.ky0 = 0;

   loc.nxGlobal = nx;
   loc.nyGlobal = ny;

   for (kl = 0; kl < nx*ny*nf; kl++) {
      kg = globalIndexFromLocal_nompi(kl, loc);

      if (kg != kl) {
         pvError().printf("FAILED:TEST_KG: (kl,kg) = (%d,%d)\n", kl, kg);
      }
   }

  // divide in halve by x, take right

  nf = loc.nf = 2;
  nx = loc.nx = 32;
  ny = loc.ny = 128;

  loc.kx0 = 32;
  loc.ky0 = 0;

  loc.nxGlobal = 2.0*nx;
  loc.nyGlobal = ny;

#ifdef FEATURES_LAST
  for (kf = 0; kf < nf; kf++) {
     for (ij = 0; ij < nx*ny; ij++) {
        kl = ij + nx*ny*kf;
        kx = kxPos(kl, loc.nx, loc.ny, nf);
        ky = kyPos(kl, loc.nx, loc.ny, nf);

        kg  = globalIndexFromLocal_nompi(kl, loc);
        kxg = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
        kyg = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
        kfg = featureIndex(kg, loc.nxGlobal, loc.nyGlobal, nf);

        if ((kg-kl) != loc.kx0 + (loc.ky0 + kyg)*loc.nx + kf*nx*ny) {
           pvError().printf("FAILED:TEST_KG: right (kl,kg) = (%d,%d)\n", kl, kg);
        }
     }
  }
#else
  for (kl = 0; kl < nx*ny*nf; kl++) {
     kx = kxPos(kl, loc.nx, loc.ny, nf);
     ky = kyPos(kl, loc.nx, loc.ny, nf);
     kf = featureIndex(kl, loc.nx, loc.ny, nf);

     kg = globalIndexFromLocal_nompi(kl, loc);
     kxg = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kyg = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kfg = featureIndex(kg, loc.nxGlobal, loc.nyGlobal, nf);

     pvErrorIf(!(loc.kx0+kx == kxg), "Test failed.\n");
     pvErrorIf(!(ky == kyg), "Test failed.\n");
     pvErrorIf(!(kf == kfg), "Test failed.\n");

     if ((kg-kl) != loc.kx0*nf*(1+ky)) {
        pvError().printf("FAILED:TEST_KG: right (kl,kg) = (%d,%d)\n", kl, kg);
     }
  }
#endif

  // divide in halve by y, take bottom

  nf = loc.nf = 5;
  nx = loc.nx = 32;
  ny = loc.ny = 128;

  loc.kx0 = 0;
  loc.ky0 = 64;

  loc.nxGlobal = nx;
  loc.nyGlobal = 2.0*ny;

#ifdef FEATURES_LAST
  for (kf = 0; kf < nf; kf++) {
     for (ij = 0; ij < nx*ny; ij++) {
        int kl = ij + nx*ny*kf;
        int kx = kxPos(kl, loc.nx, loc.ny, nf);
        int ky = kyPos(kl, loc.nx, loc.ny, nf);
        kg = globalIndexFromLocal_nompi(kl, loc);
        kx = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
        ky = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);

        // kg = ky0*nxGlobal + kf*nxGlobal*nyGlobal
        // kl = kf*nx*ny
        if ((kg-kl) != nx*loc.ky0 + kf*nx*(loc.nyGlobal - ny)) {
           pvError().printf("FAILED:TEST_KG: bottom (kl,kg) = (%d,%d)\n", kl, kg);
        }
     }
  }
#else
  for (kl = 0; kl < nx*ny*nf; kl++) {
     kx = kxPos(kl, loc.nx, loc.ny, nf);
     ky = kyPos(kl, loc.nx, loc.ny, nf);
     kf = featureIndex(kl, loc.nx, loc.ny, nf);

     kg = globalIndexFromLocal_nompi(kl, loc);
     kxg = kxPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kyg = kyPos(kg, loc.nxGlobal, loc.nyGlobal, nf);
     kfg = featureIndex(kg, loc.nxGlobal, loc.nyGlobal, nf);

     pvErrorIf(!(loc.kx0+kx == kxg), "Test failed.\n");
     pvErrorIf(!(loc.ky0+ky == kyg), "Test failed.\n");
     pvErrorIf(!(kf == kfg), "Test failed.\n");

     if ((kg-kl) != loc.ky0*nf*nx) {
        pvError().printf("FAILED:TEST_KG: bottom (kl,kg) = (%d,%d)\n", kl, kg);
     }
  }
#endif

  nf = loc.nf = 1;
  nx = loc.nx = 4096;
  ny = loc.ny = 4096+1;  // this should fail (probably not now with ints)
  ny = loc.ny = 4096;

  loc.kx0 = 0;
  loc.ky0 = 0;

  loc.nxGlobal = nx;
  loc.nyGlobal = ny;

  for (kl = 0; kl < nx*ny*nf; kl++) {
     kg = globalIndexFromLocal_nompi(kl, loc);

     if (kg != kl) {
        pvError().printf("FAILED:TEST_KG: max ny (kl,kg) = (%d,%d)\n", kl, kg);
     }
  }

  return 0;
}
