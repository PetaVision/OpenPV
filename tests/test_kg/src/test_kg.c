#include <utils/conversions.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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

   //printf("size_loc==%ld size_cube==%ld size_ptr==%ld\n", sizeof(PVLayerLoc), sizeof(PVLayerCube), sizeof(pvdata_t*));
   //printf("size_int==%ld size_float==%ld, size_size_t==%ld\n", sizeof(int), sizeof(float), sizeof(size_t));

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
         printf("FAILED:TEST_KG: (kl,kg) = (%d,%d)\n", kl, kg);
         exit(1);
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

     kg = globalIndexFromLocal_nompi(kl, loc);
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

     kg = globalIndexFromLocal_nompi(kl, loc);
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
        printf("FAILED:TEST_KG: max ny (kl,kg) = (%d,%d)\n", kl, kg);
        exit(1);
     }
  }

  return 0;
}
