/**
 * This file tests copying to boundary regions while applying mirror boundary conditions.
 *
 */

#undef DEBUG_PRINT

#include "../src/layers/HyPerLayer.hpp"
#include "../src/layers/Example.hpp"

const int numFeatures = 2;

static int check_corner(PVLayerCube* c, int nf, float start[]);
static int check_north_south(PVLayerCube* c, int nf, float start[]);

int main(int argc, char * argv[])
{
   float start[8];
   PVLayerLoc sLoc, bLoc;
   PVLayerCube * sCube, * bCube;

   PV::HyPerCol * hc = new PV::HyPerCol("column", argc, argv);
   PV::Example * l = new PV::Example("Test Borders", hc);

   int nf  = numFeatures;
   int nxB = 4;
   int nxS = 8;

   sLoc.nxGlobal = sLoc.nyGlobal = nxS + 2*nxB;  // add borders to global
   sLoc.kx0 = sLoc.ky0 = nxB;                    // shouldn't be used
   sLoc.nx = sLoc.ny = nxS;
   sLoc.nb = nxB;
   sLoc.nf = nf;

   bLoc = sLoc;
   bLoc.nx = bLoc.ny = nxB;

   sCube = pvcube_new(&sLoc, nxS*nxS*nf);
   bCube = pvcube_new(&bLoc, nxB*nxB*nf);

   for (int i = 0; i < nxS*nxS*nf; i++) {
     sCube->data[i] = i;
   }

#ifndef FEATURES_LAST
   start[0] = 54; start[1] = 38; start[2] = 22; start[3] = 6;
#else
   start[0] = 27; start[1] = 19; start[2] = 11; start[3] = 3;
#endif
   bCube->loc.nx = nxB; bCube->loc.ny = nxB;
   l->mirrorToNorthWest(bCube, sCube);
   check_corner(bCube, nf, start);

#ifndef FEATURES_LAST
   start[0] = 62; start[1] = 46; start[2] = 30; start[3] = 14;
#else
   start[0] = 31; start[1] = 23; start[2] = 15; start[3] = 7;
#endif
   bCube->loc.nx = nxB; bCube->loc.ny = nxB;
   l->mirrorToNorthEast(bCube, sCube);
   check_corner(bCube, nf, start);

#ifndef FEATURES_LAST
   start[0] = 118; start[1] = 102; start[2] = 86; start[3] = 70;
#else
   start[0] = 59; start[1] = 51; start[2] = 43; start[3] = 35;
#endif
   bCube->loc.nx = nxB; bCube->loc.ny = nxB;
   l->mirrorToSouthWest(bCube, sCube);
   check_corner(bCube, nf, start);

#ifndef FEATURES_LAST
   start[0] = 126; start[1] = 110; start[2] = 94; start[3] = 78;
#else
   start[0] = 63; start[1] = 55; start[2] = 47; start[3] = 39;
#endif
   bCube->loc.nx = nxB; bCube->loc.ny = nxB;
   l->mirrorToSouthEast(bCube, sCube);
   check_corner(bCube, nf, start);

   pvcube_delete(bCube);

   bCube->loc.nx = nxB; bCube->loc.ny = nxS;
   bCube = pvcube_new(&bLoc, nxB*nxS*nf);

#ifndef FEATURES_LAST
   start[0] = 6; start[1] = 22; start[2] = 38; start[3] = 54;
   start[4] = 70; start[5] = 86; start[6] = 102; start[7] = 118;
#else
   start[0] = 3; start[1] = 11; start[2] = 19; start[3] = 27;
   start[4] = 35; start[5] = 43; start[6] = 51; start[7] = 59;
#endif
   bCube->loc.nx = nxB; bCube->loc.ny = nxS;
   l->mirrorToWest(bCube, sCube);
   check_corner(bCube, nf, start);

#ifndef FEATURES_LAST
   start[0] = 14; start[1] = 30; start[2] = 46; start[3] = 62;
   start[4] = 78; start[5] = 94; start[6] = 110; start[7] = 126;
#else
   start[0] = 7; start[1] = 15; start[2] = 23; start[3] = 31;
   start[4] = 39; start[5] = 47; start[6] = 55; start[7] = 63;
#endif
   bCube->loc.nx = nxB; bCube->loc.ny = nxS;
   l->mirrorToEast(bCube, sCube);
   check_corner(bCube, nf, start);

#ifndef FEATURES_LAST
   start[0] = 48; start[1] = 32; start[2] = 16; start[3] = 0;
#else
   start[0] = 24; start[1] = 16; start[2] = 8; start[3] = 0;
#endif
   bCube->loc.nx = nxS; bCube->loc.ny = nxB;
   l->mirrorToNorth(bCube, sCube);
   check_north_south(bCube, nf, start);

#ifndef FEATURES_LAST
   start[0] = 112; start[1] = 96; start[2] = 80; start[3] = 64;
#else
   start[0] = 56; start[1] = 48; start[2] = 40; start[3] = 32;
#endif
   bCube->loc.nx = nxS; bCube->loc.ny = nxB;
   l->mirrorToSouth(bCube, sCube);
   check_north_south(bCube, nf, start);

#ifdef DEBUG_PRINT
   for (int f = 0; f < nf; f++) {
      for (int j = 0; j < nxB; j++) {
         int off = j*bCube->loc.nx + f*bCube->loc.nx*bCube->loc.ny;
         float * data = bCube->data + off;
         printf("%d:row %d = %f %f %f %f %f %f %f %f\n", f, j,
		data[0], data[1], data[2], data[3],
		data[4], data[5], data[6], data[7]);
      }
   }
#endif

   pvcube_delete(bCube);

   return 0;
}

static int check_north_south(PVLayerCube* c, int nf, float start[])
{
   int nx = c->loc.nx;
   int ny = c->loc.ny;

#ifndef FEATURES_LAST
   for (int j = 0; j < ny; j++) {
      float v = start[j];
      for (int i = 0; i < nx; i++) {
         float * data = c->data + i*nf + j*nf*nx;
         for (int f = 0; f < nf; f++) {
	    if (data[f] != v) {
               printf("ERROR: check_north_south %d:row %d = %f, should be %f\n", f, j, data[f], v);
               exit(1);
	    }
            v += 1;
	 }
      }
   }
#else
   for (int f = 0; f < nf; f++) {
      for (int j = 0; j < ny; j++) {
         float v = start[j] + f*64;
         float * data = c->data + j*nx + f*nx*ny;
	 for (int i = 0; i < nx; i++) {
	    if (data[i] != v) {
               printf("ERROR: check_north_south %d:row %d = %f, should be %f\n", f, j, data[i], v);
               exit(1);
	    }
            v += 1;
	 }
      }
   }
#endif

   return 0;
}

static int check_corner(PVLayerCube* c, int nf, float start[])
{
   int nx = c->loc.nx;
   int ny = c->loc.ny;

#ifndef FEATURES_LAST
   for (int j = 0; j < ny; j++) {
      float v = start[j];
      for (int i = 0; i < nx; i++) {
         float * data = c->data + i*nf + j*nf*nx;
         for (int f = 0; f < nf; f++) {
	    if (data[f] != v) {
               printf("ERROR: check_corner %d:row %d = %f, should be %f\n", i, j, data[f], v);
               exit(1);
	    }
            v = (f%2) ? v - 3 : v + 1;
	 }
      }
   }
#else
   for (int f = 0; f < nf; f++) {
      for (int j = 0; j < ny; j++) {
         float v = start[j] + f*64;
         float * data = c->data + j*nx + f*nx*ny;
	 for (int i = 0; i < nx; i++) {
	    if (data[i] != v) {
               printf("ERROR: check_corner %d:row %d = %f, should be %f\n", f, j, data[i], v);
               exit(1);
	    }
            v -= 1;
	 }
      }
   }
#endif

   return 0;
}
