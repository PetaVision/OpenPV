/*
 * PostWeightsTest.cpp
 */

#include <components/PostWeights.hpp>
#include <utils/PVLog.hpp>

#include <string.h>

// This test checks whether the PostWeights class instantiates a Weights object
// with the correct patch geometry.

void testPostWeights(
      PV::Weights &postWeights,
      int correctNumDataPatchesX,
      int correctNumDataPatchesY,
      int correctNumDataPatchesF,
      int correctPatchSizeX,
      int correctPatchSizeY,
      int correctPatchSizeF) {
   FatalIf(
         postWeights.getNumDataPatchesX() != correctNumDataPatchesX,
         "%s: number of data patches in x-direction was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getNumDataPatchesX(),
         correctNumDataPatchesX);
   FatalIf(
         postWeights.getNumDataPatchesY() != correctNumDataPatchesY,
         "%s: number of data patches in y-direction was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getNumDataPatchesY(),
         correctNumDataPatchesY);
   FatalIf(
         postWeights.getNumDataPatchesF() != correctNumDataPatchesF,
         "%s: number of data patches in feature direction was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getNumDataPatchesX(),
         correctNumDataPatchesF);

   int correctNumDataPatches =
         correctNumDataPatchesX * correctNumDataPatchesY * correctNumDataPatchesF;
   FatalIf(
         postWeights.getNumDataPatches() != correctNumDataPatches,
         "%s: number of data patches overall was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getNumDataPatchesX(),
         correctNumDataPatchesF);

   FatalIf(
         postWeights.getPatchSizeX() != correctPatchSizeX,
         "%s: patch size in x-direction was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getPatchSizeX(),
         correctPatchSizeX);
   FatalIf(
         postWeights.getPatchSizeY() != correctPatchSizeY,
         "%s: patch size in x-direction was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getPatchSizeY(),
         correctPatchSizeY);
   FatalIf(
         postWeights.getPatchSizeF() != correctPatchSizeF,
         "%s: patch size in x-direction was %d instead of the expected %d\n",
         postWeights.getName().c_str(),
         postWeights.getPatchSizeF(),
         correctPatchSizeF);
}

void testOneToOneShared() {
   std::string name("One-to-one, shared weights");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 8;
   preLoc.ny       = 8;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 2;
   preLoc.halo.rt  = 2;
   preLoc.halo.dn  = 2;
   preLoc.halo.up  = 2;
   postLoc.nx      = 8;
   postLoc.ny      = 8;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 5;
   int nyp = 5;
   int nfp = 10;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, true, 0.0);
   PV::PostWeights postWeights(name, &weightsObject);

   testPostWeights(postWeights, 1, 1, postLoc.nf, nxp, nyp, preLoc.nf);
}

void testOneToOneNonshared() {
   std::string name("One-to-one, nonshared weights");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 8;
   preLoc.ny       = 8;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 2;
   preLoc.halo.rt  = 2;
   preLoc.halo.dn  = 2;
   preLoc.halo.up  = 2;
   postLoc.nx      = 8;
   postLoc.ny      = 8;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 5;
   int nyp = 5;
   int nfp = 10;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, false, 0.0);
   PV::PostWeights postWeights(name, &weightsObject);

   int nxExt = postLoc.nx + postLoc.halo.lt + postLoc.halo.rt;
   int nyExt = postLoc.ny + postLoc.halo.dn + postLoc.halo.up;
   testPostWeights(postWeights, nxExt, nyExt, postLoc.nf, nxp, nyp, preLoc.nf);
}

void testOneToManyShared() {
   std::string name("One-to-many, shared weights");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 4;
   preLoc.ny       = 4;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 1;
   preLoc.halo.rt  = 1;
   preLoc.halo.dn  = 1;
   preLoc.halo.up  = 1;
   postLoc.nx      = 16;
   postLoc.ny      = 16;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 12;
   int nyp = 12;
   int nfp = 10;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, true, 0.0);
   PV::PostWeights postWeights(name, &weightsObject);

   int numKernelsX = postLoc.nx / preLoc.nx;
   int numKernelsY = postLoc.ny / preLoc.ny;
   int nxpPost     = nxp / numKernelsX;
   int nypPost     = nyp / numKernelsY;
   testPostWeights(postWeights, numKernelsX, numKernelsY, postLoc.nf, nxpPost, nypPost, preLoc.nf);
}

void testOneToManyNonshared() {
   std::string name("One-to-many, nonshared weights");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 4;
   preLoc.ny       = 4;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 1;
   preLoc.halo.rt  = 1;
   preLoc.halo.dn  = 1;
   preLoc.halo.up  = 1;
   postLoc.nx      = 16;
   postLoc.ny      = 16;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 12;
   int nyp = 12;
   int nfp = 10;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, false, 0.0);
   PV::PostWeights postWeights(name, &weightsObject);

   int nxExt   = postLoc.nx + postLoc.halo.lt + postLoc.halo.rt;
   int nyExt   = postLoc.ny + postLoc.halo.dn + postLoc.halo.up;
   int nxpPost = nxp / (postLoc.nx / preLoc.nx);
   int nypPost = nyp / (postLoc.ny / preLoc.ny);
   testPostWeights(postWeights, nxExt, nyExt, postLoc.nf, nxpPost, nypPost, preLoc.nf);
}

void testManyToOneShared() {
   std::string name("Many-to-one, shared weights");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 16;
   preLoc.ny       = 16;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 4;
   preLoc.halo.rt  = 4;
   preLoc.halo.dn  = 4;
   preLoc.halo.up  = 4;
   postLoc.nx      = 4;
   postLoc.ny      = 4;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 3;
   int nyp = 3;
   int nfp = 10;

   int nxpPost = nxp * preLoc.nx / postLoc.nx;
   int nypPost = nyp * preLoc.ny / postLoc.ny;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, true, 0.0);
   PV::PostWeights postWeights(name, &weightsObject);

   testPostWeights(postWeights, 1, 1, postLoc.nf, nxpPost, nypPost, preLoc.nf);
}

void testManyToOneNonshared() {
   std::string name("Many-to-one, nonshared weights");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 16;
   preLoc.ny       = 16;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 4;
   preLoc.halo.rt  = 4;
   preLoc.halo.dn  = 4;
   preLoc.halo.up  = 4;
   postLoc.nx      = 4;
   postLoc.ny      = 4;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 3;
   int nyp = 3;
   int nfp = 10;

   int nxpPost = nxp * preLoc.nx / postLoc.nx;
   int nypPost = nyp * preLoc.ny / postLoc.ny;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, false, 0.0);
   PV::PostWeights postWeights(name, &weightsObject);

   int nxExt = postLoc.nx + postLoc.halo.lt + postLoc.halo.rt;
   int nyExt = postLoc.ny + postLoc.halo.dn + postLoc.halo.up;
   testPostWeights(postWeights, nxExt, nyExt, postLoc.nf, nxpPost, nypPost, preLoc.nf);
}

int main(int argc, char *argv[]) {
   testOneToOneShared();
   testOneToOneNonshared();
   testOneToManyShared();
   testOneToManyNonshared();
   testManyToOneShared();
   testManyToOneNonshared();
   char *programPath = strdup(argv[0]);
   char *programName = basename(programPath);
   InfoLog() << programName << " passed.\n";
   free(programPath);
   return 0;
}
