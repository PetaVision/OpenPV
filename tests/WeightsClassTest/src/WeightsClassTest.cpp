/*
 * WeightsClassTest.cpp
 */

#include <components/Weights.hpp>
#include <utils/PVLog.hpp>
#include <utils/conversions.h>

#include <string.h>

void testWeights(
      PV::Weights &weights,
      int correctNumDataPatchesX,
      int correctNumDataPatchesY,
      int correctNumDataPatchesF,
      int correctPatchSizeX,
      int correctPatchSizeY,
      int correctPatchSizeF) {
   PVLayerLoc const &preLoc = weights.getGeometry()->getPreLoc();
   int nxExt                = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt                = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   FatalIf(
         weights.getGeometry()->getNumPatchesX() != nxExt,
         "%s: number of geometry patches in x-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getGeometry()->getNumPatchesX(),
         nxExt);
   FatalIf(
         weights.getGeometry()->getNumPatchesY() != nyExt,
         "%s: number of geometry patches in y-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getGeometry()->getNumPatchesY(),
         nyExt);
   FatalIf(
         weights.getGeometry()->getNumPatchesF() != preLoc.nf,
         "%s: number of geometry patches in feature direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getGeometry()->getNumPatchesF(),
         preLoc.nf);
   FatalIf(
         weights.getGeometry()->getNumPatches() != nxExt * nyExt * preLoc.nf,
         "%s: number of geometry patches overall was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getGeometry()->getNumPatchesF(),
         nxExt * nyExt * preLoc.nf);

   FatalIf(
         weights.getNumDataPatchesX() != correctNumDataPatchesX,
         "%s: number of data patches in x-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getNumDataPatchesX(),
         correctNumDataPatchesX);
   FatalIf(
         weights.getNumDataPatchesY() != correctNumDataPatchesY,
         "%s: number of data patches in y-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getNumDataPatchesY(),
         correctNumDataPatchesY);
   FatalIf(
         weights.getNumDataPatchesF() != correctNumDataPatchesF,
         "%s: number of data patches in feature direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getNumDataPatchesX(),
         correctNumDataPatchesF);

   int correctNumDataPatches =
         correctNumDataPatchesX * correctNumDataPatchesY * correctNumDataPatchesF;
   FatalIf(
         weights.getNumDataPatches() != correctNumDataPatches,
         "%s: number of data patches overall was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getNumDataPatchesX(),
         correctNumDataPatchesF);

   FatalIf(
         weights.getPatchSizeX() != correctPatchSizeX,
         "%s: patch size in x-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getPatchSizeX(),
         correctPatchSizeX);
   FatalIf(
         weights.getPatchSizeY() != correctPatchSizeY,
         "%s: patch size in x-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getPatchSizeY(),
         correctPatchSizeY);
   FatalIf(
         weights.getPatchSizeF() != correctPatchSizeF,
         "%s: patch size in x-direction was %d instead of the expected %d\n",
         weights.getName().c_str(),
         weights.getPatchSizeF(),
         correctPatchSizeF);

   // Test writing weights to the Weights object, and reading the weights back.
   weights.allocateDataStructures();

   int const numDataPatches = weights.getNumDataPatches();
   int const numItemsInPatch =
         weights.getPatchSizeX() * weights.getPatchSizeY() * weights.getPatchSizeF();
   int const numWeightValues = numDataPatches * numItemsInPatch;

   // Check that writing and reading timestamps works.
   double timestamp = weights.getTimestamp() ? 0.0 : 2.5;
   weights.setTimestamp(timestamp);
   FatalIf(
         weights.getTimestamp() != timestamp,
         "%s: timestamp was %f, instead of the expected %f\n",
         weights.getName().c_str(),
         weights.getTimestamp(),
         timestamp);

   // Write weights to the arbor as a whole
   float *dataStart = weights.getData(0);
   for (int w = 0; w < numWeightValues; w++) {
      dataStart[w] = (float)w;
   }

   // Read weights back, patch by patch, and check values
   for (int d = 0; d < numDataPatches; d++) {
      float const *dataPatchStart = weights.getDataFromDataIndex(0, d);
      for (int p = 0; p < numItemsInPatch; p++) {
         float correctValue  = d * numItemsInPatch + p;
         float observedValue = dataPatchStart[p];
         FatalIf(
               observedValue != correctValue,
               "%s: value at patch index %d, item %d, was %f instead of the expected %f\n",
               weights.getName().c_str(),
               d,
               p,
               (double)observedValue,
               (double)correctValue);
      }
   }
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

   testWeights(weightsObject, 1, 1, preLoc.nf, nxp, nyp, nfp);
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

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   testWeights(weightsObject, nxExt, nyExt, preLoc.nf, nxp, nyp, nfp);
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

   testWeights(weightsObject, 1, 1, preLoc.nf, nxp, nyp, nfp);
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

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   testWeights(weightsObject, nxExt, nyExt, preLoc.nf, nxp, nyp, nfp);
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

   int xStride = preLoc.nx / postLoc.nx;
   int yStride = preLoc.ny / postLoc.ny;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, true, 0.0);

   testWeights(weightsObject, xStride, yStride, preLoc.nf, nxp, nyp, nfp);
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

   int xStride = preLoc.nx / postLoc.nx;
   int yStride = preLoc.ny / postLoc.ny;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, 1, false, 0.0);

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   testWeights(weightsObject, nxExt, nyExt, preLoc.nf, nxp, nyp, nfp);
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
