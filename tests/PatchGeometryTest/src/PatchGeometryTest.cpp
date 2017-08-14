/*
 * PatchGeometryTest.cpp
 */

#include <components/PatchGeometry.hpp>
#include <string.h>
#include <utils/PVLog.hpp>
#include <utils/conversions.h>

void testOneToOneRestricted() {
   std::string name("One-to-one, patch size 1");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 16;
   preLoc.ny       = 16;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 0;
   preLoc.halo.rt  = 0;
   preLoc.halo.dn  = 0;
   preLoc.halo.up  = 0;
   postLoc.nx      = 16;
   postLoc.ny      = 16;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 1;
   int nyp = 1;
   int nfp = 10;

   PV::PatchGeometry patchGeometry(name, nxp, nyp, nfp, &preLoc, &postLoc);
   patchGeometry.allocateDataStructures();

   int numPatches = patchGeometry.getNumPatches();
   FatalIf(
         numPatches != preLoc.nx * preLoc.ny * preLoc.nf,
         "%s: expected %d patches; there were %d.\n",
         name.c_str(),
         preLoc.nx * preLoc.ny * preLoc.nf,
         numPatches);
   for (int p = 0; p < numPatches; p++) {
      PV::Patch const &patch = patchGeometry.getPatch(p);
      FatalIf(
            patch.nx != nxp or patch.ny != nxp or patch.offset != 0,
            "%s: patch %d is (nx=%d, ny=%d, offset=%d) instead of "
            "expected (nx=%d, ny=%d, offset=0).\n",
            name.c_str(),
            p,
            patch.nx,
            patch.ny,
            patch.offset,
            nxp,
            nyp);
      std::size_t correctGSynPatchStart = (p / preLoc.nf) * postLoc.nf; // integer division
      FatalIf(
            patchGeometry.getGSynPatchStart(p) != correctGSynPatchStart,
            "%s: patch %d has GSynPatchStart %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getGSynPatchStart(p),
            correctGSynPatchStart);
      std::size_t correctAPostOffset = correctGSynPatchStart; // postsynaptic margin zero
      FatalIf(
            patchGeometry.getAPostOffset(p) != correctAPostOffset,
            "%s: patch %d has APostOffset %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getAPostOffset(p),
            correctAPostOffset);
   }
}

void testOneToOneExtended() {
   std::string name("One-to-one, patch size 5");

   int marginWidth = 2;

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 16;
   preLoc.ny       = 16;
   preLoc.nf       = 3;
   preLoc.halo.lt  = marginWidth;
   preLoc.halo.rt  = marginWidth;
   preLoc.halo.dn  = marginWidth;
   preLoc.halo.up  = marginWidth;
   postLoc.nx      = 16;
   postLoc.ny      = 16;
   postLoc.nf      = 10;
   postLoc.halo.lt = marginWidth;
   postLoc.halo.rt = marginWidth;
   postLoc.halo.dn = marginWidth;
   postLoc.halo.up = marginWidth;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 2 * marginWidth + 1;
   int nyp = 2 * marginWidth + 1;
   int nfp = 10;

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;

   PV::PatchGeometry patchGeometry(name, nxp, nyp, nfp, &preLoc, &postLoc);
   patchGeometry.allocateDataStructures();

   int numPatches         = patchGeometry.getNumPatches();
   int expectedNumPatches = (preLoc.nx + preLoc.halo.lt + preLoc.halo.rt)
                            * (preLoc.ny + preLoc.halo.dn + preLoc.halo.up) * preLoc.nf;
   FatalIf(
         numPatches != expectedNumPatches,
         "%s: expected %d patches; there were %d.\n",
         name.c_str(),
         preLoc.nx * preLoc.ny * preLoc.nf,
         numPatches);

   std::vector<int> const correctSizes{1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1};
   std::vector<int> const correctStarts{4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   for (int p = 0; p < numPatches; p++) {
      int numPatchesX = patchGeometry.getNumPatchesX();
      int numPatchesY = patchGeometry.getNumPatchesY();
      int numPatchesF = patchGeometry.getNumPatchesF();
      int xIndex      = kxPos(p, numPatchesX, numPatchesY, numPatchesF);
      int yIndex      = kyPos(p, numPatchesX, numPatchesY, numPatchesF);
      int fIndex      = featureIndex(p, numPatchesX, numPatchesY, numPatchesF);

      auto &patch = patchGeometry.getPatch(p);

      int correctNx = correctSizes[xIndex];
      int correctNy = correctSizes[yIndex];

      int correctStartX = correctStarts[xIndex];
      int correctStartY = correctStarts[yIndex];
      int correctOffset = kIndex(correctStartX, correctStartY, 0, nxp, nyp, nfp);
      FatalIf(
            patch.nx != correctNx or patch.ny != correctNy or patch.offset != correctOffset,
            "%s: patch %d is (nx=%d, ny=%d, offset=%d) instead of "
            "expected (nx=%d, ny=%d, offset=%d).\n",
            name.c_str(),
            p,
            patch.nx,
            patch.ny,
            patch.offset,
            correctNx,
            correctNy,
            correctOffset);

      int xPost = xIndex < 2 * marginWidth ? 0 : xIndex - 2 * marginWidth;
      int yPost = yIndex < 2 * marginWidth ? 0 : yIndex - 2 * marginWidth;
      int fPost = 0;
      std::size_t correctGSynPatchStart =
            kIndex(xPost, yPost, fPost, postLoc.nx, postLoc.ny, postLoc.nf);
      FatalIf(
            patchGeometry.getGSynPatchStart(p) != correctGSynPatchStart,
            "%s: patch %d has GSynPatchStart %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getGSynPatchStart(p),
            correctGSynPatchStart);

      xPost += marginWidth;
      yPost += marginWidth;
      std::size_t correctAPostOffset =
            kIndex(xPost, yPost, fPost, numPatchesX, numPatchesY, postLoc.nf);
      FatalIf(
            patchGeometry.getAPostOffset(p) != correctAPostOffset,
            "%s: patch %d has APostOffset %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getAPostOffset(p),
            correctAPostOffset);
   }
}

void testOneToManyRestricted() {
   std::string name("One-to-many, patch size 4");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 4;
   preLoc.ny       = 4;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 0;
   preLoc.halo.rt  = 0;
   preLoc.halo.dn  = 0;
   preLoc.halo.up  = 0;
   postLoc.nx      = 16;
   postLoc.ny      = 16;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 4;
   int nyp = 4;
   int nfp = 10;

   PV::PatchGeometry patchGeometry(name, nxp, nyp, nfp, &preLoc, &postLoc);
   patchGeometry.allocateDataStructures();

   int numPatches = patchGeometry.getNumPatches();
   FatalIf(
         numPatches != preLoc.nx * preLoc.ny * preLoc.nf,
         "%s: expected %d patches; there were %d.\n",
         name.c_str(),
         preLoc.nx * preLoc.ny * preLoc.nf,
         numPatches);
   for (int p = 0; p < numPatches; p++) {
      PV::Patch const &patch = patchGeometry.getPatch(p);
      FatalIf(
            patch.nx != nxp or patch.ny != nyp or patch.offset != 0,
            "One-to-many restriced, shared weights: patch %d is (nx=%d, ny=%d, offset=%d) "
            "instead of expected (nx=%d ny=%d offset=0).\n",
            p,
            patch.nx,
            patch.ny,
            patch.offset,
            nxp,
            nyp);

      int xPost = kxPos(p, preLoc.nx, preLoc.ny, preLoc.nf) * nxp;
      int yPost = kyPos(p, preLoc.nx, preLoc.ny, preLoc.nf) * nyp;
      std::size_t correctGSynPatchStart =
            kIndex(xPost, yPost, 0, postLoc.nx, postLoc.ny, postLoc.nf);
      FatalIf(
            patchGeometry.getGSynPatchStart(p) != correctGSynPatchStart,
            "%s: patch %d has GSynPatchStart %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getGSynPatchStart(p),
            correctGSynPatchStart);
      std::size_t correctAPostOffset = correctGSynPatchStart; // postsynaptic margin zero
      FatalIf(
            patchGeometry.getAPostOffset(p) != correctAPostOffset,
            "%s: patch %d has APostOffset %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getAPostOffset(p),
            correctAPostOffset);
   }
}

void testOneToManyExtended() {
   std::string name("One-to-many, patch size 12");

   int tstride    = 4;
   int preMargin  = 1;
   int postMargin = preMargin * tstride;

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 4;
   preLoc.ny       = 4;
   preLoc.nf       = 3;
   preLoc.halo.lt  = preMargin;
   preLoc.halo.rt  = preMargin;
   preLoc.halo.dn  = preMargin;
   preLoc.halo.up  = preMargin;
   postLoc.nx      = preLoc.nx * tstride;
   postLoc.ny      = preLoc.ny * tstride;
   postLoc.nf      = 10;
   postLoc.halo.lt = preMargin * tstride;
   postLoc.halo.rt = preMargin * tstride;
   postLoc.halo.dn = preMargin * tstride;
   postLoc.halo.up = preMargin * tstride;
   // Other fields of preLoc, postLoc are not used.

   int nxp = (2 * preMargin + 1) * tstride;
   int nyp = (2 * preMargin + 1) * tstride;
   int nfp = 10;

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;

   PV::PatchGeometry patchGeometry(name, nxp, nyp, nfp, &preLoc, &postLoc);
   patchGeometry.allocateDataStructures();

   int numPatches         = patchGeometry.getNumPatches();
   int expectedNumPatches = (preLoc.nx + preLoc.halo.lt + preLoc.halo.rt)
                            * (preLoc.ny + preLoc.halo.dn + preLoc.halo.up) * preLoc.nf;
   FatalIf(
         numPatches != expectedNumPatches,
         "%s: expected %d patches; there were %d.\n",
         name.c_str(),
         expectedNumPatches,
         numPatches);

   std::vector<int> const correctSizes{4, 8, 12, 12, 8, 4};
   std::vector<int> const correctStarts{8, 4, 0, 0, 0, 0};
   for (int p = 0; p < numPatches; p++) {
      int xIndex =
            kxPos(p,
                  patchGeometry.getNumPatchesX(),
                  patchGeometry.getNumPatchesY(),
                  patchGeometry.getNumPatchesF());
      int yIndex =
            kyPos(p,
                  patchGeometry.getNumPatchesX(),
                  patchGeometry.getNumPatchesY(),
                  patchGeometry.getNumPatchesF());
      int fIndex = featureIndex(
            p,
            patchGeometry.getNumPatchesX(),
            patchGeometry.getNumPatchesY(),
            patchGeometry.getNumPatchesF());

      auto &patch = patchGeometry.getPatch(p);

      int correctNx = correctSizes[xIndex];
      int correctNy = correctSizes[yIndex];

      int correctStartX = correctStarts[xIndex];
      int correctStartY = correctStarts[yIndex];
      int correctOffset = kIndex(correctStartX, correctStartY, 0, nxp, nyp, nfp);
      FatalIf(
            patch.nx != correctNx or patch.ny != correctNy or patch.offset != correctOffset,
            "%s: patch %d is (nx=%d, ny=%d, offset=%d) instead of "
            "expected (nx=%d, ny=%d, offset=%d).\n",
            name.c_str(),
            p,
            patch.nx,
            patch.ny,
            patch.offset,
            correctNx,
            correctNy,
            correctOffset);

      int xPost     = (xIndex < 2 * preMargin ? 0 : xIndex - 2 * preMargin) * tstride;
      int yPost     = (yIndex < 2 * preMargin ? 0 : yIndex - 2 * preMargin) * tstride;
      int nxPostRes = postLoc.nx;
      int nyPostRes = postLoc.ny;
      std::size_t correctGSynPatchStart = kIndex(xPost, yPost, 0, nxPostRes, nyPostRes, postLoc.nf);
      FatalIf(
            patchGeometry.getGSynPatchStart(p) != correctGSynPatchStart,
            "%s: patch %d has GSynPatchStart %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getGSynPatchStart(p),
            correctGSynPatchStart);

      xPost += preMargin * tstride;
      yPost += preMargin * tstride;
      int nxPostExt                  = postLoc.nx + postLoc.halo.lt + postLoc.halo.rt;
      int nyPostExt                  = postLoc.ny + postLoc.halo.dn + postLoc.halo.up;
      std::size_t correctAPostOffset = kIndex(xPost, yPost, 0, nxPostExt, nyPostExt, postLoc.nf);
      FatalIf(
            patchGeometry.getAPostOffset(p) != correctAPostOffset,
            "%s: patch %d has APostOffset %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getAPostOffset(p),
            correctAPostOffset);
   }
}

void testManyToOneRestricted() {
   std::string name("Many-to-one, patch size 1");

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 16;
   preLoc.ny       = 16;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 0;
   preLoc.halo.rt  = 0;
   preLoc.halo.dn  = 0;
   preLoc.halo.up  = 0;
   postLoc.nx      = 4;
   postLoc.ny      = 4;
   postLoc.nf      = 10;
   postLoc.halo.lt = 0;
   postLoc.halo.rt = 0;
   postLoc.halo.dn = 0;
   postLoc.halo.up = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 1;
   int nyp = 1;
   int nfp = 10;

   int xStride = preLoc.nx / postLoc.nx;
   int yStride = preLoc.ny / postLoc.ny;

   PV::PatchGeometry patchGeometry(name, nxp, nyp, nfp, &preLoc, &postLoc);
   patchGeometry.allocateDataStructures();

   int numPatches = patchGeometry.getNumPatches();
   FatalIf(
         numPatches != preLoc.nx * preLoc.ny * preLoc.nf,
         "%s: expected %d patches; there were %d.\n",
         name.c_str(),
         preLoc.nx * preLoc.ny * preLoc.nf,
         numPatches);
   for (int p = 0; p < numPatches; p++) {
      PV::Patch const &patch = patchGeometry.getPatch(p);
      FatalIf(
            patch.nx != nxp or patch.ny != nyp or patch.offset != 0,
            "%s: patch %d is (nx=%d, ny=%d, offset=%d) "
            "instead of expected (nx=%d ny=%d offset=0).\n",
            name.c_str(),
            p,
            patch.nx,
            patch.ny,
            patch.offset,
            nxp,
            nyp);

      int xPost = kxPos(p, preLoc.nx, preLoc.ny, preLoc.nf) / xStride; // integer division
      int yPost = kyPos(p, preLoc.nx, preLoc.ny, preLoc.nf) / yStride; // integer division
      std::size_t correctGSynPatchStart =
            kIndex(xPost, yPost, 0, postLoc.nx, postLoc.ny, postLoc.nf);
      FatalIf(
            patchGeometry.getGSynPatchStart(p) != correctGSynPatchStart,
            "%s: patch %d has GSynPatchStart %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getGSynPatchStart(p),
            correctGSynPatchStart);
      std::size_t correctAPostOffset = correctGSynPatchStart; // postsynaptic margin zero
      FatalIf(
            patchGeometry.getAPostOffset(p) != correctAPostOffset,
            "%s: patch %d has APostOffset %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getAPostOffset(p),
            correctAPostOffset);
   }
}

void testManyToOneExtended() {
   std::string name("Many-to-one, patch size 3");

   int stride     = 4;
   int postMargin = 1;
   int preMargin  = stride * postMargin;

   PVLayerLoc preLoc, postLoc;
   preLoc.nx       = 4 * stride;
   preLoc.ny       = 4 * stride;
   preLoc.nf       = 3;
   preLoc.halo.lt  = preMargin;
   preLoc.halo.rt  = preMargin;
   preLoc.halo.dn  = preMargin;
   preLoc.halo.up  = preMargin;
   postLoc.nx      = 4;
   postLoc.ny      = 4;
   postLoc.nf      = 10;
   postLoc.halo.lt = postMargin;
   postLoc.halo.rt = postMargin;
   postLoc.halo.dn = postMargin;
   postLoc.halo.up = postMargin;
   // Other fields of preLoc, postLoc are not used.

   int nxp = 2 * postMargin + 1;
   int nyp = 2 * postMargin + 1;
   int nfp = 10;

   int xStride = preLoc.nx / postLoc.nx;
   int yStride = preLoc.ny / postLoc.ny;

   int nxExt = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyExt = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;

   PV::PatchGeometry patchGeometry(name, nxp, nyp, nfp, &preLoc, &postLoc);
   patchGeometry.allocateDataStructures();

   int numPatches         = patchGeometry.getNumPatches();
   int expectedNumPatches = (preLoc.nx + preLoc.halo.lt + preLoc.halo.rt)
                            * (preLoc.ny + preLoc.halo.dn + preLoc.halo.up) * preLoc.nf;
   FatalIf(
         numPatches != expectedNumPatches,
         "%s: expected %d patches; there were %d.\n",
         name.c_str(),
         expectedNumPatches,
         numPatches);

   std::vector<int> correctSizes(nxExt, nxp);
   std::vector<int> correctStarts(nxExt, 0);
   for (int k = 0; k < 4; k++) {
      correctSizes[k]      = 1;
      correctSizes[k + 4]  = 2;
      correctSizes[k + 16] = 2;
      correctSizes[k + 20] = 1;
      correctStarts[k]     = 2;
      correctStarts[k + 4] = 1;
   }
   for (int p = 0; p < numPatches; p++) {
      int xIndex =
            kxPos(p,
                  patchGeometry.getNumPatchesX(),
                  patchGeometry.getNumPatchesY(),
                  patchGeometry.getNumPatchesF());
      int yIndex =
            kyPos(p,
                  patchGeometry.getNumPatchesX(),
                  patchGeometry.getNumPatchesY(),
                  patchGeometry.getNumPatchesF());
      int fIndex = featureIndex(
            p,
            patchGeometry.getNumPatchesX(),
            patchGeometry.getNumPatchesY(),
            patchGeometry.getNumPatchesF());

      auto &patch = patchGeometry.getPatch(p);

      int correctNx = correctSizes[xIndex];
      int correctNy = correctSizes[yIndex];

      int correctStartX = correctStarts[xIndex];
      int correctStartY = correctStarts[yIndex];
      int correctOffset = kIndex(correctStartX, correctStartY, 0, nxp, nyp, nfp);
      FatalIf(
            patch.nx != correctNx or patch.ny != correctNy or patch.offset != correctOffset,
            "%s: patch %d is (nx=%d, ny=%d, offset=%d) instead of "
            "expected (nx=%d, ny=%d, offset=%d).\n",
            name.c_str(),
            p,
            patch.nx,
            patch.ny,
            patch.offset,
            correctNx,
            correctNy,
            correctOffset);

      int xPost     = (xIndex < 2 * preMargin ? 0 : xIndex - 2 * preMargin) / stride;
      int yPost     = (yIndex < 2 * preMargin ? 0 : yIndex - 2 * preMargin) / stride;
      int nxPostRes = postLoc.nx;
      int nyPostRes = postLoc.ny;
      std::size_t correctGSynPatchStart = kIndex(xPost, yPost, 0, nxPostRes, nyPostRes, postLoc.nf);
      FatalIf(
            patchGeometry.getGSynPatchStart(p) != correctGSynPatchStart,
            "%s: patch %d has GSynPatchStart %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getGSynPatchStart(p),
            correctGSynPatchStart);

      xPost += postMargin;
      yPost += postMargin;
      int nxPostExt                  = postLoc.nx + postLoc.halo.lt + postLoc.halo.rt;
      int nyPostExt                  = postLoc.ny + postLoc.halo.dn + postLoc.halo.up;
      std::size_t correctAPostOffset = kIndex(xPost, yPost, 0, nxPostExt, nyPostExt, postLoc.nf);
      FatalIf(
            patchGeometry.getAPostOffset(p) != correctAPostOffset,
            "%s: patch %d has APostOffset %zu instead of expected %zu.\n",
            name.c_str(),
            p,
            patchGeometry.getAPostOffset(p),
            correctAPostOffset);
   }
}

int main(int argc, char *argv[]) {
   testOneToOneRestricted();
   testOneToOneExtended();
   testOneToManyRestricted();
   testOneToManyExtended();
   testManyToOneRestricted();
   testManyToOneExtended();
   char *programPath = strdup(argv[0]);
   char *programName = basename(programPath);
   InfoLog() << programName << " passed.\n";
   free(programPath);
   return 0;
}
