/*
 * test_createPatchListDescription.cpp
 *
 */

#include <io/fileio.hpp>
#include <utils/PVLog.hpp>

#include <libgen.h> // basename
#include <stdexcept>

void testSharedOneToOne();
void testSharedManyToOne();
void testSharedOneToMany();

void testNonsharedOneToOne();
void testNonsharedManyToOne();
void testNonsharedOneToMany();

void testNonsharedOneToOneExcess();
void testNonsharedManyToOneExcess();
void testNonsharedOneToManyExcess();

int main(int argc, char *argv[]) {
   try {
      testSharedOneToOne();
      testSharedManyToOne();
      testSharedOneToMany();

      testNonsharedOneToOne();
      testNonsharedManyToOne();
      testNonsharedOneToMany();

      testNonsharedOneToOneExcess();
      testNonsharedManyToOneExcess();
      testNonsharedOneToManyExcess();
   } catch (std::exception const &except) {
      Fatal() << except.what() << " failed." << std::endl;
   }

   // If we got this far, all the tests passed.
   // Tests that fail exit with failure status instead of returning.
   char *progpath = strdup(argv[0]);
   char *progname = basename(progpath);
   InfoLog() << progname << " passed.\n";
   free(progpath);
   return EXIT_SUCCESS;
}

void testSharedOneToOne() {
   // One-to-one connection with shared weights on.
   // The cell size is 1-by-1-by-nfPre.
   // The patch list description is one line of nfPre patches, with no
   // excess border region, so that the stride is also nfPre.
   PVLayerLoc pre, post;

   pre.nx = 4;
   pre.ny = 8;
   pre.nf = 6;

   post.nx = 4;
   post.ny = 8;
   post.nf = 3;

   int nxp = 5;
   int nyp = 7;

   PV::PatchListDescription correct;
   correct.mStartIndex = 0;
   correct.mStrideY = 6;
   correct.mNumPatchesX   = 1;
   correct.mNumPatchesY   = 1;
   correct.mNumPatchesF   = 6;

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, true);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testSharedOneToOne");
   }
}

void testSharedManyToOne() {
   // Many-to-one connection with shared weights on.
   // The cell size is manyX-by-manyY-by-nfPre.
   // The patch list description is manyY lines of manyX*nfPre patches, with no
   // excess border region, so that the stride is also manyX*nfPre.
   PVLayerLoc pre, post;

   pre.nx = 16;
   pre.ny = 16;
   pre.nf = 6;

   post.nx = 4;
   post.ny = 8;
   post.nf = 3;

   int nxp = 5;
   int nyp = 7;

   PV::PatchListDescription correct;
   correct.mStartIndex = 0;
   correct.mStrideY = 24;
   correct.mNumPatchesX   = 4;
   correct.mNumPatchesY   = 2;
   correct.mNumPatchesF   = 6;

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, true);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testSharedManyToOne");
   }
}

void testSharedOneToMany() {
   // Many-to-one connection with shared weights on.
   // The cell size is 1-by-1-by-nfPre.
   // The patch list description is one line of nfPre patches, with no
   // excess border region, so that the stride is also nfPre.
   PVLayerLoc pre, post;

   pre.nx = 4;
   pre.ny = 8;
   pre.nf = 6;

   post.nx = 16;
   post.ny = 16;
   post.nf = 3;

   int nxp = 8;
   int nyp = 10;

   PV::PatchListDescription correct;
   correct.mStartIndex = 0;
   correct.mStrideY = 6;
   correct.mNumPatchesX   = 1;
   correct.mNumPatchesY   = 1;
   correct.mNumPatchesF   = 6;

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, true);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testSharedOneToMany");
   }
}

void testNonsharedOneToOne() {
   // One-to-one connection with shared weights off, and no excess border region.
   // The patch array is nxPreExt-by-nyPreExt-by-nfPre.
   // The patch list description is nyPreExt lines of nxPreExt*nfPre patches,
   // with no excess border region, so that the stride is alos nxPreExt*nfPre.
   PVLayerLoc pre, post;

   pre.nx = 4;
   pre.ny = 8;
   pre.nf = 6;

   post.nx = 4;
   post.ny = 8;
   post.nf = 3;

   int nxp = 5;
   int nyp = 7;

   pre.halo.lt = 2;
   pre.halo.rt = 2;
   pre.halo.dn = 3;
   pre.halo.up = 3;

   PV::PatchListDescription correct;
   correct.mStartIndex = 0;
   correct.mStrideY = 48;
   correct.mNumPatchesX   = 8;
   correct.mNumPatchesY   = 14;
   correct.mNumPatchesF   = 6;

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, false);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testNonsharedOneToOne");
   }
}

void testNonsharedManyToOne() {
   // Many-to-one connection with shared weights off.
   // The patch array is nxPreExt-by-nyPreExt-by-nfPre
   // The patch list description is nyPreExt lines of nxPreExt*nfPre patches,
   // with no excess border region, so that the stride is alos nxPreExt*nfPre.
   PVLayerLoc pre, post;

   pre.nx = 16;
   pre.ny = 16;
   pre.nf = 6;

   post.nx = 4;
   post.ny = 8;
   post.nf = 3;

   int nxp = 5;
   int nyp = 7;

   pre.halo.lt = 8; // 4-to-1 in the x-direction, and border requires 2 unit cells.
   pre.halo.rt = 8;
   pre.halo.dn = 6; // 2-to-1 in the y-direction, and border requires 3 unit cells.
   pre.halo.up = 6;

   PV::PatchListDescription correct;
   correct.mStartIndex = 0;
   correct.mStrideY = 192;
   correct.mNumPatchesX   = 32;
   correct.mNumPatchesY   = 28;
   correct.mNumPatchesF   = 6;

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, false);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testNonsharedManyToOne");
   }
}

void testNonsharedOneToMany() {
   // One-to-many connection with shared weights on.
   // The patch array is nxPreExt-by-nyPreExt-by-nfPre.
   // The patch list description is nyPreExt lines of nxPreExt*nfPre patches,
   // with no excess border region, so that the stride is alos nxPreExt*nfPre.
   PVLayerLoc pre, post;

   pre.nx = 4;
   pre.ny = 8;
   pre.nf = 6;

   post.nx = 16;
   post.ny = 16;
   post.nf = 3;

   int nxp = 8;
   int nyp = 10;

   pre.halo.lt = 1; // 1-to-4 in the x-direction, and the border requires 1/2 a unit cell.
   pre.halo.rt = 1;
   pre.halo.dn = 2; // 1-to-2 in the y-direction, and the border requires 2 unit cells.
   pre.halo.up = 2;

   PV::PatchListDescription correct;
   correct.mStartIndex = 0;
   correct.mStrideY = 36; // nxExt = 6; nf = 6.
   correct.mNumPatchesX   = 6; // nxExt = 6.
   correct.mNumPatchesY   = 12; // nyExt = 12.
   correct.mNumPatchesF   = 6; // nf = 6.

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, false);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testNonsharedOneToMany");
   }
}

void testNonsharedOneToOneExcess() {
   // One-to-one connection with shared weights off, with an excess border region.
   // Put nxPreReq the width of the border region required by the connection,
   // and nxPreExc the width of the border region including the excess region.
   // Similarly put nyPreReq and nyPreExc.
   // The result of createPatchListDescription should have
   //     mNumPatchesY = nyPreReq
   //     mLineLength = nxPreReq * nfPre
   //     mStrideY = nxPreExc * nfPre
   //     mStartIndex the index of of the patch with
   //         x = (nxPreExc-nyPreReq)/2, y = (nyPreExc-nyPreReq)/2, f = 0.
   PVLayerLoc pre, post;

   pre.nx = 4;
   pre.ny = 8;
   pre.nf = 6;

   post.nx = 4;
   post.ny = 8;
   post.nf = 3;

   int nxp = 5;
   int nyp = 7;

   pre.halo.lt = 3; // 1-1 connection with nxp=5 requires 2; excess is 1.
   pre.halo.rt = 3;
   pre.halo.dn = 4; // 1-1 connection with nyp=7 requires 3; excess is 1.
   pre.halo.up = 4;

   PV::PatchListDescription correct;
   correct.mStartIndex = 66; // kIndex(x=1, y=1, f=0, nx=10, ny=16, nf=6)
   correct.mStrideY = 60; // nxPreExc = 10; nf = 6.
   correct.mNumPatchesX   = 8; // nxPreReq = 8.
   correct.mNumPatchesY   = 14; // nyPreReq = 14.
   correct.mNumPatchesF   = 6; // nf = 6.

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, false);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testNonsharedOneToOneExcess");
   }
}

void testNonsharedManyToOneExcess() {
   // Many-to-one connection with shared weights off, with an excess border region.
   // Put nxPreReq the width of the border region required by the connection,
   // and nxPreExc the width of the border region including the excess region.
   // Similarly put nyPreReq and nyPreExc.
   // The result of createPatchListDescription should have
   //     mNumPatchesY = nyPreReq
   //     mLineLength = nxPreReq * nfPre
   //     mStrideY = nxPreExc * nfPre
   //     mStartIndex the index of of the patch with
   //         x = (nxPreExc-nyPreReq)/2, y = (nyPreExc-nyPreReq)/2, f = 0.
   PVLayerLoc pre, post;

   pre.nx = 16;
   pre.ny = 16;
   pre.nf = 6;

   post.nx = 4;
   post.ny = 8;
   post.nf = 3;

   int nxp = 5;
   int nyp = 7;

   pre.halo.lt = 10; // 4-1 connection with nxp=5 requires 8; excess is 2.
   pre.halo.rt = 10;
   pre.halo.dn = 10; // 2-1 connection with nyp=7 requires 6; excess is 4.
   pre.halo.up = 10;

   PV::PatchListDescription correct;
   correct.mStartIndex = 876; // kIndex(x=2, y=4, f=0, nx=36, ny=36, nf=6)
   correct.mStrideY = 216; // nxPreExc = 36; nf = 6.
   correct.mNumPatchesX   = 32; // nyPreReq = 32.
   correct.mNumPatchesY   = 28; // nyPreReq = 28.
   correct.mNumPatchesF   = 6; // nf = 6.

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, false);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testNonsharedManyToOneExcess");
   }
}

void testNonsharedOneToManyExcess() {
   // One-to-many connection with shared weights on.
   // Put nxPreReq the width of the border region required by the connection,
   // and nxPreExc the width of the border region including the excess region.
   // Similarly put nyPreReq and nyPreExc.
   // The result of createPatchListDescription should have
   //     mNumPatchesY = nyPreReq
   //     mLineLength = nxPreReq * nfPre
   //     mStrideY = nxPreExc * nfPre
   //     mStartIndex the index of of the patch with
   //         x = (nxPreExc-nyPreReq)/2, y = (nyPreExc-nyPreReq)/2, f = 0.
   PVLayerLoc pre, post;

   pre.nx = 4;
   pre.ny = 8;
   pre.nf = 6;

   post.nx = 16;
   post.ny = 16;
   post.nf = 3;

   int nxp = 8;
   int nyp = 10;

   pre.halo.lt = 4; // 1-4 connection with nxp=8 requires 1; excess is 3.
   pre.halo.rt = 4;
   pre.halo.dn = 4; // 1-2 connection with nyp=10 requires 2; excess is 2.
   pre.halo.up = 4;

   PV::PatchListDescription correct;
   correct.mStartIndex = 162; // kIndex(x=3, y=1, f=0, nx=12, ny=16, nf=6)
   correct.mStrideY = 72; // nxPreExc = 12; nf = 6.
   correct.mNumPatchesX   = 6; // nxPreReq = 28.
   correct.mNumPatchesY   = 12; // nyPreReq = 28.
   correct.mNumPatchesF   = 6; // nf = 6.

   auto patchListDescription = PV::createPatchListDescription(&pre, &post, nxp, nyp, false);
   if (memcmp(&patchListDescription, &correct, sizeof(patchListDescription)) != 0) {
      throw std::logic_error("testNonsharedOneToManyExcess");
   }
}
