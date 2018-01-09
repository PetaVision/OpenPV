/*
 * PostWeightsTest.cpp
 */

#include <components/WeightsPair.hpp>
#include <utils/PVLog.hpp>

#include <string.h>

// This test checks whether the PatchSize::calcPostPatchSize method works correctly.
void testOneToOne() {
   std::string name("One-to-one");

   int preNx  = 8;
   int postNx = 8;
   int nxp    = 5;

   int nxpPostCorrect = nxp;

   int nxpPost = PV::PatchSize::calcPostPatchSize(nxp, preNx, postNx);
   FatalIf(
         nxpPost != nxpPostCorrect,
         "%s failed. nxpPost = %d, should be %d.\n",
         name.c_str(),
         nxpPost,
         nxpPostCorrect);
}

void testOneToMany() {
   std::string name("One-to-many");

   int preNx  = 4;
   int postNx = 16;
   int nxp    = 12;

   int numKernelsX    = postNx / preNx;
   int nxpPostCorrect = nxp / numKernelsX;

   int nxpPost = PV::PatchSize::calcPostPatchSize(nxp, preNx, postNx);
   FatalIf(
         nxpPost != nxpPostCorrect,
         "%s failed. nxpPost = %d, should be %d.\n",
         name.c_str(),
         nxpPost,
         nxpPostCorrect);
}

void testManyToOne() {
   std::string name("Many-to-one");

   int preNx          = 16;
   int postNx         = 4;
   int nxp            = 3;
   int nxpPostCorrect = nxp * preNx / postNx;

   int nxpPost = PV::PatchSize::calcPostPatchSize(nxp, preNx, postNx);
   FatalIf(
         nxpPost != nxpPostCorrect,
         "%s failed. nxpPost = %d, should be %d.\n",
         name.c_str(),
         nxpPost,
         nxpPostCorrect);
}

int main(int argc, char *argv[]) {
   testOneToOne();
   testOneToMany();
   testManyToOne();
   char *programPath = strdup(argv[0]);
   char *programName = basename(programPath);
   InfoLog() << programName << " passed.\n";
   free(programPath);
   return 0;
}
