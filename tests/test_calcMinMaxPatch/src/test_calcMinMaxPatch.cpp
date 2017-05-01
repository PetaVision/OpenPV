/*
 * test_PV::calcMinMaxPatch.cpp
 *
 */

#include <io/fileio.hpp>
#include <utils/PVLog.hpp>

#include <libgen.h> // basename
#include <stdexcept>
#include <vector>

// Tests calcMinMaxPatch for unshrunken patches
void testUnshrunkenIncreasing();
void testUnshrunkenDecreasing();
void testUnshrunkenMinInMiddle();
void testUnshrunkenMaxInMiddle();

// Tests calcMinMaxPatch for shrunken patches
void testShrunkenIncreasing();
void testShrunkenDecreasing();
void testShrunkenMinInMiddle();
void testShrunkenMaxInMiddle();

// Tests that min and max accumulate with input values of {min,max}Weight
void testRunningMinMax();

int main(int argc, char *argv[]) {
   try {
      testUnshrunkenIncreasing();
      testUnshrunkenDecreasing();
      testUnshrunkenMinInMiddle();
      testUnshrunkenMaxInMiddle();
      testShrunkenIncreasing();
      testShrunkenDecreasing();
      testShrunkenMinInMiddle();
      testShrunkenMaxInMiddle();
      testRunningMinMax();
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

void testUnshrunkenIncreasing() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = -6.0f + 0.25f * (float)k;
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);

   if (minWeight != -6.0f || maxWeight != 5.75f) {
      throw std::logic_error("testUnshrunkenIncreasing");
   }
}

void testUnshrunkenDecreasing() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = 6.0f - 0.25f * (float)k;
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);

   if (minWeight != -5.75f || maxWeight != 6.0f) {
      throw std::logic_error("testUnshrunkenDecreasing");
   }
}

void testUnshrunkenMinInMiddle() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = std::abs(0.25f * (float)k - 6.0f);
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);

   if (minWeight != 0.0f || maxWeight != 6.0f) {
      throw std::logic_error("testUnshrunkenMinInMiddle");
   }
}

void testUnshrunkenMaxInMiddle() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = 8.0f - std::abs(0.25f * (float)k - 6.0f);
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);

   if (minWeight != 2.0f || maxWeight != 8.0f) {
      throw std::logic_error("testUnshrunkenMinInMiddle");
   }
}

void testShrunkenIncreasing() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = -6.0f + 0.25f * (float)k;
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   // Full patch is 4-by-4 with nf=3. Check only the 2-by-2 patch in middle.
   // Offset is kIndex(1,1,0,4,4,3) = 15.
   // Minimum occurs at kIndex(1,1,0,4,4,3) = 15. Minimum value is -2.25
   // Maximum occurs at kIndex(2,2,2,4,4,3) = 32. Maximum value is 2.0
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 2U, 2U, 15U, 12U);

   if (minWeight != -2.25f || maxWeight != 2.0f) {
      throw std::logic_error("testShrunkenIncreasing");
   }
}

void testShrunkenDecreasing() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = 6.0f - 0.25f * (float)k;
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   // Full patch is 4-by-4 with nf=3. Check only the 2-by-2 patch in middle.
   // Offset is kIndex(1,1,0,4,4,3) = 15.
   // Maximum occurs at kIndex(1,1,0,4,4,3) = 15. Maximum value is 2.25
   // Minimum occurs at kIndex(2,2,2,4,4,3) = 32. Minimum value is -2.0
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 2U, 2U, 15U, 12U);

   if (minWeight != -2.0f || maxWeight != 2.25f) {
      throw std::logic_error("testShrunkenDecreasing");
   }
}

void testShrunkenMinInMiddle() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = std::abs(0.25f * (float)k - 6.0f);
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   // Full patch is 4-by-4 with nf=3. Check only the 2-by-2 patch in middle.
   // Offset is kIndex(1,1,0,4,4,3) = 15.
   // Minimum occurs at kIndex(1,2,0,4,4,3) = 27. Minimum value is 0.75
   // Maximum occurs at kIndex(1,1,0,4,4,3) = 15. Maximum value is 2.25
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 2U, 2U, 15U, 12U);

   if (minWeight != 0.75f || maxWeight != 2.25f) {
      throw std::logic_error("testShrunkenMinInMiddle");
   }
}

void testShrunkenMaxInMiddle() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = 8.0f - std::abs(0.25f * (float)k - 6.0f);
   }

   float minWeight = std::numeric_limits<float>::infinity();
   float maxWeight = -std::numeric_limits<float>::infinity();

   // Full patch is 4-by-4 with nf=3. Check only the 2-by-2 patch in middle.
   // Offset is kIndex(1,1,0,4,4,3) = 15.
   // Minimum occurs at kIndex(1,1,0,4,4,3) = 15. Minimum value is 5.75
   // Maximum occurs at kIndex(1,2,0,4,4,3) = 27. Maximum value is 7.25
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 2U, 2U, 15U, 12U);

   if (minWeight != 5.75f || maxWeight != 7.25f) {
      throw std::logic_error("testShrunkenMinInMiddle");
   }
}

void testRunningMinMax() {
   std::vector<float> patch(48);
   for (int k = 0; k < 48; k++) {
      patch[k] = -6.0f + 0.25f * (float)k;
   }

   float minWeight, maxWeight;

   minWeight = -10.0f; // Less than the minumum over the patch
   maxWeight = 8.0f; // Greater than the maximum over the patch
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);
   // minWeight and maxWeight should not change.
   if (minWeight != -10.0f || maxWeight != 8.0f) {
      throw std::logic_error("testRunningMinMax");
   }

   minWeight = -10.0f; // Less than the minumum over the patch
   maxWeight = 4.0f; // Less than the maximum over the patch
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);
   // minWeight should not change, but maxWeight should.
   if (minWeight != -10.0f || maxWeight != 5.75f) {
      throw std::logic_error("testRunningMinMax");
   }

   minWeight = -1.0f; // Greater than the minumum over the patch
   maxWeight = 10.0f; // Greater than the maximum over the patch
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);
   // minWeight should change, but maxWeight should not.
   if (minWeight != -6.0f || maxWeight != 10.0f) {
      throw std::logic_error("testRunningMinMax");
   }

   minWeight = -1.0f; // Greater than the minumum over the patch
   maxWeight = 10.0f; // Greater than the maximum over the patch
   PV::calcMinMaxPatch(minWeight, maxWeight, patch.data(), 3U, 4U, 4U, 0U, 12U);
   // minWeight should change, but maxWeight should not.
   if (minWeight != -6.0f || maxWeight != 10.0f) {
      throw std::logic_error("testRunningMinMax");
   }
}
