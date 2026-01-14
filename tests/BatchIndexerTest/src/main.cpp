#include "structures/BatchIndexer.hpp"
#include "utils/PVLog.hpp"
#include "utils/cl_random.h"

#include <memory>
#include <vector>

using PV::BatchIndexer;

void testByFile() {
   int value;
   std::vector<int> startIndices;
   std::vector<int> skipAmounts;

   // Test MPI batch dimension == 1
   startIndices = std::vector<int>{0, 1};
   skipAmounts  = std::vector<int>{2, 2};
   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile1"),
         2, // Global batch size 2
         0, // This MPI block starts at batch element 0.
         2, // 2 batch elements in MPI block (therefore, 1 MPI block).
         4, // 4 files to batch across
         startIndices,
         skipAmounts);
   batchIndexer->setWrapToStartIndex(false);

   // Test initial indices.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 0,
         "Failed. Expected %d, found %d instead.\n",
         0,
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 1,
         "Failed. Expected %d, found %d instead.\n",
         1,
         value);

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);

   // Test multiple MPI blocks in batch dimension
   startIndices = std::vector<int>{2, 3};
   skipAmounts  = std::vector<int>{4, 4};
   batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile2"),
         4, // Global batch size 4
         2, // This MPI block starts at batch element 2.
         2, // 2 batch elements in MPI block (therefore, 2 MPI blocks and this is the second one)
         8, // 8 files to batch across
         startIndices,
         skipAmounts);
   batchIndexer->setWrapToStartIndex(false);

   // Test initial indices.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 7,
         "Failed. Expected 7, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);
}

void testByList() {
   int value = 0;
   std::vector<int> startIndices;
   std::vector<int> skipAmounts;

   // Test MPI batch dimension == 1
   startIndices = std::vector<int>{0, 2};
   skipAmounts  = std::vector<int>{1, 1};
   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByList1"),
         2, // Global batch size 2
         0, // This MPI block starts at batch element 0.
         2, // 2 batch elements in MPI block (therefore, 1 MPI block).
         4, // 4 files to batch across
         startIndices,
         skipAmounts);
   batchIndexer->setWrapToStartIndex(true);

   // Test initial indices.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // BatchIndexer increments the index after returning the current
   // index, so looping actually happens one nextInput call before we
   // get the looped index. Store the indices now so we can test
   // both looping modes.
   std::vector<int> indicesBeforeLoop = batchIndexer->getIndices();

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index.
   // Because setWrapToStartIndex is true, these should be
   // the same as the initial state, not 0.

   // Batch 0 won't loop, it's going to march
   // right into where batch 1 started.
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Batch 1 should loop.
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Rewind our indices, try going over the loop again.
   // Since we disabled setWrapToStartIndex, we should
   // land on index 0
   batchIndexer->setIndices(indicesBeforeLoop);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 0, found %d instead.\n",
         value);
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);

   // Test MPI batch dimension > 1
   startIndices = std::vector<int>{4, 6};
   skipAmounts  = std::vector<int>{1, 1};
   batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByList2"),
         4, // Global batch size 4
         2, // This MPI block starts at batch element 2
         2, // 2 batch elements in MPI block (therefore, 2 MPI blocks).
         8, // 8 files to batch across
         startIndices,
         skipAmounts);
   batchIndexer->setWrapToStartIndex(true);

   // Test initial indices.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 4,
         "Failed. Expected 4, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);

   // BatchIndexer increments the index after returning the current
   // index, so looping actually happens one nextInput call before we
   // get the looped index. Store the indices now so we can test
   // both looping modes.
   indicesBeforeLoop = batchIndexer->getIndices();

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 5,
         "Failed. Expected 5, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 7,
         "Failed. Expected 7, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index.
   // Because setWrapToStartIndex is true, these should be
   // the same as the initial state, not 0.

   // Batch 0 won't loop, it's going to march
   // right into where batch 1 started.
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);

   // Batch 1 should loop.
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);

   // Rewind our indices, try going over the loop again.
   // Since we disabled setWrapToStartIndex, we should
   // land on index 0
   batchIndexer->setIndices(indicesBeforeLoop);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 7,
         "Failed. Expected 0, found %d instead.\n",
         value);
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
}

void testBySpecified() {
   int value;
   std::vector<int> startIndices;
   std::vector<int> skipAmounts;

   startIndices = std::vector<int>{2, 0};
   skipAmounts  = std::vector<int>{1, 2};
   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("BySpecified"),
         2, // Global batch size 2
         0, // This MPI block starts at batch element 0
         2, // 2 batch elements in MPI block (therefore, 1 MPI block)
         4, // 4 files to batch across
         startIndices,
         skipAmounts);
   batchIndexer->setWrapToStartIndex(true);

   // Test initial indices. The first call to getIndex() after
   // initializeBatch just returns the initial value.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);

   // BatchIndexer increments the index after returning the current
   // index, so looping actually happens one nextInput call before we
   // get the looped index. Store the indices now so we can test
   // both looping modes.
   std::vector<int> indicesBeforeLoop = batchIndexer->getIndices();

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index.
   // Because setWrapToStartIndex is true, these should be
   // the same as the initial state, not 0.

   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);

   // Rewind our indices, try going over the loop again.
   // Since we disabled setWrapToStartIndex, we should
   // land on index 0 for both
   batchIndexer->setIndices(indicesBeforeLoop);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 3,
         "Failed. Expected 0, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 2,
         "Failed. Expected 0, found %d instead.\n",
         value);
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
}

void testRandom() {
   int value;
   unsigned int seed;
   taus_uint4 rng;

   // Test MPI batch dimension == 1
   seed = 1439876414U;
   cl_random_init(&rng, 1, seed);
   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile1"),
         4, // Global batch size 4
         0, // This MPI block starts at batch element 0.
         4, // 4 batch elements in MPI block (therefore, 1 MPI block).
         8, // 8 files to batch across
         rng);
   batchIndexer->setWrapToStartIndex(false);

   // Test initial indices. Initial index lookup table is {4, 7, 5, 2, 1, 3, 6, 0}
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 4,
         "Failed. Expected %d, found %d instead.\n",
         4,
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 7,
         "Failed. Expected %d, found %d instead.\n",
         7,
         value);

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index.
   batchIndexer->advanceIndices();
   // Reshuffled index lookup table is now {1, 4, 6, 5, 3, 7, 0, 2}.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 4,
         "Failed. Expected 4, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(2)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(3)) != 5,
         "Failed. Expected 5, found %d instead.\n",
         value);

   // // Test multiple MPI blocks in batch dimension
   seed = 1439876417U;
   cl_random_init(&rng, 1, seed);
   batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile2"),
         4, // Global batch size 4
         2, // This MPI block starts at batch element 2.
         2, // 2 batch elements in MPI block (therefore, 2 MPI blocks and this is the second one)
         8, // 8 files to batch across
         rng);
   batchIndexer->setWrapToStartIndex(false);

   // Test initial indices. Initial index lookup table is {5, 3, 7, 4, 0, 6, 1, 2}
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 7,
         "Failed. Expected 7, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 4,
         "Failed. Expected 4, found %d instead.\n",
         value);

   // Test advanceIndices()
   batchIndexer->advanceIndices();
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Test advanceIndices() when wrapping around the last index
   batchIndexer->advanceIndices();
   // Reshuffled index lookup table is now {3, 2, 7, 6, 0, 5, 1, 4}.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 7,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);
}

void testGetSetRandomState() {
   int value;
   unsigned int seed;
   taus_uint4 rng;

   // Test MPI batch dimension == 1
   seed = 1439876414U;
   cl_random_init(&rng, 1, seed);
   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile1"),
         4, // Global batch size 4
         0, // This MPI block starts at batch element 0.
         4, // 4 batch elements in MPI block (therefore, 1 MPI block).
         8, // 8 files to batch across
         rng);
   batchIndexer->setWrapToStartIndex(false);

   // Test initial indices. Initial index lookup table is {4, 7, 5, 2, 1, 3, 6, 0}
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 4,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 7,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(2)) != 5,
         "Failed. Expected 5, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(3)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   batchIndexer->advanceIndices();
   batchIndexer->advanceIndices();
   // Reshuffled index lookup table is now {1, 4, 6, 5, 3, 7, 0, 2}.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 4,
         "Failed. Expected 4, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(2)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(3)) != 5,
         "Failed. Expected 5, found %d instead.\n",
         value);

   batchIndexer->setRandomState(rng);
   // Index lookup table once again should be {4, 7, 5, 2, 1, 3, 6, 0}
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 4,
         "setRandomState failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 7,
         "setRandomState failed. Expected 4, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(2)) != 5,
         "setRandomState failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(3)) != 2,
         "setRandomState failed. Expected 5, found %d instead.\n",
         value);
   auto rngCheck = batchIndexer->getRandomState();
   bool match = rngCheck.s0 == rng.s0;
   match &= rngCheck.state.s1 == rng.state.s1;
   match &= rngCheck.state.s2 == rng.state.s2;
   match &= rngCheck.state.s3 == rng.state.s3;
   FatalIf(
         !match,
         "getRandomState failed. Expected {%u, {%u, %u, %u}}, found {%u, {%u, %u, %u}} instead.\n",
         rng.s0, rng.state.s1, rng.state.s2, rng.state.s3,
         rngCheck.s0, rngCheck.state.s1, rngCheck.state.s2, rngCheck.state.s3);
}

int main(int argc, char **argv) {
   InfoLog() << "Testing BYFILE: ";
   testByFile();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing BYLIST: ";
   testByList();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing BYSPECIFIED: ";
   testBySpecified();
   InfoLog() << "Completed.\n";
   
   InfoLog() << "Testing RANDOM: ";
   testRandom();
   InfoLog() << "Completed.\n";
   
   InfoLog() << "Testing getRandomState/setRandomState: ";
   testGetSetRandomState();
   InfoLog() << "Completed.\n";

   InfoLog() << "BatchIndexer tests completed successfully!\n";
   return EXIT_SUCCESS;
}
