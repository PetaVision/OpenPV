#include "components/BatchIndexer.hpp"
#include "utils/PVLog.hpp"

#include <memory>
#include <vector>

using PV::BatchIndexer;

// BatchIndexer::BatchIndexer(int globalBatchCount, int globalBatchIndex, int batchWidth, int
// fileCount, enum BatchMethod batchMethod)
// BatchIndexer::nextIndex(int localBatchIndex)
void testByFile() {
   int value = 0;

   // Test batchWidth == 1

   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile1"),
         2, // Global batch size 2
         0, // This MPI block starts at batch element 0.
         2, // 2 batch elements in MPI block (therefore, 1 MPI block).
         4, // 4 files to batch across
         BatchIndexer::BYFILE,
         false /*initializeFromCheckpointFlag*/);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->initializeBatch(0);
   batchIndexer->initializeBatch(1);

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

   // Test nextIndex()
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test nextIndex() when wrapping around the last index
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);

   // Test batchWidth > 1

   batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByFile2"),
         4, // Global batch size 4
         2, // This MPI block starts at batch element 2.
         2, // 2 batch elements in MPI block (therefore, 2 MPI blocks and this is the second one)
         8, // 8 files to batch across
         BatchIndexer::BYFILE,
         false /*initializeFromCheckpointFlag*/);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->initializeBatch(0);
   batchIndexer->initializeBatch(1);

   // Test initial indices. The first call to nextIndex after
   // initializeBatch just returns their initial value.
   FatalIf(
         (value = batchIndexer->getIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->getIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test nextIndex()
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 7,
         "Failed. Expected 7, found %d instead.\n",
         value);

   // Test nextIndex() when wrapping around the last index
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);
}

// BatchIndexer::BatchIndexer(int globalBatchCount, int globalBatchIndex, int batchWidth, int
// fileCount, enum BatchMethod batchMethod)
// BatchIndexer::nextIndex(int localBatchIndex)
void testByList() {
   int value = 0;

   // Test batchWidth == 1

   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByList1"),
         2, // Global batch size 2
         0, // This MPI block starts at batch element 0.
         2, // 2 batch elements in MPI block (therefore, 1 MPI block).
         4, // 4 files to batch across
         BatchIndexer::BYLIST,
         false /*initializeFromCheckpointFlag*/);
   batchIndexer->setWrapToStartIndex(true);
   batchIndexer->initializeBatch(0);
   batchIndexer->initializeBatch(1);

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

   // Test nextIndex()
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 1,
         "Failed. Expected 1, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);

   // Test nextIndex() when wrapping around the last index.
   // Because setWrapToStartIndex is true, these should be
   // the same as the initial state, not 0.

   // Batch 0 won't loop, it's going to march
   // right into where batch 1 started.
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Batch 1 should loop.
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Rewind our indices, try going over the loop again.
   // Since we disabled setWrapToStartIndex, we should
   // land on index 0
   batchIndexer->setIndices(indicesBeforeLoop);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->nextIndex(1);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);

   // Test batchWidth > 1

   batchIndexer = std::make_shared<BatchIndexer>(
         std::string("ByList2"),
         4, // Global batch size 4
         2, // This MPI block starts at batch element 2
         2, // 2 batch elements in MPI block (therefore, 2 MPI blocks).
         8, // 8 files to batch across
         BatchIndexer::BYLIST,
         false /*initializeFromCheckpointFlag*/);
   batchIndexer->setWrapToStartIndex(true);
   batchIndexer->initializeBatch(0);
   batchIndexer->initializeBatch(1);

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

   // Test nextIndex()
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 5,
         "Failed. Expected 5, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 7,
         "Failed. Expected 7, found %d instead.\n",
         value);

   // Test nextIndex() when wrapping around the last index.
   // Because setWrapToStartIndex is true, these should be
   // the same as the initial state, not 0.

   // Batch 0 won't loop, it's going to march
   // right into where batch 1 started.
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);

   // Batch 1 should loop.
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 6,
         "Failed. Expected 6, found %d instead.\n",
         value);

   // Rewind our indices, try going over the loop again.
   // Since we disabled setWrapToStartIndex, we should
   // land on index 0
   batchIndexer->setIndices(indicesBeforeLoop);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->nextIndex(1);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
}

// BatchIndexer::BatchIndexer(int globalBatchCount, int globalBatchIndex, int batchWidth, int
// fileCount, enum BatchMethod batchMethod)
// BatchIndexer::nextIndex(int localBatchIndex)
void testBySpecified() {
   int value                                  = 0;
   std::shared_ptr<BatchIndexer> batchIndexer = std::make_shared<BatchIndexer>(
         std::string("BySpecified"),
         2, // Global batch size 2
         0, // This MPI block starts at batch element 0
         2, // 2 batch elements in MPI block (therefore, 1 MPI block)
         4, // 4 files to batch across
         BatchIndexer::BYSPECIFIED,
         false /*initializeFromCheckpointFlag*/);
   batchIndexer->setWrapToStartIndex(true);
   batchIndexer->specifyBatching(0, 2, 1); // Start at 2, increment by 1
   batchIndexer->specifyBatching(1, 0, 2); // Start at 0, increment by 2
   batchIndexer->initializeBatch(0);
   batchIndexer->initializeBatch(1);

   // Test initial indices. The first call to nextIndex after
   // initializeBatch just returns their initial value.
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

   // Test nextIndex()
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 3,
         "Failed. Expected 3, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);

   // Test nextIndex() when wrapping around the last index.
   // Because setWrapToStartIndex is true, these should be
   // the same as the initial state, not 0.

   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 2,
         "Failed. Expected 2, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);

   // Rewind our indices, try going over the loop again.
   // Since we disabled setWrapToStartIndex, we should
   // land on index 0 for both
   batchIndexer->setIndices(indicesBeforeLoop);
   batchIndexer->setWrapToStartIndex(false);
   batchIndexer->nextIndex(0);
   batchIndexer->nextIndex(1);
   FatalIf(
         (value = batchIndexer->nextIndex(0)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
   FatalIf(
         (value = batchIndexer->nextIndex(1)) != 0,
         "Failed. Expected 0, found %d instead.\n",
         value);
}

int main(int argc, char **argv) {
   InfoLog() << "Testing BatchIndexer::BYFILE: ";
   testByFile();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing BatchIndexer::BYLIST: ";
   testByList();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing BatchIndexer::BYSPECIFIED: ";
   testBySpecified();
   InfoLog() << "Completed.\n";

   InfoLog() << "BatchIndexer tests completed successfully!\n";
   return EXIT_SUCCESS;
}
