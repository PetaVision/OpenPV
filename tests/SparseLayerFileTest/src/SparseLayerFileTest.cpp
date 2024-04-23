/*
 * SparseLayerFileTest.cpp
 *
 */

#include "columns/PV_Init.hpp"
#include "columns/Random.hpp"
#include "components/LayerGeometry.hpp" // setLocalLayerLocFields()
#include "io/FileManager.hpp"
#include "io/SparseLayerFile.hpp"
#include "structures/SparseList.hpp"
#include "utils/BufferUtilsMPI.hpp"     // gather, scatter
#include "utils/BufferUtilsPvp.hpp"     // struct ActivityHeader
#include "utils/cl_random.h"            // random number generator

#include <cstdlib> // system()
#include <ios>     // ios_base openmodes
#include <memory>  // unique_ptr
#include <string>  // std::string

unsigned int const frame1Seed = 1280447331U;
unsigned int const frame2Seed = 1665927470U;

using namespace PV;

void applyDefaultOutputPath(PV_Init &pv_init_obj);
int run(
      std::shared_ptr<FileManager> fileManager,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

int checkHeader(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      BufferUtils::ActivityHeader const &correctHeader);

int checkFrame1Contents(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double correctTimestamp,
      BufferUtils::ActivityHeader const &correctHeader,
      std::vector<SparseList<float>> const &correctData,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

BufferUtils::ActivityHeader createInitialHeader(
      PVLayerLoc const &layerLoc,
      std::shared_ptr<MPIBlock const> &mpiBlock,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

PVLayerLoc createLayerLoc(PV_Init const & pv_init_obj, int xMargin, int yMargin);

std::shared_ptr<MPIBlock> createMPIBlock(PV_Init const &pv_init_obj);

int deleteOldOutputDirectory(std::string const &outputDir);

std::vector<SparseList<float>> generateCorrectFileDataFrame1(
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      std::shared_ptr<MPIBlock const> &mpiBlock);

std::vector<SparseList<float>> generateGlobalSparseLayerData(
      unsigned int seed,
      PVLayerLoc const &layerLoc);

std::vector<SparseList<float>> generateLayerDataFrame(
      PVLayerLoc const &layerLoc, bool dataExtendedFlag, unsigned int seed);

std::vector<SparseList<float>> readFrame2(
      PVLayerLoc const &layerLoc,
      std::string const &filename,
      std::shared_ptr<FileManager> fileManager,
      double &timestamp,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

void setLogFile(char const *logfile);

int verifyRead(
      std::vector<SparseList<float>> const &dataFromLayer,
      std::vector<SparseList<float>> const &dataFromFile,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

int writeFrameToFileStream(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<SparseList<float>> writeData,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

int writeFrameToSparseLayerFile(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<SparseList<float>> writeData,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   PV_Init pv_init_obj(&argc, &argv, false /* do not allow extra arguments */);
   applyDefaultOutputPath(pv_init_obj);

   auto *communicator = pv_init_obj.getCommunicator();
   auto mpiBlock      = communicator->getIOMPIBlock();
   std::string baseDirectory = pv_init_obj.getStringArgument("OutputPath");
   auto fileManager = std::make_shared<FileManager>(mpiBlock, baseDirectory);

   // Delete old output directory, to start with a clean slate.
   int deleteStatus = 0;
   for (int rank = 0; rank < communicator->globalCommSize(); rank++) {
      if (rank == communicator->globalCommRank() and fileManager->isRoot()) {
         pvAssert(!baseDirectory.empty());
         deleteStatus = deleteOldOutputDirectory(baseDirectory) != PV_SUCCESS;
      }
      MPI_Barrier(communicator->globalCommunicator());
   }
   MPI_Allreduce(MPI_IN_PLACE, &deleteStatus, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
   status = deleteStatus ? PV_FAILURE : PV_SUCCESS;

   // Create LayerLoc structure with local and global layer dimensions
   PVLayerLoc layerLoc = createLayerLoc(pv_init_obj, 0, 0);

   if (status == PV_SUCCESS) {
      for (int halo = 0; halo <= 2; halo += 2) {
         layerLoc.halo.lt = layerLoc.halo.rt = layerLoc.halo.dn = layerLoc.halo.up = halo;
         for (int d = 0; d <= 1; ++d) {
            for (int f = 0; f <= 1; ++f) {
               bool dataExtendedFlag = d != 0;
               bool fileExtendedFlag = f != 0;
               InfoLog() << "Running with dataExtendedFlag = " << dataExtendedFlag <<
                          ", fileExtendedFlag = " << fileExtendedFlag <<
                          ", halo = " << halo << "\n";
               std::string runDirectory(baseDirectory+"/");
               runDirectory.append(dataExtendedFlag ? "dataExt_" : "dataRes_");
               runDirectory.append(fileExtendedFlag ? "fileExt_" : "fileRes_");
               runDirectory.append("Halo").append(std::to_string(halo));
               fileManager = std::make_shared<FileManager>(mpiBlock, runDirectory);
               int status1 = run(fileManager, layerLoc, dataExtendedFlag, fileExtendedFlag);
               if (status1) { status = PV_FAILURE; }
            }
         }
      }
   }
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   else {
      Fatal() << "Test failed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void applyDefaultOutputPath(PV_Init &pv_init_obj) {
   std::string outputPathArg("OutputPath");
   std::string outputPath = pv_init_obj.getStringArgument(outputPathArg);

   if (outputPath.empty()) {
      outputPath = "output";
      pv_init_obj.setStringArgument(outputPathArg, outputPath);
   }
}

int run(
      std::shared_ptr<FileManager> fileManager,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   int status = PV_SUCCESS;

   std::string filename("testSparseLayerFile.pvp");

   // Create the MPIBlock for the run
   auto mpiBlock = fileManager->getMPIBlock();
   fileManager->ensureDirectoryExists("."); // "." is relative to FileManager's baseDir.

   // Create correct header for comparison with file contents after writing
   BufferUtils::ActivityHeader correctHeader =
         createInitialHeader(layerLoc, mpiBlock, dataExtendedFlag, fileExtendedFlag);

   fileManager->ensureDirectoryExists("."); // "." is relative to FileManager's baseDir.

   std::vector<SparseList<float>> layerData;

   // Generate layer data
   double timestamp = 5.0;
   layerData = generateLayerDataFrame(layerLoc, dataExtendedFlag, frame1Seed);

   // Write layer data using SparseLayerFile.
   InfoLog() << "Creating test SparseLayerFile...\n";
   status = writeFrameToSparseLayerFile(
       layerLoc, filename, fileManager, timestamp, layerData, dataExtendedFlag, fileExtendedFlag);
   if (status != PV_SUCCESS) { return status; }

   // Read back the file outside of the LayerFile framework, to verify its contents
   InfoLog() << "Verifying header contents...\n";
   correctHeader.nBands = layerLoc.nbatch * mpiBlock->getBatchDimension();
   status = checkHeader(layerLoc, filename, fileManager, correctHeader);
   if (status != PV_SUCCESS) { return status; }

   InfoLog() << "Verifying LayerFile write...\n";
   std::vector<SparseList<float>> correctData =
         generateCorrectFileDataFrame1(layerLoc, dataExtendedFlag, fileExtendedFlag, mpiBlock);

   status = checkFrame1Contents(
         layerLoc,
         filename,
         fileManager,
         timestamp,
         correctHeader,
         correctData,
         dataExtendedFlag,
         fileExtendedFlag);
   if (status != PV_SUCCESS) { return status; }

   // Generate new data for the second read/write check.
   timestamp = 8.0;
   layerData = generateLayerDataFrame(layerLoc, dataExtendedFlag, frame2Seed);

   // Write data outside of the LayerFile framework, to verify reading from a file works.
   InfoLog() << "Writing layer data (2)...\n";
   status = writeFrameToFileStream(
       layerLoc, filename, fileManager, timestamp, layerData, dataExtendedFlag, fileExtendedFlag);
   if (status != PV_SUCCESS) { return status; }

   // Read data back using LayerFile.
   InfoLog() << "Verifying LayerFile read...\n";
   std::vector<SparseList<float>> frame2 = readFrame2(
         layerLoc, filename, fileManager, timestamp, dataExtendedFlag, fileExtendedFlag);
   status = verifyRead(layerData, frame2, layerLoc, dataExtendedFlag, fileExtendedFlag);
   if (status != PV_SUCCESS) { return status; }

   return status;
}

int checkHeader(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      BufferUtils::ActivityHeader const &correctHeader) {
   auto mpiBlock = fileManager->getMPIBlock();
   bool isRoot    = mpiBlock->getRank() == 0;

   auto headerFile = fileManager->open(path, std::ios_base::in);
   FatalIf(
         isRoot and !headerFile,
         "FileManager failed to open \"%s\" on global rank %d: %s\n",
         path.c_str(),
         mpiBlock->getGlobalRank(),
         strerror(errno));
   FatalIf(
         !isRoot and headerFile,
         "FileManager) opened file \"%s\" on global rank %d, which is not a root process.\n",
         path.c_str(),
         mpiBlock->getGlobalRank());

   int status = PV_SUCCESS;
   if (headerFile) {
      long headerSize = static_cast<long>(sizeof(BufferUtils::ActivityHeader));
      BufferUtils::ActivityHeader headerData;
      FatalIf(
            headerSize != 80L, "sizeof(ActivityHeader) is %ld, not the expected 80.\n", headerSize);
      headerFile->read(&headerData, static_cast<long>(sizeof(BufferUtils::ActivityHeader)));

      if (headerData.headerSize != correctHeader.headerSize) { status = PV_FAILURE; }
      if (headerData.numParams  != correctHeader.numParams ) { status = PV_FAILURE; }
      if (headerData.fileType   != correctHeader.fileType  ) { status = PV_FAILURE; }
      if (headerData.nx         != correctHeader.nx        ) { status = PV_FAILURE; }
      if (headerData.ny         != correctHeader.ny        ) { status = PV_FAILURE; }
      if (headerData.nf         != correctHeader.nf        ) { status = PV_FAILURE; }
      if (headerData.numRecords != correctHeader.numRecords) { status = PV_FAILURE; }
      if (headerData.recordSize != correctHeader.recordSize) { status = PV_FAILURE; }
      if (headerData.dataSize   != correctHeader.dataSize  ) { status = PV_FAILURE; }
      if (headerData.dataType   != correctHeader.dataType  ) { status = PV_FAILURE; }
      if (headerData.nxProcs    != correctHeader.nxProcs   ) { status = PV_FAILURE; }
      if (headerData.nyProcs    != correctHeader.nyProcs   ) { status = PV_FAILURE; }
      if (headerData.nxExtended != correctHeader.nxExtended) { status = PV_FAILURE; }
      if (headerData.nyExtended != correctHeader.nyExtended) { status = PV_FAILURE; }
      if (headerData.kx0        != correctHeader.kx0       ) { status = PV_FAILURE; }
      if (headerData.ky0        != correctHeader.ky0       ) { status = PV_FAILURE; }
      if (headerData.nBatch     != correctHeader.nBatch    ) { status = PV_FAILURE; }
      if (headerData.nBands     != correctHeader.nBands    ) { status = PV_FAILURE; }
      if (headerData.timestamp  != correctHeader.timestamp ) { status = PV_FAILURE; }

      if (status != PV_SUCCESS) {
         ErrorLog().printf("checkHeader() found incorrect values for \"%s\".\n", path.c_str());
      }
   }

   return status;
}

int checkFrame1Contents(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double correctTimestamp,
      BufferUtils::ActivityHeader const &correctHeader,
      std::vector<SparseList<float>> const &correctData,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   int status = PV_SUCCESS;
   status = checkHeader(layerLoc, path, fileManager, correctHeader);
   if (status != PV_SUCCESS) { return status; }

   auto mpiBlock = fileManager->getMPIBlock();
   auto sparseLayerStream = fileManager->open(path, std::ios_base::in);
   FatalIf(
         fileManager->isRoot() and !sparseLayerStream,
         "FileManager) failed to open \"%s\" on global rank %d: %s\n",
         path.c_str(),
         mpiBlock->getGlobalRank(),
         strerror(errno));
   FatalIf(
         !fileManager->isRoot() and sparseLayerStream,
         "FileManager) opened file \"%s\" on global rank %d, which is not a root process.\n",
         path.c_str(),
         mpiBlock->getGlobalRank());
   if (!sparseLayerStream) { return status; }

   int expectedBatchWidth = layerLoc.nbatch * mpiBlock->getBatchDimension();
   int expectedNx         = layerLoc.nx * mpiBlock->getNumColumns();
   int expectedNy         = layerLoc.ny * mpiBlock->getNumRows();
   int expectedNf         = layerLoc.nf;
   if (dataExtendedFlag and fileExtendedFlag) {
      expectedNx += layerLoc.halo.lt + layerLoc.halo.rt;
      expectedNy += layerLoc.halo.dn + layerLoc.halo.up;
   }
   pvAssert(correctHeader.nx == expectedNx);
   pvAssert(correctHeader.ny == expectedNy);
   pvAssert(correctHeader.nf == expectedNf);
   // Batch width not stored in file header, so we have to trust expectedBatchWidth.

   long headerSize = static_cast<long>(sizeof(correctHeader));
   sparseLayerStream->setInPos(headerSize, std::ios_base::beg);

   FatalIf(
         correctData.size() != static_cast<std::size_t>(expectedBatchWidth),
         "checkFrame1Contents(): correctData argument has length %zu, "
         "but layerLoc argument indicates length %d\n",
         correctData.size(), expectedBatchWidth);
   for (int b = 0; b < expectedBatchWidth; ++b) {
      FatalIf(
            expectedNx != correctData[b].getWidth(),
            "Global rank %d, file \"%s\", batch element %d: "
            "correctData list has width %d but from layerLoc it should be %d\n",
            mpiBlock->getGlobalRank(), path.c_str(), b,
            correctData[b].getWidth(), expectedNx);
      FatalIf(
            expectedNy != correctData[b].getHeight(),
            "Global rank %d, file \"%s\", batch element %d: "
            "correctData list has height %d but from layerLoc it should be %d\n",
            mpiBlock->getGlobalRank(), path.c_str(), b,
            correctData[b].getHeight(), expectedNy);
      FatalIf(
            expectedNf != correctData[b].getFeatures(),
            "Global rank %d, file \"%s\", batch element %d: "
            "correctData list has %d features but from layerLoc there should be %d\n",
            mpiBlock->getGlobalRank(), path.c_str(), b,
            correctData[b].getFeatures(), expectedNf);
   }

   for (int b = 0; b < expectedBatchWidth; ++b) {
      double fileTimestamp;
      sparseLayerStream->read(&fileTimestamp, sizeof(fileTimestamp));
      if (fileTimestamp != correctTimestamp) {
         ErrorLog().printf(
               "Global rank %d, file \"%s\", batch element %d: "
               "timestamp is %f instead of the expected %f\n",
               mpiBlock->getGlobalRank(), path.c_str(), b,
               fileTimestamp, correctTimestamp);
         status = PV_FAILURE;
      }
      uint32_t numNonzeroValues;
      sparseLayerStream->read(&numNonzeroValues, static_cast<long>(sizeof(numNonzeroValues)));

      std::vector<SparseList<float>::Entry> fileContents(numNonzeroValues);
      long dataSize       = static_cast<long>(sizeof(SparseList<float>::Entry));
      long frameSizeBytes = static_cast<long>(numNonzeroValues) * dataSize;
      sparseLayerStream->read(fileContents.data(), frameSizeBytes);

      auto correctContents = correctData[b].getContents();
      FatalIf(numNonzeroValues != static_cast<uint32_t>(correctContents.size()),
            "Global rank %d, file \"%s\", batch element %d: "
            "Frame indicates %u nonzero values but there should be %u.\n",
            mpiBlock->getGlobalRank(), path.c_str(), b,
            static_cast<unsigned>(numNonzeroValues), static_cast<unsigned>(correctContents.size()));
            
      for (uint32_t k = 0; k < numNonzeroValues; ++k) {
         SparseList<float>::Entry fileEntry    = fileContents.at(k);
         SparseList<float>::Entry correctEntry = correctContents.at(k);
         if (fileEntry.index != correctEntry.index and fileEntry.value != correctEntry.value) {
            ErrorLog().printf(
                  "Global rank %d, file \"%s\", batch element %d: Nonzero value %d of %d "
                  "has index %d, value %f instead of the expected index %u, value %f\n",
                  mpiBlock->getGlobalRank(), path.c_str(), b,
                  static_cast<unsigned>(k), static_cast<unsigned>(numNonzeroValues), 
                  fileEntry.index, (double)fileEntry.value,
                  correctEntry.index, (double)correctEntry.value);
            status = PV_FAILURE;
         }
      }
   }

   return status;
}

BufferUtils::ActivityHeader createInitialHeader(
      PVLayerLoc const &layerLoc,
      std::shared_ptr<MPIBlock const> &mpiBlock,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   int nxBlock = layerLoc.nx * mpiBlock->getNumColumns();
   int nyBlock = layerLoc.ny * mpiBlock->getNumRows();
   if (dataExtendedFlag and fileExtendedFlag) {
      nxBlock += layerLoc.halo.lt + layerLoc.halo.rt;
      nyBlock += layerLoc.halo.dn + layerLoc.halo.up;
   }
   int recordSize = 0;
   int dataSize = static_cast<int>(sizeof(SparseList<float>::Entry));
   int dataType = BufferUtils::returnDataType<float>();

   BufferUtils::ActivityHeader headerData;
   headerData.headerSize  = 80;
   headerData.numParams   = 20;
   headerData.fileType    = PVP_ACT_SPARSEVALUES_FILE_TYPE;
   headerData.nx          = nxBlock;
   headerData.ny          = nyBlock;
   headerData.nf          = layerLoc.nf;
   headerData.numRecords  = 1;
   headerData.recordSize  = recordSize;
   headerData.dataSize    = dataSize;
   headerData.dataType    = dataType;
   headerData.nxProcs     = 1;
   headerData.nyProcs     = 1;
   headerData.nxExtended  = nxBlock;
   headerData.nyExtended  = nyBlock;
   headerData.kx0         = 0;
   headerData.ky0         = 0;
   headerData.nBatch      = 1;
   headerData.nBands      = 0;
   headerData.timestamp   = 0.0;
   // Note: nBatch == 1 is correct even though the LayerLoc specifies 4 batch elements.
   // Currently, nBatch is not used, so perhaps in the future we could use it to flag how many
   // batch elements there are; i.e. how many frames at a time should be read as a unit.

   return headerData;
}

PVLayerLoc createLayerLoc(PV_Init const &pv_init_obj, int xMargin, int yMargin) {
   PVLayerLoc layerLoc;
   layerLoc.nbatchGlobal = 4;
   layerLoc.nxGlobal     = 16;
   layerLoc.nyGlobal     = 16;
   layerLoc.nf           = 3;
   LayerGeometry::setLocalLayerLocFields(
         &layerLoc, pv_init_obj.getCommunicator(), std::string("testLayer"));
   layerLoc.halo.lt = xMargin;
   layerLoc.halo.rt = xMargin;
   layerLoc.halo.dn = yMargin;
   layerLoc.halo.up = yMargin;
   return layerLoc;
}

std::shared_ptr<MPIBlock> createMPIBlock(PV_Init const &pv_init_obj) {
   auto globalMPIBlock = pv_init_obj.getCommunicator()->getGlobalMPIBlock();
   auto globalComm     = globalMPIBlock->getComm();
   int numRows         = pv_init_obj.getIntegerArgument("NumRows");
   int numColumns      = pv_init_obj.getIntegerArgument("NumColumns");
   int batchDim        = pv_init_obj.getIntegerArgument("BatchWidth");
   int cellNumRows     = pv_init_obj.getIntegerArgument("CheckpointCellNumRows");
   int cellNumColumns  = pv_init_obj.getIntegerArgument("CheckpointCellNumColumns");
   int cellBatchDim    = pv_init_obj.getIntegerArgument("CheckpointCellBatchDimension");

   auto mpiBlock = std::make_shared<MPIBlock>(
         globalComm, numRows, numColumns, batchDim, cellNumRows, cellNumColumns, cellBatchDim);
   return mpiBlock;
}

int deleteOldOutputDirectory(std::string const &outputDir) {
   std::string systemCommand("rm -rf \"");
   systemCommand.append(outputDir).append("\"");
   int status = std::system(systemCommand.c_str());
   if (status) {
      ErrorLog() << "system command rm -fr \"" << outputDir << "\" returned " << status << "\n";
   }
   return status ? PV_FAILURE : PV_SUCCESS;
}

std::vector<SparseList<float>> generateCorrectFileDataFrame1(
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      std::shared_ptr<MPIBlock const> &mpiBlock) {
   if (mpiBlock->getRank() != 0) { return std::vector<SparseList<float>>(); }

   int blockBatchDimension = layerLoc.nbatch * mpiBlock->getBatchDimension();

   // Create the LayerLoc that would apply if all MPI processes in a block were on a single
   // MPI process
   PVLayerLoc blockLayerLoc;
   blockLayerLoc.nbatch       = blockBatchDimension;
   blockLayerLoc.nx           = layerLoc.nx * mpiBlock->getNumColumns();
   blockLayerLoc.ny           = layerLoc.ny * mpiBlock->getNumRows();
   blockLayerLoc.nf           = layerLoc.nf;
   blockLayerLoc.nbatchGlobal = layerLoc.nbatchGlobal;
   blockLayerLoc.nxGlobal     = layerLoc.nxGlobal;
   blockLayerLoc.nyGlobal     = layerLoc.nyGlobal;
   blockLayerLoc.kb0          = blockBatchDimension * mpiBlock->getStartBatch();
   blockLayerLoc.kx0          = layerLoc.nx * mpiBlock->getStartColumn();
   blockLayerLoc.ky0          = layerLoc.ny * mpiBlock->getStartRow();
   blockLayerLoc.halo.lt      = dataExtendedFlag ? layerLoc.halo.lt : 0;
   blockLayerLoc.halo.rt      = dataExtendedFlag ? layerLoc.halo.rt : 0;
   blockLayerLoc.halo.dn      = dataExtendedFlag ? layerLoc.halo.dn : 0;
   blockLayerLoc.halo.up      = dataExtendedFlag ? layerLoc.halo.up : 0;

   auto correctData = generateGlobalSparseLayerData(frame1Seed, blockLayerLoc);
   correctData.erase(correctData.begin(), correctData.begin() + blockLayerLoc.kb0);
   correctData.erase(correctData.begin() + blockLayerLoc.nbatch, correctData.end());
   auto correctSize = static_cast<std::vector<SparseList<float>>::size_type>(blockLayerLoc.nbatch);
   pvAssert(correctData.size() == correctSize);
   int blockWidth  = blockLayerLoc.nx;
   int blockHeight = blockLayerLoc.ny;
   if (dataExtendedFlag and fileExtendedFlag) {
      int globalExtWidth  = layerLoc.nxGlobal + layerLoc.halo.lt + layerLoc.halo.rt;
      int globalExtHeight = layerLoc.nyGlobal + layerLoc.halo.dn + layerLoc.halo.up;
      blockWidth += layerLoc.halo.lt + layerLoc.halo.rt;
      blockHeight += layerLoc.halo.dn + layerLoc.halo.up;
      for (auto &s : correctData) {
         s.grow(globalExtWidth, globalExtHeight, blockLayerLoc.halo.lt, blockLayerLoc.halo.up);
      }
   }
   for (auto &s : correctData) {
      s.crop(blockWidth, blockHeight, blockLayerLoc.kx0, blockLayerLoc.ky0);
   }
   return correctData;
}

std::vector<SparseList<float>> generateGlobalSparseLayerData(
      unsigned int seed,
      PVLayerLoc const &layerLoc) {
   taus_uint4 rng;
   cl_random_init(&rng, (std::size_t)1, seed);
   std::vector<SparseList<float>> result(layerLoc.nbatchGlobal);
   int value = 0;
   for (int b = 0; b < layerLoc.nbatchGlobal; ++b) {
      result[b].reset(layerLoc.nxGlobal, layerLoc.nyGlobal, layerLoc.nf);
      for (int y = 0; y < layerLoc.nyGlobal; ++y) {
         for (int x = 0; x < layerLoc.nxGlobal; ++x) {
            for (int f = 0; f < layerLoc.nf; ++f) {
               float u = cl_random_prob(rng = cl_random_get(rng));
               bool indexon = u < 0.025f;
               if (indexon) {
                  ++value;
                  int k = f + layerLoc.nf * (x + layerLoc.nxGlobal * y);
                  result[b].addEntry(k, static_cast<float>(value));
               }
            }
         }
      }
   }
   return result;
}

std::vector<SparseList<float>> generateLayerDataFrame(
      PVLayerLoc const &layerLoc, bool dataExtendedFlag, unsigned int seed) {
   auto sparseLayerData = generateGlobalSparseLayerData(seed, layerLoc);
   sparseLayerData.erase(sparseLayerData.begin(), sparseLayerData.begin() + layerLoc.kb0);
   sparseLayerData.erase(sparseLayerData.begin() + layerLoc.nbatch, sparseLayerData.end());
   pvAssert(sparseLayerData.size() == static_cast<std::vector<SparseList<float>>::size_type>(layerLoc.nbatch));
   int globalWidth  = layerLoc.nxGlobal;
   int globalHeight = layerLoc.nyGlobal;
   int localWidth   = layerLoc.nx;
   int localHeight  = layerLoc.ny;
   if (dataExtendedFlag) {
      globalWidth   += layerLoc.halo.lt + layerLoc.halo.rt;
      globalHeight  += layerLoc.halo.dn + layerLoc.halo.up;
      localWidth    += layerLoc.halo.lt + layerLoc.halo.rt;
      localHeight   += layerLoc.halo.dn + layerLoc.halo.up;
      int leftMargin = layerLoc.halo.lt;
      int topMargin  = layerLoc.halo.up;
      for (auto &s : sparseLayerData) {
         s.grow(globalWidth, globalHeight, leftMargin, topMargin);
      }
   }
   for (auto &s : sparseLayerData) {
      s.crop(localWidth, localHeight, layerLoc.kx0, layerLoc.ky0);
   }
   return sparseLayerData;
}

std::vector<SparseList<float>> readFrame2(
      PVLayerLoc const &layerLoc,
      std::string const &filename,
      std::shared_ptr<FileManager> fileManager,
      double &timestamp,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   SparseLayerFile sparseLayerFile(
         fileManager,
         filename,
         layerLoc,
         dataExtendedFlag,
         fileExtendedFlag,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   int positionIndex = 1; // Position index is zero-based, so this is the second frame
   sparseLayerFile.setIndex(positionIndex);

   std::vector<SparseList<float>> frame2Contents(layerLoc.nbatch);
   int width = layerLoc.nx;
   int height = layerLoc.ny;
   if (dataExtendedFlag) {
      width += layerLoc.halo.lt + layerLoc.halo.rt;
      height += layerLoc.halo.dn + layerLoc.halo.up;
   }

   for (int b = 0; b < layerLoc.nbatch; ++b) {
      frame2Contents[b].reset(width, height, layerLoc.nf);
      sparseLayerFile.setListLocation(&frame2Contents[b], b);
   }
   sparseLayerFile.read(timestamp);

   return frame2Contents;
}

void setLogFile(char const *logfile) {
   pvAssert(logfile);
   std::string logfileString(logfile);
   int globalRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
   if (globalRank != 0) {
      auto finalSlash      = logfileString.rfind('/');
      auto insertionPoint  = logfileString.rfind('.');
      if (finalSlash == std::string::npos) { finalSlash = 0; }
      if (insertionPoint == std::string::npos) { insertionPoint = logfileString.size(); }
      if (finalSlash > insertionPoint) { insertionPoint = logfileString.size(); }

      std::string insertion("_");
      insertion.append(std::to_string(globalRank));
      logfileString.insert(insertionPoint, insertion);
   }
   setLogFile(logfileString, std::ios_base::out);
}

int verifyRead(
      std::vector<SparseList<float>> const &dataFromLayer,
      std::vector<SparseList<float>> const &dataFromFile,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   int status = PV_SUCCESS;

   int layerBatchSize = static_cast<int>(dataFromLayer.size());
   int fileBatchSize = static_cast<int>(dataFromFile.size());
   FatalIf(
         layerBatchSize != layerLoc.nbatch,
         "verifyRead() failed: dataFromLayer has length %d instead of the expected %d\n",
         layerBatchSize,
         layerLoc.nbatch);
   FatalIf(
         fileBatchSize != layerLoc.nbatch,
         "verifyRead() failed: dataFromFile has length %d instead of the expected %d\n",
         fileBatchSize,
         layerLoc.nbatch);

   // Check dimensions of dataFromLayer[b].
   // Should be extended if dataExtended is true and restricted if false
   int correctWidth  = layerLoc.nx + (dataExtendedFlag ? layerLoc.halo.lt + layerLoc.halo.rt : 0);
   int correctHeight = layerLoc.ny + (dataExtendedFlag ? layerLoc.halo.dn + layerLoc.halo.up : 0);
   for (int b = 0; b < layerBatchSize; ++b) {
      auto layerBuffer  = dataFromLayer[b];
      if (layerBuffer.getWidth() != correctWidth) {
         ErrorLog().printf(
               "verifyRead(): batch element %d of dataFromFile has width %d instead of the expected %d.\n",
               b, layerBuffer.getWidth(), correctWidth);
         status = PV_FAILURE;
      }
      if (layerBuffer.getHeight() != correctHeight) {
         ErrorLog().printf(
               "verifyRead(): batch element %d of dataFromFile has height %d instead of the expected %d.\n",
               b, layerBuffer.getHeight(), correctHeight);
         status = PV_FAILURE;
      }
      if (layerBuffer.getFeatures() != layerLoc.nf) {
         ErrorLog().printf(
               "verifyRead(): batch element %d of dataFromFile has %d features instead of the expected %d.\n",
               b, layerBuffer.getFeatures(), layerLoc.nf);
         status = PV_FAILURE;
      }
   }

   // Check dimensions of dataFromFile[b].
   // Should be extended if dataExtended is true and restricted if false
   for (int b = 0; b < layerBatchSize; ++b) {
      auto fileBuffer  = dataFromFile[b];
      if (fileBuffer.getWidth() != correctWidth) {
         ErrorLog().printf(
               "verifyRead(): batch element %d of dataFromFile has width %d instead of the expected %d.\n",
               b, fileBuffer.getWidth(), correctWidth);
         status = PV_FAILURE;
      }
      if (fileBuffer.getHeight() != correctHeight) {
         ErrorLog().printf(
               "verifyRead(): batch element %d of dataFromFile has height %d instead of the expected %d.\n",
               b, fileBuffer.getHeight(), correctHeight);
         status = PV_FAILURE;
      }
      if (fileBuffer.getFeatures() != layerLoc.nf) {
         ErrorLog().printf(
               "verifyRead(): batch element %d of dataFromFile has %d features instead of the expected %d.\n",
               b, fileBuffer.getFeatures(), layerLoc.nf);
         status = PV_FAILURE;
      }
   }

   if (status != PV_SUCCESS) { return status; }

   // Compare values in dataFromLayer and dataFromFile.
   // If dataExtendedFlag is true and fileExtendedFlag is false, we need to crop to the restricted size.
   for (int b = 0; b < layerBatchSize; ++b) {
      auto layerSparseList = dataFromLayer[b];
      auto fileSparseList = dataFromFile[b];
      if (dataExtendedFlag and !fileExtendedFlag) {
         layerSparseList.crop(layerLoc.nx, layerLoc.ny, layerLoc.halo.lt, layerLoc.halo.up);
         fileSparseList.crop(layerLoc.nx, layerLoc.ny, layerLoc.halo.lt, layerLoc.halo.up);
      }
      auto layerContents = layerSparseList.getContents();
      auto fileContents  = fileSparseList.getContents();
      if (layerContents.size() != fileContents.size()) {
         ErrorLog().printf(
               "verifyRead(), batch element %d: "
               "layer data has %zu nonzero values but file data has %zu.\n",
               b, layerContents.size(), fileContents.size());
         status = PV_FAILURE;
      }
      if (status == PV_SUCCESS) {
         int numElements = static_cast<int>(layerContents.size());
         for (int k = 0; k < numElements; ++k) {
            if (layerContents.at(k).index != fileContents.at(k).index) {
               ErrorLog().printf(
                     "    nonzero index %d of layer data is %u; in file data it is %u\n",
                     k, (unsigned)layerContents.at(k).index, (unsigned)fileContents.at(k).index);
            }
            else if (layerContents.at(k).value != fileContents.at(k).value) {
               ErrorLog().printf(
                     "    nonzero value %d of layer data is %f; in file data it is %f\n",
                     k, (double)layerContents.at(k).value, (double)fileContents.at(k).value);
            }
         }
      }
   }

   return status;
}

int writeFrameToFileStream(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<SparseList<float>> writeData,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   int status = PV_SUCCESS;
   auto mpiBlock = fileManager->getMPIBlock();

   int dataNx = layerLoc.nx;
   int dataNy = layerLoc.ny;
   int dataNf = layerLoc.nf;
   if (dataExtendedFlag) {
      dataNx += layerLoc.halo.lt + layerLoc.halo.rt;
      dataNy += layerLoc.halo.dn + layerLoc.halo.up;
   }
   int const numValues = dataNx * dataNy * dataNf;
   long const numValuesAcrossBatch = numValues * static_cast<long>(layerLoc.nbatch);
   FatalIf(
         static_cast<int>(writeData.size()) != layerLoc.nbatch,
         "Global rank %d, file \"%s\", "
         "writeFrameToFileStream() writeData has %d batch elements instead of the expected %zu.\n",
         mpiBlock->getGlobalRank(), path.c_str(), numValuesAcrossBatch, writeData.size());
   for (int b = 0; b < layerLoc.nbatch; ++b) {
      if (static_cast<long>(writeData[b].getWidth()) != dataNx) {
         ErrorLog().printf(
               "Global rank %d, file \"%s\", writeFrameToFileStream() writeData "
               "batch element %d has width %d instead of the expected %d\n",
               mpiBlock->getGlobalRank(), path.c_str(), b, writeData[b].getWidth(), dataNx);
         status = PV_FAILURE;
      }
      if (static_cast<long>(writeData[b].getHeight()) != dataNy) {
         ErrorLog().printf(
               "Global rank %d, file \"%s\", writeFrameToFileStream() writeData "
               "batch element %d has height %d instead of the expected %d\n",
               mpiBlock->getGlobalRank(), path.c_str(), b, writeData[b].getHeight(), dataNy);
         status = PV_FAILURE;
      }
      if (static_cast<long>(writeData[b].getFeatures()) != dataNf) {
         ErrorLog().printf(
               "Global rank %d, file \"%s\", writeFrameToFileStream() writeData "
               "has %d features instead of the expected %d\n",
               mpiBlock->getGlobalRank(), path.c_str(), b, writeData[b].getFeatures(), dataNf);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) { return status; }

   int rootProcess = fileManager->getRootProcessRank();
   if (fileManager->isRoot()) {
      int fileNx = layerLoc.nx * mpiBlock->getNumColumns();
      int fileNy = layerLoc.ny * mpiBlock->getNumRows();
      int fileNf = layerLoc.nf;

      if (dataExtendedFlag and fileExtendedFlag) {
         fileNx += layerLoc.halo.lt + layerLoc.halo.rt;
         fileNy += layerLoc.halo.dn + layerLoc.halo.up;
      }
      std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out | std::ios_base::binary;
      auto fileStream = fileManager->open(path, mode);
      fileStream->setOutPos(0L, std::ios_base::end);
      for (int m = 0; m < mpiBlock->getBatchDimension(); ++m) {
         for (int b = 0; b < layerLoc.nbatch; ++b) {
            // de-sparsify the SparseList, gather, and then re-sparsify
            Buffer<float> localBuffer(
                  writeData[b].getWidth(),
                  writeData[b].getHeight(),
                  writeData[b].getFeatures());
            writeData[b].toBuffer(localBuffer, 0.0f);
            auto gatheredBuffer = BufferUtils::gather<float>(
                  mpiBlock, localBuffer, layerLoc.nx, layerLoc.ny, m, rootProcess);
            if (dataExtendedFlag and !fileExtendedFlag) {
               gatheredBuffer.crop(fileNx, fileNx, Buffer<float>::CENTER);
            }
            SparseList<float> gatheredList;
            gatheredList.fromBuffer(gatheredBuffer, 0.0f);

            fileStream->write(&timestamp, 8L);
            pvAssert(gatheredList.getWidth() == fileNx);
            pvAssert(gatheredList.getHeight() == fileNy);
            pvAssert(gatheredList.getFeatures() == fileNf);
            std::vector<SparseList<float>::Entry> gatheredContents = gatheredList.getContents();
            uint32_t gatheredListLength = static_cast<uint32_t>(gatheredContents.size());
            fileStream->write(&gatheredListLength, sizeof(gatheredListLength));
            long dataSize               = static_cast<long>(sizeof(SparseList<float>::Entry));
            long writeLength            = dataSize * static_cast<long>(gatheredListLength);
            
            fileStream->write(gatheredContents.data(), writeLength);
         }
      }

      // Update NBands
      fileStream->setInPos(68L, std::ios_base::beg);
      uint32_t nBands;
      fileStream->read(&nBands, 4L);
      int numNewBands = layerLoc.nbatch * mpiBlock->getBatchDimension();
      nBands += static_cast<uint32_t>(numNewBands);
      fileStream->setOutPos(68L, std::ios_base::beg);
      fileStream->write(&nBands, 4L);
   }
   else {
      for (int b = 0; b < layerLoc.nbatch; ++b) {
         // de-sparsify the SparseList, gather, and then re-sparsify
         Buffer<float> localBuffer(
               writeData[b].getWidth(),
               writeData[b].getHeight(),
               writeData[b].getFeatures());
         writeData[b].toBuffer(localBuffer, 0.0f);
         int blockBatchIndex = mpiBlock->getBatchIndex();
         auto gatheredBuffer = BufferUtils::gather<float>(
               mpiBlock, localBuffer, layerLoc.nx, layerLoc.ny, blockBatchIndex, rootProcess);
      }
   }
   return status;
}

int writeFrameToSparseLayerFile(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<SparseList<float>> writeData,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   // Create the pointer for the LayerFile object.
   std::unique_ptr<SparseLayerFile> sparseLayerFile;

   // Create the file in write mode.
   sparseLayerFile = std::unique_ptr<SparseLayerFile>(new SparseLayerFile(
        fileManager,
        path,
        layerLoc,
        dataExtendedFlag,
        fileExtendedFlag,
        false /* readOnlyFlag */,
        false /* clobberFlag */,
        false /* verifyWrites */));

   // Write layer data
   for (int b = 0; b < layerLoc.nbatch; ++b) {
      sparseLayerFile->setListLocation(&writeData[b], b);
   }
   sparseLayerFile->write(timestamp);

   return PV_SUCCESS;
}
