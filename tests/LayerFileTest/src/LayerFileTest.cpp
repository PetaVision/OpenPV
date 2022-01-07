/*
 * LayerFileTest.cpp
 *
 */

#include "columns/PV_Init.hpp"
#include "components/LayerGeometry.hpp" // setLocalLayerLocFields()
#include "io/FileManager.hpp"
#include "io/LayerFile.hpp"
#include "utils/BufferUtilsMPI.hpp"     // gather, scatter
#include "utils/BufferUtilsPvp.hpp"     // struct ActivityHeader

#include <cstdlib> // system()
#include <ios>     // ios_base openmodes
#include <memory>  // shared_ptr, unique_ptr
#include <string>  // std::string

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
      std::vector<Buffer<float>> const &correctData,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

BufferUtils::ActivityHeader createInitialHeader(
      PVLayerLoc const &layerLoc,
      std::shared_ptr<MPIBlock const> &mpiBlock,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

PVLayerLoc createLayerLoc(PV_Init const & pv_init_obj, int xMargin, int yMargin);

int deleteOldOutputDirectory(std::string const &outputDir);

std::vector<Buffer<float>> generateCorrectFileDataFrame1(
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      std::shared_ptr<MPIBlock const> &mpiBlock);

std::vector<Buffer<float>> generateLayerDataFrame1(
      PVLayerLoc const &layerLoc, bool dataExtendedFlag);

std::vector<Buffer<float>> generateLayerDataFrame2(
      PVLayerLoc const &layerLoc, bool dataExtendedFlag);

std::vector<Buffer<float>> readFrame2(
      PVLayerLoc const &layerLoc,
      std::string const &filename,
      std::shared_ptr<FileManager const> fileManager,
      double &timestamp,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

void setLogFile(char const *logfile);

int verifyRead(
      std::vector<Buffer<float>> const &dataFromLayer,
      std::vector<Buffer<float>> const &dataFromFile,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

int writeFrameToFileStream(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<Buffer<float>> writeData,
      bool dataExtendedFlag,
      bool fileExtendedFlag);

int writeFrameToLayerFile(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<Buffer<float>> writeData,
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
   if (fileManager->isRoot()) {
      pvAssert(!baseDirectory.empty());
      status = deleteOldOutputDirectory(baseDirectory);
      if (status != PV_SUCCESS) { return status; }
   }
   MPI_Barrier(mpiBlock->getGlobalComm());

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

   std::string filename("testLayerFile.pvp");

   // Create the MPIBlock for the run
   auto mpiBlock = fileManager->getMPIBlock();
   fileManager->ensureDirectoryExists("."); // "." is relative to FileManager's baseDir.

   // Create correct header for comparison with file contents after writing
   BufferUtils::ActivityHeader correctHeader =
         createInitialHeader(layerLoc, mpiBlock, dataExtendedFlag, fileExtendedFlag);

   // Generate layer data
   double timestamp = 5.0;
   std::vector<Buffer<float>> layerData = generateLayerDataFrame1(layerLoc, dataExtendedFlag);

   // Write layer data using LayerFile.
   InfoLog() << "Creating test LayerFile...\n";
   status = writeFrameToLayerFile(
       layerLoc, filename, fileManager, timestamp, layerData, dataExtendedFlag, fileExtendedFlag);
   if (status != PV_SUCCESS) { return status; }

   // Read back the file outside of the LayerFile framework, to verify its contents
   InfoLog() << "Verifying header contents...\n";
   correctHeader.nBands = layerLoc.nbatch * mpiBlock->getBatchDimension();
   status = checkHeader(layerLoc, filename, fileManager, correctHeader);
   if (status != PV_SUCCESS) { return status; }

   InfoLog() << "Verifying LayerFile write...\n";
   std::vector<Buffer<float>> correctData =
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
   layerData = generateLayerDataFrame2(layerLoc, dataExtendedFlag);

   // Write data outside of the LayerFile framework, to verify reading from a file works.
   InfoLog() << "Writing layer data (2)...\n";
   status = writeFrameToFileStream(
       layerLoc, filename, fileManager, timestamp, layerData, dataExtendedFlag, fileExtendedFlag);
   if (status != PV_SUCCESS) { return status; }

   // Read data back using LayerFile.
   InfoLog() << "Verifying LayerFile read...\n";
   std::vector<Buffer<float>> frame2 = readFrame2(
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
      std::vector<Buffer<float>> const &correctData,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   int status = PV_SUCCESS;
   status = checkHeader(layerLoc, path, fileManager, correctHeader);
   if (status != PV_SUCCESS) { return status; }

   auto mpiBlock = fileManager->getMPIBlock();
   bool isRoot    = mpiBlock->getRank() == 0;
   auto layerFile = fileManager->open(path, std::ios_base::in);
   FatalIf(
         isRoot and !layerFile,
         "FileManager) failed to open \"%s\" on global rank %d: %s\n",
         path.c_str(),
         mpiBlock->getGlobalRank(),
         strerror(errno));
   FatalIf(
         !isRoot and layerFile,
         "FileManager) opened file \"%s\" on global rank %d, which is not a root process.\n",
         path.c_str(),
         mpiBlock->getGlobalRank());
   if (!layerFile) { return status; }

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
   long numValues = static_cast<long>(expectedNx * expectedNy * expectedNf);

   long headerSize = static_cast<long>(sizeof(correctHeader));
   layerFile->setInPos(headerSize, std::ios_base::beg);

   FatalIf(
         correctData.size() != static_cast<std::size_t>(expectedBatchWidth),
         "checkFrame1Contents(): correctData argument has length %zu, "
         "but layerLoc argument indicates length %d\n",
         correctData.size(), expectedBatchWidth);

   for (int b = 0; b < expectedBatchWidth; ++b) {
      double fileTimestamp;
      layerFile->read(&fileTimestamp, sizeof(fileTimestamp));
      if (fileTimestamp != correctTimestamp) {
         ErrorLog().printf(
               "Global rank %d, file \"%s\", batch element %d: "
               "timestamp is %f instead of the expected %f\n",
               mpiBlock->getGlobalRank(), path.c_str(), b,
               fileTimestamp, correctTimestamp);
         status = PV_FAILURE;
      }
      FatalIf(
            numValues != static_cast<long>(correctData[b].getTotalElements()),
            "checkFrame1Contents(): correctData batch element %d has length %zu, "
            "but layerLoc argument indicates length %ld\n",
            b, correctData[b].getTotalElements(), numValues);
      std::vector<float> fileData(numValues);
      long frameSizeBytes = numValues * static_cast<long>(sizeof(float));
      layerFile->read(fileData.data(), frameSizeBytes);
      for (long k = 0; k < numValues; ++k) {
         if (fileData.at(k) != correctData[b].at(k)) {
            ErrorLog().printf(
                  "Global rank %d, file \"%s\", batch element %d: Value %ld of %ld "
                  "is %f instead of the expected %f\n",
                  mpiBlock->getGlobalRank(), path.c_str(), 
                  k / numValues /*integer division*/, k % numValues, numValues,
                   (double)fileData.at(k), (double)correctData[b].at(k));
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
   int recordSize = nxBlock * nyBlock * layerLoc.nf;
   int dataSize = static_cast<int>(sizeof(float));
   int dataType = BufferUtils::returnDataType<float>();

   BufferUtils::ActivityHeader headerData;
   headerData.headerSize  = 80;
   headerData.numParams   = 20;
   headerData.fileType    = PVP_NONSPIKING_ACT_FILE_TYPE;
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

int deleteOldOutputDirectory(std::string const &outputDir) {
   std::string systemCommand("rm -rf \"");
   systemCommand.append(outputDir).append("\"");
   int status = std::system(systemCommand.c_str());
   if (status) {
      ErrorLog() << "system command rm -fr \"" << outputDir << "\" returned " << status << "\n";
   }
   return status ? PV_FAILURE : PV_SUCCESS;
}

std::vector<Buffer<float>> generateCorrectFileDataFrame1(
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      std::shared_ptr<MPIBlock const> &mpiBlock) {
   if (mpiBlock->getRank() != 0) { return std::vector<Buffer<float>>(); }
   int nx = layerLoc.nx * mpiBlock->getNumColumns();
   int ny = layerLoc.ny * mpiBlock->getNumRows();
   int nxGlobal = layerLoc.nxGlobal;
   int nyGlobal = layerLoc.nyGlobal;
   if (dataExtendedFlag) {
      nx += layerLoc.halo.lt + layerLoc.halo.rt;
      ny += layerLoc.halo.dn + layerLoc.halo.up;
      nxGlobal += layerLoc.halo.lt + layerLoc.halo.rt;
      nyGlobal += layerLoc.halo.dn + layerLoc.halo.up;
   }
   int nf = layerLoc.nf;

   int blockBatchDimension = layerLoc.nbatch * mpiBlock->getBatchDimension();
   std::vector<Buffer<float>> correctData(blockBatchDimension);

   // Create the LayerLoc that would apply if all MPI processes in a block were on a single
   // MPI process, and
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

   int blockNxExtended = blockLayerLoc.nx + blockLayerLoc.halo.lt + blockLayerLoc.halo.rt;
   int blockNyExtended = blockLayerLoc.ny + blockLayerLoc.halo.dn + blockLayerLoc.halo.up;
   int numNeuronsBlock = blockNxExtended * blockNyExtended * blockLayerLoc.nf;

   int globalNxExtended = blockLayerLoc.nxGlobal + blockLayerLoc.halo.lt + blockLayerLoc.halo.rt;
   int globalNyExtended = blockLayerLoc.nyGlobal + blockLayerLoc.halo.dn + blockLayerLoc.halo.up;
   int numNeuronsGlobal = globalNxExtended * globalNyExtended * blockLayerLoc.nf;
   for (int b = 0; b < blockBatchDimension; ++b) {
      correctData[b].resize(blockNxExtended, blockNyExtended, blockLayerLoc.nf);
      for (int k = 0; k < numNeuronsBlock; ++k) {
         int kf = k % nf;
         int kx = (k / nf) % blockNxExtended + blockLayerLoc.kx0;
         int ky = (k / (nf * blockNxExtended)) % blockNyExtended + blockLayerLoc.ky0;
         int const kGlobal =
               kIndex(kx, ky, kf, globalNxExtended, globalNyExtended, blockLayerLoc.nf);
         int const batchIndexGlobal = blockLayerLoc.kb0 + b;
         int const kGlobalAcrossBatch = kGlobal + batchIndexGlobal * numNeuronsGlobal;
         correctData[b].set(k, static_cast<float>(kGlobalAcrossBatch));
      }
      if (!fileExtendedFlag) {
         correctData[b].crop(blockLayerLoc.nx, blockLayerLoc.ny, Buffer<float>::CENTER);
      }
   }
   return correctData;
}

std::vector<Buffer<float>> generateLayerDataFrame1(
      PVLayerLoc const &layerLoc, bool dataExtendedFlag) {

   std::vector<Buffer<float>> layerData(layerLoc.nbatch);

   int width        = layerLoc.nx;
   int height       = layerLoc.ny;
   int globalWidth  = layerLoc.nxGlobal;
   int globalHeight = layerLoc.nyGlobal;
   if (dataExtendedFlag) {
      width += layerLoc.halo.lt + layerLoc.halo.rt;
      height += layerLoc.halo.dn + layerLoc.halo.up;
      globalWidth += layerLoc.halo.lt + layerLoc.halo.rt;
      globalHeight += layerLoc.halo.dn + layerLoc.halo.up;
   }
   int nf = layerLoc.nf;
   int numNeurons = width * height * nf;
   int numGlobalNeurons = globalWidth * globalHeight * nf;
   for (int b = 0; b < layerLoc.nbatch; ++b) {
      layerData[b].resize(width, height, nf);
      for (int k = 0; k < width * height * nf; ++k) {
         int kf = k % nf;
         int kx = (k / nf) % width + layerLoc.kx0;
         int ky = (k / (nf * width)) % height + layerLoc.ky0;
         int kGlobal = kIndex(kx, ky, kf, globalWidth, globalHeight, nf);
         int kGlobalBatchIndex = b + layerLoc.kb0;
         int kGlobalAcrossBatch = kGlobal + kGlobalBatchIndex * numGlobalNeurons;
         layerData[b].set(k, static_cast<float>(kGlobalAcrossBatch));
      }
   }
   return layerData;
}

std::vector<Buffer<float>> generateLayerDataFrame2(
      PVLayerLoc const &layerLoc, bool dataExtendedFlag) {
   
   std::vector<Buffer<float>> layerData(layerLoc.nbatch);

   int width        = layerLoc.nx;
   int height       = layerLoc.ny;
   int globalWidth  = layerLoc.nxGlobal;
   int globalHeight = layerLoc.nyGlobal;
   if (dataExtendedFlag) {
      width += layerLoc.halo.lt + layerLoc.halo.rt;
      height += layerLoc.halo.dn + layerLoc.halo.up;
      globalWidth += layerLoc.halo.lt + layerLoc.halo.rt;
      globalHeight += layerLoc.halo.dn + layerLoc.halo.up;
   }
   int nf = layerLoc.nf;
   int numNeurons = width * height * nf;
   int numGlobalNeurons = globalWidth * globalHeight * nf;
   int numGlobalAcrossBatch = numGlobalNeurons * layerLoc.nbatchGlobal;
   float numGlobalAcrossBatchF = static_cast<float>(numGlobalAcrossBatch);
   for (int b = 0; b < layerLoc.nbatch; ++b) {
      layerData[b].resize(width, height, nf);
      int numNeurons = width * height * nf;
      for (int k = 0; k < width * height * nf; ++k) {
         int kf = k % nf;
         int kx = (k / nf) % width + layerLoc.kx0;
         int ky = (k / (nf * width)) % height + layerLoc.ky0;
         int kGlobal = kIndex(kx, ky, kf, globalWidth, globalHeight, nf);
         int kGlobalBatchIndex = b + layerLoc.kb0;
         int kGlobalAcrossBatch = kGlobal + kGlobalBatchIndex * numGlobalNeurons;
         float kGlobalAcrossBatchF = static_cast<float>(kGlobalAcrossBatch);
         float val = 1.0f - kGlobalAcrossBatchF / numGlobalAcrossBatchF;
         layerData[b].set(k, val);
      }
   }
   return layerData;
}

std::vector<Buffer <float>> readFrame2(
      PVLayerLoc const &layerLoc,
      std::string const &filename,
      std::shared_ptr<FileManager const> fileManager,
      double &timestamp,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   LayerFile layerFile(
         fileManager,
         filename,
         layerLoc,
         dataExtendedFlag,
         fileExtendedFlag,
         true /*readOnlyFlag*/,
         false /*verifyWrites*/);
   int positionIndex = 1; // Position index is zero-based, so this is the second frame
   layerFile.setIndex(positionIndex);

   std::vector<Buffer <float>> frame2Contents(layerLoc.nbatch);
   int width = layerLoc.nx;
   int height = layerLoc.ny;
   if (dataExtendedFlag) {
      width += layerLoc.halo.lt + layerLoc.halo.rt;
      height += layerLoc.halo.dn + layerLoc.halo.up;
   }

   for (int b = 0; b < layerLoc.nbatch; ++b) {
      frame2Contents[b].resize(width, height, layerLoc.nf);
      layerFile.setDataLocation(frame2Contents[b].asVector().data(), b);
   }
   layerFile.read(timestamp);

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
      std::vector<Buffer<float>> const &dataFromLayer,
      std::vector<Buffer<float>> const &dataFromFile,
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

   int nxRestricted = layerLoc.nx;
   int nxExtended   = layerLoc.nx + layerLoc.halo.lt + layerLoc.halo.rt;
   int nyRestricted = layerLoc.ny;
   int nyExtended   = layerLoc.ny + layerLoc.halo.dn + layerLoc.halo.up;

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
      auto layerBuffer = dataFromLayer[b];
      auto fileBuffer = dataFromFile[b];
      if (dataExtendedFlag and !fileExtendedFlag) {
         layerBuffer.crop(layerLoc.nx, layerLoc.ny, Buffer<float>::CENTER);
         fileBuffer.crop(layerLoc.nx, layerLoc.ny, Buffer<float>::CENTER);
      }
      if (layerBuffer.asVector() != fileBuffer.asVector()) {
         pvAssert(layerBuffer.getTotalElements() == fileBuffer.getTotalElements());
         ErrorLog().printf("verifyRead(): discrepancy in batch element %d:\n", b);
         int numElements = static_cast<int>(layerBuffer.getTotalElements());
         for (int k = 0; k < numElements; ++k) {
            if (layerBuffer.at(k) != fileBuffer.at(k)) {
               ErrorLog().printf(
                     "    neuron %d of dataFromLayer is %f; of dataFromFile is %f\n",
                     k, (double)layerBuffer.at(k), (double)fileBuffer.at(k));
            }
         }
         status = PV_FAILURE;
      }
   }

   return status;
}

int writeFrameToFileStream(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<Buffer<float>> writeData,
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
   long const numValues = static_cast<long>(dataNx * dataNy * dataNf);
   long const numValuesAcrossBatch = numValues * static_cast<long>(layerLoc.nbatch);
   FatalIf(
         static_cast<int>(writeData.size()) != layerLoc.nbatch,
         "Global rank %d, file \"%s\", "
         "writeFrameToFileStream() writeData has %d batch elements instead of the expected %zu.\n",
         mpiBlock->getGlobalRank(), path.c_str(), numValuesAcrossBatch, writeData.size());
   for (int b = 0; b < layerLoc.nbatch; ++b) {
      if (static_cast<long>(writeData[b].getTotalElements()) != numValues) {
         ErrorLog().printf(
               "Global rank %d, file \"%s\", writeFrameToFileStream() writeData "
               "has %d neurons instead of the expected %ld\n",
               mpiBlock->getGlobalRank(), path.c_str(), writeData[b].getTotalElements(), numValues);
         status = PV_FAILURE;
      }
   }
   if (status != PV_SUCCESS) { return status; }

   int rootProcess = 0;
   if (mpiBlock->getRank() == rootProcess) {
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
             auto gatheredData = BufferUtils::gather<float>(
                   mpiBlock, writeData[b], layerLoc.nx, layerLoc.ny, m, rootProcess);
             if (dataExtendedFlag and !fileExtendedFlag) {
                gatheredData.crop(fileNx, fileNx, Buffer<float>::CENTER);
             }
             fileStream->write(&timestamp, 8L);

             pvAssert(gatheredData.getWidth() == fileNx);
             pvAssert(gatheredData.getHeight() == fileNy);
             pvAssert(gatheredData.getFeatures() == fileNf);
             long gatheredBufferSize = static_cast<long>(fileNx * fileNy * fileNf);
             long dataSize           = static_cast<long>(sizeof(float));
             fileStream->write(gatheredData.asVector().data(), gatheredBufferSize * dataSize);
         }
      }

      // Update NBands
      fileStream->setInPos(68L, std::ios_base::beg);
      int nBands;
      fileStream->read(&nBands, 4L);
      nBands += layerLoc.nbatch * mpiBlock->getBatchDimension();
      fileStream->setOutPos(68L, std::ios_base::beg);
      fileStream->write(&nBands, 4L);
   }
   else {
      for (int b = 0; b < layerLoc.nbatch; ++b) {
         int blockBatchIndex = mpiBlock->getBatchIndex();
         BufferUtils::gather<float>(
               mpiBlock, writeData[b], layerLoc.nx, layerLoc.ny, blockBatchIndex, rootProcess);
      }
   }

   return status;
}

int writeFrameToLayerFile(
      PVLayerLoc const &layerLoc,
      std::string const &path,
      std::shared_ptr<FileManager const> fileManager,
      double timestamp,
      std::vector<Buffer<float>> writeData,
      bool dataExtendedFlag,
      bool fileExtendedFlag) {
   // Create the pointer for the LayerFile object.
   std::unique_ptr<LayerFile> layerFile;

   // Create the file in write mode.
   layerFile = std::unique_ptr<LayerFile>(new LayerFile(
        fileManager,
        path,
        layerLoc,
        dataExtendedFlag,
        fileExtendedFlag,
        false /* readOnlyFlag */,
        false /* verifyWrites */));

   // Write layer data
   for (int b = 0; b < layerLoc.nbatch; ++b) {
      layerFile->setDataLocation(writeData[b].asVector().data(), b);
   }
   layerFile->write(timestamp);

   return PV_SUCCESS;
}
