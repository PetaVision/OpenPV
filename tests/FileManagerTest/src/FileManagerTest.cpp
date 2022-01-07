/*
 * FileManagerTest.cpp
 *
 */

#include "include/pv_common.h"
#include "io/FileManager.hpp"
#include "io/io.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/WaitForReturn.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "sys/stat.h"

#define GLOBAL_MPI_SIZE 32

using namespace PV;

int deleteOldOutputDirectory();
int run();
int runconfig(
      int globalNumRows,
      int globalNumColumns,
      int globalBatchDimension,
      int blockNumRows,
      int blockNumColumns,
      int blockBatchDimension);
void applyLogFileOption(int argc, char *argv[]);
void checkFileContents(
      std::string const &contents, FileStream *stream, std::string const &message);
std::string createBlockDirectoryPath(std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDir);
std::string createConfigID(FileManager &fileManager);
std::string createConfigID(std::shared_ptr<MPIBlock const> mpiBlock);
std::string createFullPath(
      std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDir, std::string const &filePath);
void testDeleteDirectory(
      FileManager &fileManager,
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::string const &baseDir,
      std::string const &localDirPath);
void testDeleteFile(
      FileManager &fileManager,
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::string const &baseDir,
      std::string const &filePath);
void testRead(
      FileManager &fileManager, std::string const &filePath, std::string const &fileContents);
void testWrite(
      FileManager &fileManager, std::string const &filePath, std::string const &fileContents);
void verifyIndependently(
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::string const &baseDir,
      std::string const &filePath,
      std::string const &fileContents);

int main(int argc, char *argv[]) {
   MPI_Init(&argc, &argv);

   int status = EXIT_SUCCESS;
   int globalSize;
   MPI_Comm_size(MPI_COMM_WORLD, &globalSize);
   if ( globalSize != GLOBAL_MPI_SIZE) {
      ErrorLog() << argv[0] << " must be run with exactly " << GLOBAL_MPI_SIZE <<
                   " MPI processes. There are " << globalSize << ".\n";
      status = EXIT_FAILURE;
   }

   applyLogFileOption(argc, argv);

   bool requireReturn = (pv_getopt(argc, argv, "--require-return", nullptr /*paramusage*/) == 0);
   if (requireReturn) { WaitForReturn(MPI_COMM_WORLD); }

   if (status == EXIT_SUCCESS) {
      status = deleteOldOutputDirectory();
   }

   if (status == EXIT_SUCCESS) {
      status = run() ? EXIT_FAILURE : EXIT_SUCCESS;
   }
   MPI_Finalize();

   return status;
}

int deleteOldOutputDirectory() {
   int status = std::system("rm -rf output/");
   if (status) {
      ErrorLog() << "system command \"rm -fr output/\" returned " << status << "\n";
   }
   return status ? PV_FAILURE : PV_SUCCESS;
}

int run() {
   int status = PV_SUCCESS;

   // Run several configurations. In each configuration, the
   // product of the global parameters must equal GLOBAL_MPI_SIZE,
   // Each block parameter must divide the corresponding global
   // parameter evenly
   status = runconfig(4,4,2,4,4,2) ? PV_FAILURE : status;
   status = runconfig(4,4,2,2,4,2) ? PV_FAILURE : status;
   status = runconfig(4,4,2,4,2,2) ? PV_FAILURE : status;
   status = runconfig(4,4,2,4,4,1) ? PV_FAILURE : status;

   return status;
}

int runconfig(
      int globalNumRows,
      int globalNumColumns,
      int globalBatchDimension,
      int blockNumRows,
      int blockNumColumns,
      int blockBatchDimension) {
   int globalSize;
   MPI_Comm_size(MPI_COMM_WORLD, &globalSize);
   pvAssert(globalNumRows * globalNumColumns * globalBatchDimension == globalSize);
   pvAssert(globalNumRows % blockNumRows == 0);
   pvAssert(globalNumColumns % blockNumColumns == 0);
   pvAssert(globalBatchDimension % blockBatchDimension == 0);

   std::string testDirectoryName("test");
   testDirectoryName.append(std::to_string(globalNumRows));
   testDirectoryName.append(std::to_string(globalNumColumns));
   testDirectoryName.append(std::to_string(globalBatchDimension));
   testDirectoryName.append(std::to_string(blockNumRows));
   testDirectoryName.append(std::to_string(blockNumColumns));
   testDirectoryName.append(std::to_string(blockBatchDimension));

   std::stringstream configStream;
   configStream << "NumRows:" << globalNumRows << "\n";
   configStream << "NumColumns:" << globalNumColumns << "\n";
   configStream << "BatchWidth:" << globalBatchDimension << "\n";
   configStream << "CheckpointCellNumRows:" << blockNumRows << "\n";
   configStream << "CheckpointCellNumColumns:" << blockNumColumns << "\n";
   configStream << "CheckpointCellBatchDimension:" << blockBatchDimension << "\n";
   Arguments arguments(configStream, false /*do not allow unrecognized arguments*/);

   auto mpiBlock = std::make_shared<MPIBlock>(
         MPI_COMM_WORLD,
         globalNumRows,
         globalNumColumns,
         globalBatchDimension,
         blockNumRows,
         blockNumColumns,
         blockBatchDimension);

   std::string baseDir("./output");
   baseDir.append("/").append(testDirectoryName);
   FileManager fileManager(mpiBlock, baseDir);

   fileManager.ensureDirectoryExists(std::string("dir"));
   int const rank = mpiBlock->getRank();

   int const globalRank = mpiBlock->getGlobalRank();
   std::string testFileContents("Global rank ");
   testFileContents.append(std::to_string(globalRank)).append("\n");

   std::string filePath("dir/file.txt");

   // Create a file using FileManager and write to it
   testWrite(fileManager, filePath, testFileContents);

   // Read back contents using FileManager
   testRead(fileManager, filePath, testFileContents);
      
   // Read back without using FileManager
   if (rank == fileManager.getRootProcessRank()) {
      verifyIndependently(mpiBlock, baseDir, filePath, testFileContents);
   }

   // Delete a file using FileManager, and verify it's gone without using FileManager
   testDeleteFile(fileManager, mpiBlock, baseDir, filePath);

   // Delete a directory using FileManager, and verify it's gone without using FileManager
   testDeleteDirectory(fileManager, mpiBlock, baseDir, std::string("dir"));

   InfoLog().printf(
         "Test passed on global rank %d with "
         "global MPI configuration %d rows, %d columns, %d batchwidth and "
         "block MPI configuration %d rows, %d columns, %d batchwidth.\n",
         globalRank,
         globalNumRows, globalNumColumns, globalBatchDimension,
         blockNumRows, blockNumColumns, blockBatchDimension);
   return PV_SUCCESS;
}

void applyLogFileOption(int argc, char *argv[]) {
   char *logfile = nullptr;
   pv_getopt_str(argc, argv, "-l", &logfile, nullptr /*paramusage*/);

   if (logfile) {
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
      free(logfile);
   }
}

void checkFileContents(
      std::string const &contents, FileStream *stream, std::string const &message) {
   FatalIf(contents.empty(), "%s: checkFileContents() called with empty string\n", message.c_str());
   auto contentsSize = contents.size();
   std::string fileContents(contentsSize, '.');
   stream->read(&fileContents.front(), static_cast<long>(contentsSize));
   FatalIf(fileContents != contents, "%s: reading back test file failed.\n", message.c_str());
}

std::string createBlockDirectoryPath(std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDir) {
   std::string blockDirectoryName(baseDir);
   if ( mpiBlock->getGlobalNumRows() != mpiBlock->getNumRows() or
        mpiBlock->getGlobalNumColumns() != mpiBlock->getNumColumns() or
        mpiBlock->getGlobalBatchDimension() != mpiBlock->getBatchDimension()) {
      int blockColumnIndex = mpiBlock->getStartColumn() / mpiBlock->getNumColumns();
      int blockRowIndex    = mpiBlock->getStartRow() / mpiBlock->getNumRows();
      int blockElemIndex   = mpiBlock->getStartBatch() / mpiBlock->getBatchDimension();
      blockDirectoryName.append("/block_");
      blockDirectoryName.append("col").append(std::to_string(blockColumnIndex));
      blockDirectoryName.append("row").append(std::to_string(blockRowIndex));
      blockDirectoryName.append("elem").append(std::to_string(blockElemIndex));
      blockDirectoryName.append("/");
   }
   return blockDirectoryName;
}

std::string createConfigID(FileManager &fileManager) {
   return createConfigID(fileManager.getMPIBlock());
}

std::string createConfigID(std::shared_ptr<MPIBlock const> mpiBlock) {
   std::string idMessage("Global rank ");
   idMessage.append(std::to_string(mpiBlock->getGlobalRank()));
   idMessage.append(", block rank ").append(std::to_string(mpiBlock->getRank()));
   return idMessage;
}

std::string createFullPath(
      std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDir, std::string const &filePath) {
   std::string fullPath = createBlockDirectoryPath(mpiBlock, baseDir) + "/" + filePath;
   return fullPath;
}

void testDeleteDirectory(
      FileManager &fileManager,
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::string const &baseDir,
      std::string const &localDirPath) {
   // Delete the directory "dir" using FileManager
   fileManager.deleteDirectory(localDirPath);

   // Check that the directory is gone without using FileManager
   if (mpiBlock->getRank() == fileManager.getRootProcessRank()) {
      std::string fullPath = createFullPath(mpiBlock, baseDir, localDirPath);
      struct stat statbuffer;
      int statstatus = ::stat(fullPath.c_str(), &statbuffer);
      FatalIf(
            !statstatus,
            "%s: directory \"%s\" under base directory %s should have been deleted, "
            "but it is still present.\n",
            createConfigID(fileManager).c_str(),
            localDirPath.c_str(),
            baseDir.c_str());
      FatalIf(
            errno != ENOENT,
            "%s: error getting status of directory \"%s\" under base directory %s, "
            "which should be deleted: %s\n",
            createConfigID(fileManager).c_str(),
            localDirPath.c_str(),
            baseDir.c_str(),
            strerror(errno));
   }
}

void testDeleteFile(
      FileManager &fileManager,
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::string const &baseDir,
      std::string const &filePath) {
   fileManager.deleteFile(filePath);

   // Check that the file is gone without using FileManager
   std::string fullPath = createFullPath(mpiBlock, baseDir, filePath);
   if (mpiBlock->getRank() == fileManager.getRootProcessRank()) {
      struct stat statbuffer;
      int statstatus = ::stat(fullPath.c_str(), &statbuffer);
      FatalIf(
            !statstatus,
            "%s: %s under base directory %s should have been deleted, but it is still present.\n",
            createConfigID(fileManager).c_str(),
            filePath.c_str(),
            baseDir.c_str());
      FatalIf(
            errno != ENOENT,
            "%s: error getting status of %s under base directory %s, which should be deleted: %s\n",
            createConfigID(fileManager).c_str(),
            filePath.c_str(),
            baseDir.c_str(),
            strerror(errno));
   }
}

void testRead(
      FileManager &fileManager, std::string const &filePath, std::string const &fileContents) {
   auto readbackStream = fileManager.open(filePath, std::ios_base::in);
   std::string idMessage = createConfigID(fileManager);
   if (fileManager.isRoot()) {
      FatalIf(
            !readbackStream,
            "%s: filestream should be non-null for root processes, but is null.\n",
            idMessage.c_str());
      checkFileContents(fileContents, readbackStream.get(), idMessage);
   }
   else {
      FatalIf(
            readbackStream,
            "%s: filestream should be null for non-root processes, but is not.\n",
            idMessage.c_str());
   }
}

void testWrite(
      FileManager &fileManager, std::string const &filePath, std::string const &fileContents) {
   auto filestream = fileManager.open(filePath, std::ios_base::out);
   if (fileManager.isRoot()) {
      FatalIf(
            !filestream,
            "idMessage: filestream should be non-null for root processes, but is null.\n",
            createConfigID(fileManager).c_str());

      pvAssert(!fileContents.empty());
      long contentsSize = static_cast<long>(fileContents.size());
      filestream->write(&fileContents.front(), contentsSize);
   }
   else {
      FatalIf(
            filestream,
            "%s: filestream should be null for non-root processes, but is not.\n",
            createConfigID(fileManager).c_str());
   }
}

void verifyIndependently(
      std::shared_ptr<MPIBlock const> mpiBlock,
      std::string const &baseDir,
      std::string const &filePath,
      std::string const &fileContents) {
   std::string fullPath = createFullPath(mpiBlock, baseDir, filePath);
   FileStream *fullPathStream = new FileStream(fullPath.c_str(), std::ios_base::in);
   checkFileContents(fileContents, fullPathStream, createConfigID(mpiBlock));
   delete fullPathStream;
}
