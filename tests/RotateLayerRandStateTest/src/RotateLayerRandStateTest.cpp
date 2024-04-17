/*
 * RotateLayerRandTest.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include <structures/MPIBlock.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace PV;

int compare(PV_Init &pv_init_obj);
int deleteOldOutputDirectory(PV_Init &pv_init_obj);
int deletePath(std::shared_ptr<MPIBlock const> &ioMPIBlock, std::string const &path);
std::vector<uint8_t> loadFile(std::shared_ptr<FileStream> fileStream);
std::shared_ptr<FileStream> findFile(
      std::shared_ptr<MPIBlock const> ioMPIBlock,
      std::string const &dir,
      std::string const &filename);
int runReadTestStage(PV_Init &pv_init_obj);
int runWriteStage(PV_Init &pv_init_obj);

int main(int argc, char *argv[]) {
   PV_Init pv_init_obj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = PV_SUCCESS;

   // Delete any previously existing output directory
   if (status == PV_SUCCESS) {
      status = deleteOldOutputDirectory(pv_init_obj);
   }

   if (status == PV_SUCCESS) {
      status = runWriteStage(pv_init_obj);
   }

   if (status == PV_SUCCESS) {
      status = runReadTestStage(pv_init_obj);
   }

   if (status == PV_SUCCESS) {
      status = compare(pv_init_obj);
   }
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

std::shared_ptr<FileStream> findFile(
      std::shared_ptr<MPIBlock const> ioMPIBlock,
      std::string const &dir,
      std::string const &filename) {
   auto fileManager = std::make_shared<FileManager>(ioMPIBlock, dir);
   auto fileStream = FileStreamBuilder(
         fileManager,
         filename, 
         false /*isTextFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWritesFlag*/).get();
   return fileStream;
}

int compare(PV_Init &pv_init_obj) {
   auto *communicator = pv_init_obj.getCommunicator();
   auto ioMPIBlock    = communicator->getIOMPIBlock();
   auto fileStream1 = findFile(
         ioMPIBlock,
         "output/Checkpoints1/Checkpoint10",
         "Rotate_rand_state.bin");
   auto fileStream2 = findFile(
         ioMPIBlock,
         "output/Checkpoints2/Checkpoint10",
         "Rotate_rand_state.bin");
   pvAssert( (fileStream1 == nullptr) == (fileStream2 == nullptr) );
   if (ioMPIBlock->getRank() != 0) { return PV_SUCCESS; }
   pvAssert(fileStream1 != nullptr);
   pvAssert(fileStream2 != nullptr);
   pvAssert(fileStream1 != fileStream2); 

   // Read the contents and see whether they're identical.
   auto contents1 = loadFile(fileStream1);
   auto contents2 = loadFile(fileStream2);
   FatalIf(
         contents1.size() != contents2.size(), 
         "Files \"%s\" and \"%s\" have different sizes (%zu versus %zu)\n",
         fileStream1->getFileName().c_str(),
         fileStream2->getFileName().c_str(),
         contents1.size(),
         contents2.size());
   std::size_t N = contents1.size();
   int status = PV_SUCCESS;
   for (std::size_t n = 0UL; n < N; ++n) {
      if (contents1[n] != contents2[n]) {
         ErrorLog().printf(
               "Byte %zu (zero indexed) of %zu mismatch: file 1 has %u but file 2 has %u\n",
               n,
               N,
               static_cast<unsigned int>(contents1[n]),
               static_cast<unsigned int>(contents2[n]));
         status = PV_FAILURE;
      }
   }
   FatalIf(
         status != PV_SUCCESS, 
         "Files \"%s\" and \"%s\" do not agree.\n",
         fileStream1->getFileName().c_str(),
         fileStream2->getFileName().c_str());

   return PV_SUCCESS;
}

int deleteOldOutputDirectory(PV_Init &pv_init_obj) {
   auto *communicator = pv_init_obj.getCommunicator();
   auto ioMPIBlock    = communicator->getIOMPIBlock();
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) {
      status = deletePath(ioMPIBlock, "output");
   }
   char const *fmtstring1 = "output/Checkpoints1/Checkpoint%02d";
   char const *fmtstring2 = "output/Checkpoints2/Checkpoint%02d";
   std::string path("output/Checkpoints#/Checkpoint##");
   auto pathsize = path.size();
   ++pathsize;
   for (int k=0; k <= 10; ++k) {
      if (status == PV_SUCCESS) {
         std::snprintf(&path.at(0), pathsize, fmtstring1, k);
         status = deletePath(ioMPIBlock, path);
      }
      if (status == PV_SUCCESS) {
         std::snprintf(&path.at(0), pathsize, fmtstring2, k);
         status = deletePath(ioMPIBlock, path);
      }
   }
   return PV_SUCCESS;
}

int deletePath(std::shared_ptr<MPIBlock const> &ioMPIBlock, std::string const &path) {
   FileManager fileManager(ioMPIBlock, path);
   if (!fileManager.isRoot()) { return PV_SUCCESS; }
   std::string blockFilename   = fileManager.makeBlockFilename(".");
   char *resolvedBlockFilename = ::realpath(blockFilename.c_str(), nullptr);
   if (!resolvedBlockFilename) {
      if (errno == ENOENT) {
         return PV_SUCCESS;
      }
      else {
         ErrorLog().printf(
               "Unable to resolve path \"%s\" in IO MPIBlock: %s\n",
               blockFilename.c_str(),
               strerror(errno));
      }
      return PV_FAILURE;
   }
   std::string systemCommand("rm -fr \"");
   systemCommand.append(resolvedBlockFilename).append("\"");
   InfoLog().printf("std::system(%s)\n", systemCommand.c_str());
   int status = std::system(systemCommand.c_str());
   if (status) {
      ErrorLog() << "system command '" << systemCommand << "' returned " << status << "\n";
      return PV_FAILURE;
   }
   return PV_SUCCESS;
}

std::vector<uint8_t> loadFile(std::shared_ptr<FileStream> fileStream) {
   fileStream->open();
   fileStream->setInPos(0L, std::ios_base::end);
   long int filesize = fileStream->getInPos();
   fileStream->setInPos(0L, std::ios_base::beg);
   std::vector<uint8_t> contents(filesize);
   fileStream->read(&contents.at(0), filesize);
   return contents;
}

int runReadTestStage(PV_Init &pv_init_obj) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) {
      status = pv_init_obj.setParams("input/RotateLayerRandStateTest_CheckpointReadTest.params");
   }
   if (status == PV_SUCCESS) {
      bool succeeded = pv_init_obj.setStringArgument(
            "CheckpointReadDirectory", "output/Checkpoints1/Checkpoint05");
      status = succeeded ? PV_SUCCESS : PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      status = buildandrun(&pv_init_obj);
   }
   return status;
}

int runWriteStage(PV_Init &pv_init_obj) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) {
      status = pv_init_obj.setParams("input/RotateLayerRandStateTest_CheckpointWrite.params");
   }
   if (status == PV_SUCCESS) {
      status = buildandrun(&pv_init_obj);
   }
   return status;
}

