// Code for the ProbeOutputTest system test. It does a PetaVision run with probe output, and then
// reads the output back to check for correctness.
//
// It repeats the process for initializing from a checkpoint directory, for restarting from a
// checkpoint, and restarting from the end with a larger stopTime.
//
// The program expects to receive the PetaVision configuration through a configuration file,
// as opposed to the command line. It looks for two extra configuration parameters:
// RunName, a string of filename-friendly characters, used to separate output from different runs.
// RunDescription, a string used to describe the specific run in diagnostic messages,
// OutputTruncation, a positive integer used to truncate probe output files before restarting
//     from checkpoint

#include "manageFiles.hpp"

#include <arch/mpi/mpi.h>
#include <columns/buildandrun.hpp>
#include <columns/Communicator.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include "utils/PathComponents.hpp"
#include <utils/PVLog.hpp>

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>

using namespace PV;

void archiveLogFile(std::shared_ptr<PV_Init> pv_init_obj, std::string const &destDir);
int compareFileContents(std::shared_ptr<FileStream> observed, std::shared_ptr<FileStream> correct);
int compareProbeOutput(
      std::string const &probeName,
      std::string const &correctDir,
      std::shared_ptr<FileManager> fileManager);
int compareWords(std::string const &observedWord, std::string const &correctWord);
int copyCheckpointDir(Communicator *communicator);
std::shared_ptr<PV_Init> createPV_Init(
      std::string const &programPath, std::string const &runName, std::string const &stageName);
std::string generateMPIConfigString(Communicator *communicator);
int getBatchWidth(std::shared_ptr<PV_Init> pv_init_obj);
std::string getWord(std::string const &str, std::string::size_type &pos);
std::string readFileContents(std::shared_ptr<FileStream> fileStream);
int run(int argc, char **argv);
int runBase(int argc, char **argv);
int runInitFromCheckpoint(int argc, char **argv);
int runRestartFromCheckpoint(int argc, char **argv);
int runRestartFromEnd(int argc, char **argv);

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);

   int result = run(argc, argv);

   MPI_Finalize();

   return result == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void archiveLogFile(std::shared_ptr<PV_Init> pv_init_obj, std::string const &destDir) {
   auto *communicator = pv_init_obj->getCommunicator();
   // Move log file into output directory
   int globalRank = communicator->globalCommRank();
   std::string const &logFile = pv_init_obj->getStringArgument("LogFile");
   std::string srcDirectory   = dirName(logFile);
   std::string logPath;
   if (!logFile.empty()) {
      std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);
      std::string destPath = destDir + '/';
      if (globalRank != 0) {
         std::string stripExt    = stripExtension(logFile);
         std::string fileExt     = extension(logFile);
         std::string logFileName = stripExt + '_' + std::to_string(globalRank) + fileExt;
         logPath                 = srcDirectory + '/' + logFileName;
         destPath += logFileName;
      }
      else {
         std::string fileName  = baseName(logFile);
         logPath               = srcDirectory + '/' + fileName;
         destPath += fileName;
      }
      std::filesystem::rename(logPath, destPath);
   }
}

int compareFileContents(std::shared_ptr<FileStream> observed, std::shared_ptr<FileStream> correct) {
   int status = PV_SUCCESS;
   std::string observedString = readFileContents(observed);
   std::string correctString = readFileContents(correct);

   auto observedSize = observedString.size();
   auto correctSize  = correctString.size();
   decltype(observedSize) observedPos = 0;
   decltype(correctSize) correctPos   = 0;
   int lineNumber = 1;
   int fieldNumber = 1;
   while (observedPos < observedSize and correctPos < correctSize) {
      std::string observedWord = getWord(observedString, observedPos);
      std::string correctWord  = getWord(correctString, correctPos);
      assert(observedWord.size() and correctWord.size());

      char observedStop = observedWord.back(); observedWord.pop_back();
      char correctStop  = correctWord.back();  correctWord.pop_back();

      if (compareWords(observedWord, correctWord) != PV_SUCCESS) {
         ErrorLog().printf(
               "Observed file does not match correct file at line %d, field %d\n",
               lineNumber, fieldNumber);
         status = PV_FAILURE;
         break;
      }

      FatalIf(
            observedPos > observedSize,
            "Position in observed values file (%lu) is past the file size (%lu).\n",
            static_cast<unsigned long int>(observedPos),
            static_cast<unsigned long int>(observedSize));
      FatalIf(
            correctPos > correctSize,
            "Position in correct values file (%lu) is past the file size (%lu).\n",
            static_cast<unsigned long int>(correctPos),
            static_cast<unsigned long int>(correctSize));
      if (observedPos == observedSize and correctPos < correctSize) {
         ErrorLog().printf(
               "Observed values file \"%s\" ended before correct values file \"%s\" "
               "at line %d, field %d\n",
               observed->getFileName().c_str(), lineNumber, fieldNumber);
         status = PV_FAILURE;
         break;
      }
      if (correctPos == correctSize and observedPos < observedSize) {
         ErrorLog().printf(
               "Observed values file \"%s\" continues beyond correct values file \"%s\" "
               "at line %d, field %d\n",
               observed->getFileName().c_str(), correct->getFileName().c_str(),
               lineNumber, fieldNumber);
         status = PV_FAILURE;
         break;
      }

      if (observedStop == correctStop) {
         if (observedStop == '\n') {
            ++lineNumber;
            fieldNumber = 1;
         }
         else {
            ++fieldNumber;
         }
      }
      else {
         ErrorLog().printf(
               "Observed field delimiter differs from correct field delimiter "
               "(%d versus %d), after line %d, field %d\n",
               static_cast<int>(observedStop), static_cast<int>(correctStop),
               lineNumber, fieldNumber);
         status = PV_FAILURE;
         break;
      }
   }
   if (status == PV_SUCCESS) {
      InfoLog().printf(
            "Observed values in \"%s\" agree with correct values in \"%s\"\n",
            observed->getFileName().c_str(), correct->getFileName().c_str());
   }
   else {
      ErrorLog().printf(
            "Comparison failed between observed values in \"%s\" and correct values in \"%s\"\n",
            observed->getFileName().c_str(), correct->getFileName().c_str());
   }
   return status;
}

int compareProbeOutput(
      std::string const &probeName,
      std::string const &correctDir,
      std::shared_ptr<FileManager> fileManager) {
   int batchWidth = 8;
   std::vector<int> statusByBatchElement(batchWidth, PV_SUCCESS);
   if (fileManager->isRoot()) {
      for (int b = 0; b < batchWidth; ++b) {
         std::string fileName = probeName + "_batchElement_#.txt";
         fileName.replace(fileName.find("#"), 1, std::to_string(b));
         struct stat statbuf;
         int statstatus = fileManager->stat(fileName, statbuf);
         if (statstatus != 0) {
            if (errno != ENOENT) {
               statusByBatchElement[b] = PV_FAILURE;
               ErrorLog().printf("stat(\"%s\") failed: %s\n", fileName.c_str(), strerror(errno));
            }
            continue;
         }
         std::shared_ptr<FileStream> outputL2Norm = FileStreamBuilder(
               fileManager,
               fileName,
               true /*isTestFlag*/,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWrites*/).get();
         std::string correctDataPath("input/correctProbeOutput/#1/correct_#2_#3.txt");
         correctDataPath.replace(correctDataPath.find("#1"), 2, correctDir);
         correctDataPath.replace(correctDataPath.find("#2"), 2, probeName);
         correctDataPath.replace(correctDataPath.find("#3"), 2, std::to_string(b));
         auto correctL2Norm = std::make_shared<FileStream>(correctDataPath.c_str(), std::ios_base::in);
         statusByBatchElement[b] = compareFileContents(outputL2Norm, correctL2Norm);
      }
   }
   int status = PV_SUCCESS;
   for (auto const &s : statusByBatchElement) {
      if (s != PV_SUCCESS) { status = PV_FAILURE; }
   }
   return status;
}

int compareWords(std::string const &observedWord, std::string const &correctWord) {
   float observedValue, correctValue;
   std::size_t observedPos, correctPos;
   try {
      observedValue = std::stof(observedWord, &observedPos);
   } catch (std::exception const &e) {
      observedPos = 0UL;
   }
   try {
      correctValue = std::stof(correctWord, &correctPos);
   } catch (std::exception const &e) {
      correctPos = 0UL;
   }
   if (observedPos > 0UL and correctPos > 0UL) {
      auto observedRemnant = observedWord.substr(observedPos);
      auto correctRemnant  = correctWord.substr(correctPos);
      if (observedRemnant != correctRemnant) {
         ErrorLog().printf(
               "Observed field \"%s\" does not match correct field \"%s\"\n",
               observedWord.c_str(), correctWord.c_str());
         return PV_FAILURE;
      }
      float discrepancy = std::fabs(observedValue - correctValue);
      if (discrepancy > 1.0e-6f * std::fabs(correctValue)) {
         ErrorLog().printf(
               "Observed value %f differs significantly from correct value %f (discrepancy %g)\n",
               static_cast<double>(observedValue),
               static_cast<double>(correctValue),
               static_cast<double>(discrepancy));
         return PV_FAILURE;
      }
      return PV_SUCCESS;
   }
   else if (observedWord != correctWord) {
      ErrorLog().printf(
            "Observed field \"%s\" does not match correct field \"%s\"\n",
            observedWord.c_str(), correctWord.c_str());
      return PV_FAILURE;
   }
   else {
      return PV_SUCCESS;
   }
}

int copyCheckpointDir(Communicator *communicator) {
   int status = PV_SUCCESS;
   if (status == PV_SUCCESS) {
      std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);
      std::string initChkptDir = mpiConfigString + "/checkpoints/Checkpoint040";
      status = recursiveCopy(initChkptDir, "initializing_checkpoint", communicator);
   }
   return status;
}

std::shared_ptr<PV_Init> createPV_Init(
      std::string const &programPath, std::string const &runName, std::string const &stageName) {
   int argc = 2;
   std::vector<std::string> convertedArgStrings(argc);
   std::vector<char*> convertedArgs(argc + 1);

   convertedArgStrings[0] = programPath;

   std::string configFile =
         dirName(programPath) + "/../input/config_" + runName + stageName + ".txt";
   auto configPath = std::filesystem::canonical(configFile);
   convertedArgStrings[1] = configPath.string();

   convertedArgs[0] = convertedArgStrings[0].data();
   convertedArgs[1] = convertedArgStrings[1].data();
   convertedArgs[argc] = nullptr;
   char **convertedArgV = convertedArgs.data();
   auto pv_init_obj =
         std::make_shared<PV_Init>(&argc, &convertedArgV, true /*allowUnrecognizedArgumentsFlag*/);
   return pv_init_obj;
}

std::string generateMPIConfigString(Communicator *communicator) {
   // Analyze MPI configuration to deduce a run name.
   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();

   int globalSize       = communicator->globalCommSize();
   int globalNumRows    = ioMPIBlock->getGlobalNumRows();
   int globalNumColumns = ioMPIBlock->getGlobalNumColumns();
   int globalBatchDim   = ioMPIBlock->getGlobalBatchDimension();
   int cellNumRows      = ioMPIBlock->getNumRows();
   int cellNumColumns   = ioMPIBlock->getNumColumns();
   int cellBatchDim     = ioMPIBlock->getBatchDimension();

   std::string mpiConfigString;
   mpiConfigString.append(std::to_string(globalSize)).append("procs_");
   mpiConfigString.append(std::to_string(globalNumRows)).append("r-by-");
   mpiConfigString.append(std::to_string(globalNumColumns)).append("c-by-");
   mpiConfigString.append(std::to_string(globalBatchDim)).append("b");
   mpiConfigString.append("_cellsize_");
   mpiConfigString.append(std::to_string(cellNumRows)).append("r-by-");
   mpiConfigString.append(std::to_string(cellNumColumns)).append("c-by-");
   mpiConfigString.append(std::to_string(cellBatchDim)).append("b");
   return mpiConfigString;
}

int getBatchWidth(std::shared_ptr<PV_Init> pv_init_obj) {
   auto *params = pv_init_obj->getParams();
   HyPerCol dummyHyPerCol(pv_init_obj.get());
   int batchWidth = dummyHyPerCol.getNBatchGlobal();
   return batchWidth;
}

std::string getWord(std::string const &str, std::string::size_type &pos) {
   auto newPos = str.find_first_of(",\n", pos);
   if (newPos == std::string::npos) {
      newPos = str.size();
   }
   else {
      ++newPos;
   }
   std::string result = str.substr(pos, newPos - pos);
   pos = newPos;
   return result;
}

std::string readFileContents(std::shared_ptr<FileStream> fileStream) {
   int status = PV_SUCCESS;
   fileStream->setInPos(0L, std::ios_base::end);
   long int streamSize = fileStream->getInPos();
   fileStream->setInPos(0L, std::ios_base::beg);
   std::string result(streamSize, '\0');
   fileStream->read(result.data(), streamSize);
   return result;
}

int run(int argc, char **argv) {
   int status = PV_SUCCESS;

   if (status == PV_SUCCESS) {
      status = runBase(argc, argv);
   }
   if (status == PV_SUCCESS) {
      if (runInitFromCheckpoint(argc, argv) != PV_SUCCESS) {
         status = PV_FAILURE;
      }
      if (runRestartFromCheckpoint(argc, argv) != PV_SUCCESS) {
         status = PV_FAILURE;
      }
      if (runRestartFromEnd(argc, argv) != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

int runBase(int argc, char **argv) {
   auto pv_init_obj = createPV_Init(argv[0], argv[1], "");
   auto *communicator = pv_init_obj->getCommunicator();

   // Delete any existing output directory, to be sure that any files
   // are the result of this run and not left over from previous runs
   int status = recursiveDelete("output", communicator, false /*warnIfAbsentFlag*/);
   if (status != PV_SUCCESS) {
      ErrorLog().printf("Unable to delete previously existing directory \"output\"\n");
      status = PV_FAILURE;
   }

   if (status == PV_SUCCESS) {
      InfoLog().printf("Executing base run\n");
      status = buildandrun(pv_init_obj.get(), nullptr, nullptr);
   }
   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   auto fileManager = std::make_shared<FileManager>(ioMPIBlock, "output");
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("OutputL2Norm", "base", fileManager);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("TotalEnergy", "base", fileManager);
   }
   std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);
   if (status == PV_SUCCESS) {
      int moveStatus = renamePath("output", mpiConfigString, communicator);
      if (moveStatus != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      status = copyCheckpointDir(communicator);
   }

   if (status == PV_SUCCESS) {
      InfoLog().printf("Base run passed.\n");
      // Archive log file(s) into output directory
      archiveLogFile(pv_init_obj, mpiConfigString);
   }
   return status;
}

int runInitFromCheckpoint(int argc, char **argv) {
   int status = PV_SUCCESS;

   auto pv_init_obj = createPV_Init(argv[0], argv[1], "-ifcp");
   InfoLog().printf("Initializing from checkpoint...\n");

   auto *communicator = pv_init_obj->getCommunicator();
   auto globalComm = communicator->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);

   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   auto archiveFileManager = std::make_shared<FileManager>(ioMPIBlock, mpiConfigString);
   auto outputFileManager = std::make_shared<FileManager>(ioMPIBlock, "output");
   auto initchkptFileManager = std::make_shared<FileManager>(ioMPIBlock, "initializing_checkpoint");
   outputFileManager->ensureDirectoryExists(std::string("."));

   int batchWidth = getBatchWidth(pv_init_obj);

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank and communicator->getIOMPIBlock()->getRank() == 0) {
         for (int b = 0; b < batchWidth; ++b) {
            std::string filename("TotalEnergy_batchElement_");
            filename.append(std::to_string(b)).append(".txt");
            std::string archivePath = archiveFileManager->makeBlockFilename(filename);
            std::string outputPath;
            auto archiveFileType = std::filesystem::symlink_status(archivePath).type();
            bool shouldCopy = (archiveFileType == std::filesystem::file_type::regular);
            if (shouldCopy) {
               outputPath = outputFileManager->makeBlockFilename(filename);
               auto outputFileType = std::filesystem::symlink_status(outputPath).type();
               shouldCopy = (outputFileType == std::filesystem::file_type::not_found);
            }
            if (shouldCopy) {
               std::filesystem::copy(archivePath, outputPath);
               std::string fileposFilename = filename + "_filepos_FileStreamWrite.bin";
               auto filepos = initchkptFileManager->open(fileposFilename, std::ios_base::in);
               filepos->setInPos(0L, std::ios_base::beg);
               long int checkpointedPosition;
               filepos->read(&checkpointedPosition, sizeof(checkpointedPosition));
               filepos = nullptr; // close the file
               long int newSize = checkpointedPosition + 100L;
               std::filesystem::resize_file(outputPath, newSize);
               FatalIf(
                     std::filesystem::file_size(outputPath) != newSize,
                     "resize_file() failed to resize \"%s\" to %ld characters.\n",
                     outputPath.c_str(), newSize);
               FileStream outputFile(outputPath.c_str(), std::ios_base::app);
               char const *extraData = "xxxxxxxx\n";
               outputFile.write(extraData, std::strlen(extraData));
            }
         }
      }
      // Prevent collisions if multiple root processes see the same filesystem
      MPI_Barrier(communicator->globalCommunicator());
   }

   if (status == PV_SUCCESS) {
      status = buildandrun(pv_init_obj.get(), nullptr, nullptr);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("OutputL2Norm", "initfromchkpt", outputFileManager);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("TotalEnergy", "initfromchkpt", outputFileManager);
   }
   std::string archiveString = mpiConfigString + "_initfromchkpt";
   if (status == PV_SUCCESS) {
      int moveStatus = renamePath("output", archiveString, communicator);
      if (moveStatus != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      InfoLog().printf("Initializing from checkpoint passed.\n");
      // Archive log file(s) into output directory
      archiveLogFile(pv_init_obj, archiveString);
   }
   return status;
}

int runRestartFromCheckpoint(int argc, char **argv) {
   int status = PV_SUCCESS;
   InfoLog().printf("Restarting from checkpoint...\n");
   auto pv_init_obj = createPV_Init(argv[0], argv[1], "-restartfromchkpt");

   auto *communicator = pv_init_obj->getCommunicator();
   auto globalComm = communicator->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);
   // Copy output directory from initial stage
   if (status == PV_SUCCESS) {
      status = recursiveCopy(mpiConfigString, "output", communicator);
   }
   if (status == PV_SUCCESS) {
      for (int k = 0; k < commSize; ++k) {
         if (k == commRank and communicator->getIOMPIBlock()->getRank() == 0) {
            for (auto &entry : std::filesystem::directory_iterator("output")) {
               auto filename = entry.path().string();
               auto ext      = extension(filename);
               if (ext == ".log") {
                  std::filesystem::remove(entry.path());
               }
            }
         }
         MPI_Barrier(communicator->globalCommunicator());
      }
   }

   int batchWidth = getBatchWidth(pv_init_obj);
   std::vector<std::uintmax_t> filesizes(batchWidth, static_cast<uintmax_t>(-1));

   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(ioMPIBlock, "output");
   auto initchkptFileManager = std::make_shared<FileManager>(ioMPIBlock, "initializing_checkpoint");

   for (int k = 0; k < commSize; ++k) {
      if (k == commRank and communicator->getIOMPIBlock()->getRank() == 0) {
         for (int b = 0; b < batchWidth; ++b) {
            std::string filename("OutputL2Norm_batchElement_");
            filename.append(std::to_string(b)).append(".txt");
            std::string outputPath = outputFileManager->makeBlockFilename(filename);
            auto outputFileType = std::filesystem::symlink_status(outputPath).type();
            if (outputFileType == std::filesystem::file_type::regular) {
               filesizes[b] = std::filesystem::file_size(outputPath); 
            }
         }
      }
   }
   MPI_Barrier(communicator->globalCommunicator());
   for (int k = 0; k < commSize; ++k) {
      if (k == commRank and communicator->getIOMPIBlock()->getRank() == 0) {
         for (int b = 0; b < batchWidth; ++b) {
            std::string filename("OutputL2Norm_batchElement_");
            filename.append(std::to_string(b)).append(".txt");
            std::string outputPath = outputFileManager->makeBlockFilename(filename);
            auto outputFileType = std::filesystem::symlink_status(outputPath).type();
            bool shouldTruncate = (outputFileType == std::filesystem::file_type::regular);
            if (shouldTruncate) {
               shouldTruncate = filesizes[b] == std::filesystem::file_size(outputPath); 
            }
            if (shouldTruncate) {
               std::string fileposFilename = filename + "_filepos_FileStreamWrite.bin";
               auto filepos = initchkptFileManager->open(fileposFilename, std::ios_base::in);
               filepos->setInPos(0L, std::ios_base::beg);
               long int checkpointedPosition;
               filepos->read(&checkpointedPosition, sizeof(checkpointedPosition));
               filepos = nullptr; // close the file
               long int newSize = checkpointedPosition + 100L;
               std::filesystem::resize_file(outputPath, newSize);
               FatalIf(
                     std::filesystem::file_size(outputPath) != newSize,
                     "resize_file() failed to resize \"%s\" to %ld characters.\n",
                     outputPath.c_str(), newSize);
               FileStream outputFile(outputPath.c_str(), std::ios_base::app);
               char const *extraData = "xxxxxxxx\n";
               outputFile.write(extraData, std::strlen(extraData));
            }
         }
      }
      // Prevent collisions if multiple root processes see the same filesystem
      MPI_Barrier(communicator->globalCommunicator());
   }

   if (status == PV_SUCCESS) {
      status = buildandrun(pv_init_obj.get(), nullptr, nullptr);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("OutputL2Norm", "restartfromchkpt", outputFileManager);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("TotalEnergy", "restartfromchkpt", outputFileManager);
   }
   std::string archiveString = mpiConfigString + "_restartfromchkpt";
   if (status == PV_SUCCESS) {
      int moveStatus = renamePath("output", archiveString, communicator);
      if (moveStatus != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      InfoLog().printf("Restarting from checkpoint passed.\n");
      // Archive log file(s) into output directory
      archiveLogFile(pv_init_obj, archiveString);
   }

   return status;
}

int runRestartFromEnd(int argc, char **argv) {
   int status = PV_SUCCESS;
   InfoLog().printf("Restarting from end...\n");
   auto pv_init_obj = createPV_Init(argv[0], argv[1], "-restartfromend");

   auto *communicator = pv_init_obj->getCommunicator();
   auto globalComm = communicator->globalCommunicator();
   int commSize;
   MPI_Comm_size(globalComm, &commSize);
   int commRank;
   MPI_Comm_rank(globalComm, &commRank);

   std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);

   // Copy output directory from initial stage
   status = recursiveCopy(mpiConfigString, "output", communicator);
   if (status != PV_SUCCESS) { return PV_FAILURE; }

   // Delete log files from copied directory; they came from the base run
   for (int k = 0; k < commSize; ++k) {
      if (k == commRank and communicator->getIOMPIBlock()->getRank() == 0) {
         for (auto &entry : std::filesystem::directory_iterator("output")) {
            auto filename = entry.path().string();
            auto ext      = extension(filename);
            if (ext == ".log") {
               std::filesystem::remove(entry.path());
            }
         }
      }
      MPI_Barrier(communicator->globalCommunicator());
   }

   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   auto outputFileManager = std::make_shared<FileManager>(ioMPIBlock, "output");

   if (status == PV_SUCCESS) {
      status = buildandrun(pv_init_obj.get(), nullptr, nullptr);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("OutputL2Norm", "restartfromend", outputFileManager);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("TotalEnergy", "restartfromend", outputFileManager);
   }
   std::string archiveString = mpiConfigString + "_restartfromend";
   if (status == PV_SUCCESS) {
      int moveStatus = renamePath("output", archiveString, communicator);
      if (moveStatus != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      InfoLog().printf("Restarting from end passed.\n");
      // Archive log file(s) into output directory
      archiveLogFile(pv_init_obj, archiveString);
   }

   return status;
}
