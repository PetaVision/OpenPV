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

void archiveLogFile();
int compareFileContents(std::shared_ptr<FileStream> observed, std::shared_ptr<FileStream> correct);
int compareProbeOutput(
      std::string const &probeName,
      std::string const &correctDir,
      std::shared_ptr<FileManager> fileManager);
int compareWords(std::string const &observedWord, std::string const &correctWord);
std::string generateMPIConfigString(Communicator *communicator);
std::string getWord(std::string const &str, std::string::size_type &pos);
std::string readFileContents(std::shared_ptr<FileStream> fileStream);
int run(int argc, char **argv);
int runBase(PV_Init *pv_init_obj);
int runInitFromCheckpoint(PV_Init *pv_init_obj);
int runRestartFromCheckpoint(PV_Init *pv_init_obj);
int runRestartFromEnd(PV_Init *pv_init_obj);

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);

   int result = run(argc, argv);

   MPI_Finalize();

   return result == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void archiveLogFile(PV_Init *pv_init_obj, std::string const &destDir) {
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
               observed->getFileName().c_str(), lineNumber, fieldNumber);
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
   auto pv_init_obj = new PV_Init(&argc, &argv, true /*allowUnrecognizedArgumentsFlag*/);

   if (status == PV_SUCCESS) {
      status = runBase(pv_init_obj);
   }
   if (runInitFromCheckpoint(pv_init_obj) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (runRestartFromCheckpoint(pv_init_obj) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (runRestartFromEnd(pv_init_obj) != PV_SUCCESS) {
      status = PV_FAILURE;
   }

   delete pv_init_obj;
   return status;
}

int runBase(PV_Init *pv_init_obj) {
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
      status = buildandrun(pv_init_obj, nullptr, nullptr);
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
      InfoLog().printf("Base run passed.\n");
      // Archive log file(s) into output directory
      archiveLogFile(pv_init_obj, mpiConfigString);
   }
   return status;
}

int runInitFromCheckpoint(PV_Init *pv_init_obj) {
   int status = PV_SUCCESS;

   // Switch log file and params file for initializing from checkpoint
   std::string initfromchkptLogFile;
   std::string const &logFile = pv_init_obj->getStringArgument("LogFile");
   if (!logFile.empty()) {
      std::string logFileDir  = dirName(logFile);
      std::string logFileBase = stripExtension(logFile);
      std::string logFileExt  = extension(logFile);
      initfromchkptLogFile = logFileDir + '/' + logFileBase + "_initfromchkpt" + logFileExt;
   }
   pv_init_obj->setLogFile(initfromchkptLogFile.c_str());
   pv_init_obj->setParams("input/initfromchkpt.params");

   auto *communicator = pv_init_obj->getCommunicator();
   std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);
   if (status == PV_SUCCESS) {
      InfoLog().printf("Initializing from checkpoint...\n");
      status = recursiveDelete("initializing_checkpoint", communicator, false /*warnIfAbsentFlag*/);
   }
   if (status == PV_SUCCESS) {
      std::string initChkptDir = mpiConfigString + "/checkpoints/Checkpoint040";
      status = recursiveCopy(initChkptDir, "initializing_checkpoint", communicator);
   }
   if (status == PV_SUCCESS) {
      status = recursiveCopy(mpiConfigString, "output", communicator);
   }
   if (status == PV_SUCCESS) {
      for (auto &entry : std::filesystem::recursive_directory_iterator("output")) {
         // If not match TotalEnergy_batchelement_[0-9][0-9]*.txt, delete the file
         // (and if file is a directory, delete its contents)
         auto filestatus = std::filesystem::symlink_status(entry);
         if (filestatus.type() != std::filesystem::file_type::regular) { continue; }
         std::string fileString = entry.path().filename().string();
         std::string head("TotalEnergy_batchElement_");
         std::string tail(".txt");
         auto headFound = fileString.find(head);
         auto tailFound = fileString.rfind(tail);
         bool matches = (headFound == 0 and tailFound + tail.size() == fileString.size());
         if (matches) {
            for (auto pos = head.size() ; pos < tailFound; ++pos) {
               matches &= isdigit(fileString[pos]);
            }
         }
         if (matches) {
            if (appendExtraneousData(entry.path(), communicator) != PV_SUCCESS) {
               status = PV_FAILURE;
            }
         }
         else {
            status = recursiveDelete(
                  entry.path().string(), communicator, false /*warnIfAbsentFlag*/);
            if (status != PV_SUCCESS) {
               ErrorLog() << "Unable to delete files from directory \"output\"\n";
            }
         }
      }
   }
   if (status == PV_SUCCESS) {
      status = buildandrun(pv_init_obj, nullptr, nullptr);
   }
   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   auto fileManager = std::make_shared<FileManager>(ioMPIBlock, "output");
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("OutputL2Norm", "initfromchkpt", fileManager);
   }
   if (status == PV_SUCCESS) {
      status = compareProbeOutput("TotalEnergy", "initfromchkpt", fileManager);
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

int runRestartFromCheckpoint(PV_Init *pv_init_obj) {
   int status = PV_SUCCESS;
   return status;
}

int runRestartFromEnd(PV_Init *pv_init_obj) {
   int status = PV_SUCCESS;
   return status;
}
