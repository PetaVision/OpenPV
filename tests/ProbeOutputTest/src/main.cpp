// Code for the ProbeOutputTest system test. It does a PetaVision run with probe output, and then
// reads the output back to check fro correctness.
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
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace PV;

int compareFileContents(std::shared_ptr<FileStream> observed, std::shared_ptr<FileStream> correct);
std::string generateMPIConfigString(Communicator *communicator);
int run(int argc, char **argv);

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);

   int result = run(argc, argv);

   MPI_Finalize();

   return result == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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

int compareFileContents(std::shared_ptr<FileStream> observed, std::shared_ptr<FileStream> correct) {
   int status = PV_SUCCESS;
   observed->setInPos(0L, std::ios_base::end);
   long int observedSize = observed->getInPos();
   observed->setInPos(0L, std::ios_base::beg);

   correct->setInPos(0L, std::ios_base::end);
   long int correctSize = correct->getInPos();
   correct->setInPos(0L, std::ios_base::beg);

   std::string observedContents(observedSize, '\0'), correctContents(correctSize, '\0');
   observed->read(&observedContents.at(0), observedSize);
   correct->read(&correctContents.at(0), correctSize);

   std::stringstream observedStream(observedContents, std::ios_base::in);
   std::stringstream correctStream(correctContents, std::ios_base::in);

   int lineNumber = 1;
   int fieldNumber = 1;
   while (true) {
      double observedValue;
      char observedStop;
      observedStream >> observedValue;
      observedStream.get(observedStop);
      double correctValue;
      char correctStop;
      correctStream >> correctValue;
      correctStream.get(correctStop);
      if (observedStream.good() and correctStream.good()) {
         double discrepancy = std::abs(observedValue - correctValue);
         if (discrepancy > 1.0e-6 * std::abs(correctValue)) {
            ErrorLog().printf(
                  "Line %d, field %d has significantly different probe values: "
                  "correct value %f, observed value %f (discrepancy %g)\n",
                   lineNumber, fieldNumber, correctValue, observedValue);
            status = PV_FAILURE;
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
                  "Observed file differs from correct file delimeter, after line %d, field %d\n",
                  lineNumber, fieldNumber);
            status = PV_FAILURE;
            break;
         }
      }
      else if (observedStream.eof() and correctStream.eof()) {
         break;
      }
      else {
         assert(observedStream.fail() or correctStream.fail());
         if (observedStream.fail()) {
            ErrorLog().printf(
                  "Observed values file \"%s\" failed at line %d, field %d\n",
                  observed->getFileName().c_str(), lineNumber, fieldNumber);
         }
         if (correctStream.fail()) {
            ErrorLog().printf(
                  "Correct values file \"%s\" failed at line %d, field %d\n",
                  correct->getFileName().c_str(), lineNumber, fieldNumber);
         }
         status = PV_FAILURE;
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

int run(int argc, char **argv) {
   int status = PV_SUCCESS;
   auto pv_init_obj = new PV_Init(&argc, &argv, true /*allowUnrecognizedArgumentsFlag*/);

   auto *communicator = pv_init_obj->getCommunicator();

   // Delete any existing output directory, to be sure that any files
   // are the result of this run and not left over from previous runs
   status = recursiveDelete("output", communicator, false /*warnIfAbsentFlag*/);

   InfoLog().printf("Executing base run\n");
   status = buildandrun(pv_init_obj, nullptr, nullptr);

   // Move log file into output directory
   int globalRank = communicator->globalCommRank();
   std::string const &logFile = pv_init_obj->getStringArgument("LogFile");
   std::string logPath;
   if (!logFile.empty()) {
      std::string destPath("output/");
      if (globalRank != 0) {
         std::string directory     = dirName(logFile);
         std::string stripExt      = stripExtension(logFile);
         std::string fileExt       = extension(logFile);
         std::string logFileName   = stripExt + '_' + std::to_string(globalRank) + fileExt;
         logPath                   = directory + '/' + logFileName;
         destPath += logFileName;
      }
      else {
         std::string directory = dirName(logFile);
         std::string fileName  = baseName(logFile);
         logPath               = directory + '/' + fileName;
         destPath += fileName;
      }
      ::rename(logPath.c_str(), destPath.c_str());
   }

   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   auto fileManager = std::make_shared<FileManager>(ioMPIBlock, "output");
   int batchWidth = 8;
   std::vector<int> statusByBatchElement(batchWidth, PV_FAILURE);
   if (fileManager->isRoot()) {
      for (int b = 0; b < batchWidth; ++b) {
         std::string fileName("OutputL2Norm_batchElement_#.txt");
         fileName.replace(fileName.find("#"), 1, std::to_string(b));
         struct stat statbuf;
         int statstatus = fileManager->stat(fileName, statbuf);
         if (statstatus != 0) {
            if (errno != ENOENT) {
               status = PV_FAILURE;
               ErrorLog().printf("stat(\"%s\") failed: %s\n", fileName, strerror(errno));
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
         std::string correctDataPath("input/correctProbeOutput/base/correct_OutputL2Norm_#.txt");
         correctDataPath.replace(correctDataPath.find("#"), 1, std::to_string(b));
         auto correctL2Norm = std::make_shared<FileStream>(correctDataPath.c_str(), std::ios_base::in);
         statusByBatchElement[b] = compareFileContents(outputL2Norm, correctL2Norm);
      }
   }
   std::string mpiConfigString = std::string("output_") + generateMPIConfigString(communicator);
   int moveStatus = renamePath("output", mpiConfigString, communicator);
   if (moveStatus != PV_SUCCESS) {
      status = PV_FAILURE;
   }

   delete pv_init_obj;
   return status;
}
