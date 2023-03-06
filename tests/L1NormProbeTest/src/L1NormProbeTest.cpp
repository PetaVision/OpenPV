/*
 * L1NormProbeTest.cpp
 */

#include "CheckValue.hpp"

#include <columns/Communicator.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStream.hpp>
#include <io/FileStreamBuilder.hpp>
#include <structures/MPIBlock.hpp>
#include <utils/PVLog.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace PV;

struct L1NormProbeValues {
   double mTimestamp;
   int mBatchIndex;
   int mNumNeurons;
   double mNorm;
};

int checkOutput(int nbatchGlobal, std::string const &outputPath, Communicator const *communicator);
int checkProbe(
      std::string const &probeNameBase,
      int startElement,
      int stopElement,
      std::shared_ptr<FileManager> fileManager);
int checkL1NormProbe(
      std::string const &probeNameBase,
      int batchIndex,
      std::shared_ptr<FileManager> fileManager);
int compareL1NormFiles(std::string const &correct, std::string const &observed);
int compareL1NormValues(
      L1NormProbeValues const &correct,
      L1NormProbeValues const &observed,
      std::string const &observedPath,
      int linenumber);
void reportFailure(
      int linenumber,
      char const *correctFilename,
      bool correctAtEOF,
      int numReadCorrect,
      char const *observedFilename,
      bool observedAtEOF,
      int numReadObserved);

int run(PV_Init *pv_init);

int main(int argc, char *argv[]) {
   PV_Init *pv_init = new PV_Init(&argc, &argv, false /*value of allowUnrecognizedArguments*/);
   int status       = run(pv_init);
   delete pv_init;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(PV_Init *pv_init) {
   HyPerCol *hc = new HyPerCol(pv_init);
   int status   = hc->run();
   FatalIf(status != PV_SUCCESS, "HyPerCol::run() returned with error code %d\n", status);
   Communicator const *communicator = pv_init->getCommunicator();
   std::string outputPath(hc->getOutputPath());
   int nbatchGlobal = hc->getNBatchGlobal();
   delete hc; // We need to call the HyPerCol destructor before calling checkOutput(),
              // since that is when the L1NormProbe output files are certain to be flushed.

   status = checkOutput(nbatchGlobal, outputPath, communicator);
   return status;
}

int checkOutput(int nbatchGlobal, std::string const &outputPath, Communicator const *communicator) {
   if (communicator->commRank() != 0) {
      return PV_SUCCESS;
   }
   std::shared_ptr<MPIBlock const> ioMPIBlock = communicator->getIOMPIBlock();
   if (ioMPIBlock->getRank() != 0) {
      return PV_SUCCESS;
   }

   int batchDimGlobal = ioMPIBlock->getGlobalBatchDimension();
   int nbatchLocal    = nbatchGlobal / batchDimGlobal;
   FatalIf(
         nbatchLocal * batchDimGlobal != nbatchGlobal,
         "HyPerCol's NBatchGlobal %d is not a multiple of the global batch dimension %d\n",
         nbatchGlobal,
         batchDimGlobal);
   int startElement = nbatchLocal * (ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex());
   int stopElement  = startElement + nbatchLocal * ioMPIBlock->getBatchDimension();
   auto observedValuesFileManager = FileManager::build(ioMPIBlock, outputPath);

   std::vector<std::string> probeNames{"Input", "Output"};
   int status = PV_SUCCESS;
   for (auto const &p : probeNames) {
      int status1 = checkProbe(p, startElement, stopElement, observedValuesFileManager);
      if (status1 != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

int checkProbe(
      std::string const &probeNameBase,
      int startElement,
      int stopElement,
      std::shared_ptr<FileManager> fileManager) {
   int status = PV_SUCCESS;
   for (int b = startElement; b < stopElement; ++b) {
      if (checkL1NormProbe(probeNameBase, b, fileManager) != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

// Compares two files with filename [probeNameBase]L1Norm_batchElement_[b].txt,
// one of which is in the input directory, and the other is in the block directory
// used by the specified fileManager.
int checkL1NormProbe(
      std::string const &probeNameBase,
      int batchIndex,
      std::shared_ptr<FileManager> fileManager) {
   std::string filename = probeNameBase + ("L1Norm_batchElement_");
   filename.append(std::to_string(batchIndex)).append(".txt");

   std::string correctValuesPath("input/");
   correctValuesPath.append(filename);

   std::shared_ptr<FileStream> observedL1NormStream = FileStreamBuilder(
                                                            fileManager,
                                                            filename,
                                                            true /*isText*/,
                                                            true /*readOnlyFlag*/,
                                                            false /*clobberFlag*/,
                                                            false /*verifyWrites*/)
                                                            .get();
   std::string observedValuesPath = observedL1NormStream->getFileName();
   observedL1NormStream->close();
   observedL1NormStream = nullptr;

   int status = compareL1NormFiles(correctValuesPath, observedValuesPath);
   return status;
}

int compareL1NormFiles(std::string const &correctPath, std::string const &observedPath) {
   FILE *correctfp  = fopen(correctPath.c_str(), "r");
   FILE *observedfp = fopen(observedPath.c_str(), "r");
   FatalIf(!correctfp, "compareL1NormFiles unable to open %s for reading.\n", correctPath.c_str());
   FatalIf(
         !observedfp, "compareL1NormFiles unable to open %s for reading.\n", observedPath.c_str());

   L1NormProbeValues correctValues, observedValues;
   int status     = PV_SUCCESS;
   int linenumber = 0;
   while (true) {
      ++linenumber;
      int numReadCorrect = fscanf(
            correctfp,
            "%lf, %d, %d, %lf\n",
            &correctValues.mTimestamp,
            &correctValues.mBatchIndex,
            &correctValues.mNumNeurons,
            &correctValues.mNorm);
      int numReadObserved = fscanf(
            observedfp,
            "%lf, %d, %d, %lf\n",
            &observedValues.mTimestamp,
            &observedValues.mBatchIndex,
            &observedValues.mNumNeurons,
            &observedValues.mNorm);
      if (numReadCorrect == 4 && numReadObserved == 4) {
         int status1 = compareL1NormValues(correctValues, observedValues, observedPath, linenumber);
         if (status1 != PV_SUCCESS) {
            status = PV_FAILURE;
         }
         continue;
      }
      else {
         if (feof(correctfp) && feof(observedfp) && numReadCorrect == -1 && numReadObserved == -1) {
            break;
         }
         else {
            reportFailure(
                  linenumber,
                  correctPath.c_str(),
                  feof(correctfp),
                  numReadCorrect,
                  observedPath.c_str(),
                  feof(observedfp),
                  numReadObserved);
            status = PV_FAILURE;
            break;
         }
      }
   }
   fclose(correctfp);
   fclose(observedfp);
   return status;
}

int compareL1NormValues(
      L1NormProbeValues const &correct,
      L1NormProbeValues const &observed,
      std::string const &observedPath,
      int linenumber) {
   int status = PV_SUCCESS;
   std::string const description(observedPath + ":" + std::to_string(linenumber));
   try {
      checkValue(description, std::string("time"), observed.mTimestamp, correct.mTimestamp, 0.0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(
            description, std::string("batch index"), observed.mBatchIndex, correct.mBatchIndex, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(description, std::string("N"), observed.mNumNeurons, correct.mNumNeurons, 0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   try {
      checkValue(description, std::string("Norm"), observed.mNorm, correct.mNorm, 0.0);
   } catch (std::exception const &e) {
      ErrorLog() << e.what();
      status = PV_FAILURE;
   }
   return status;
}

void reportFailure(
      int linenumber,
      char const *correctFilename,
      bool correctAtEOF,
      int numReadCorrect,
      char const *observedFilename,
      bool observedAtEOF,
      int numReadObserved) {
   if (correctAtEOF && !observedAtEOF) {
      ErrorLog().printf(
            "%s reached end of file before %s (line %d)\n",
            correctFilename,
            observedFilename,
            linenumber);
   }
   else if (!correctAtEOF && observedAtEOF) {
      ErrorLog().printf(
            "%s reached end of file before %s (line %d)\n",
            observedFilename,
            correctFilename,
            linenumber);
   }
   else if (!correctAtEOF && !observedAtEOF) {
      if (numReadCorrect != 4) {
         ErrorLog().printf(
               "%s only read %d values instead of the expected 4 (line %d).\n",
               correctFilename,
               numReadCorrect,
               linenumber);
      }
      if (numReadObserved != 4) {
         ErrorLog().printf(
               "%s only read %d values instead of the expected 4 (line %d).\n",
               observedFilename,
               numReadObserved,
               linenumber);
      }
   }
   else {
      ErrorLog().printf(
            "Unexpected failure comparing line %d of %s to that of %s\n",
            linenumber,
            observedFilename,
            correctFilename);
   }
}
