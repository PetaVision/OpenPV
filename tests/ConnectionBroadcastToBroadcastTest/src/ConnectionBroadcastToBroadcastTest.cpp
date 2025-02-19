/*
 * ConnectionBroadcastToBroadcastTest.cpp
 */

#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <io/FileManager.hpp>
#include <structures/Patch.hpp>
#include <utils/BufferUtilsPvp.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ios>
#include <memory>
#include <vector>

using namespace PV;

bool checkHeaderValue(int observed, int correct, char const *headerDesc, char const *valueDesc);

int main(int argc, char *argv[]) {
   // Run params file
   PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = buildandrun(&pv_initObj, nullptr, nullptr);
   FatalIf(status != PV_SUCCESS, "PetaVision run failed.\n");

   // From now on, we'll read the output file back; hence only root processes need to continue
   auto outputFileManager = pv_initObj.getCommunicator()->getOutputFileManager();
   if (!outputFileManager->isRoot()) {
      return status;
   }

   // Read params values we'll need later
   PVParams *params = pv_initObj.getParams();
   int nfPre  = params->valueInt("Pre", "nf");
   int nfPost = params->valueInt("Post", "nf");
   int nbatch = params->valueInt("column", "nbatch");

   // Find the last checkpoints directory
   char const *lastCheckpointDir = params->stringValue("column", "lastCheckpointDir");
   std::string checkpointDir;
   if (lastCheckpointDir) {
      checkpointDir = lastCheckpointDir;
   }
   else {
      char const *writeCheckpointDir = params->stringValue("column", "checkpointWriteDir");
      float stopTime = params->value("column", "stopTime");
      float dt = params->value("column", "dt");
      int numSteps = static_cast<int>(std::round(stopTime / dt));
      checkpointDir = std::string(writeCheckpointDir) + "/Checkpoint" + std::to_string(numSteps);
   }
   std::string checkpointPath(checkpointDir + "/" + "PreToPost_W.pvp");

   // Open the weights file
   std::string weightsPath("output/PreToPost.pvp");
   std::string weightsFileDesc("File \"output/PreToPost.pvp\"");
   auto weightsStream = std::make_shared<FileStream>(weightsPath.c_str(), std::ios_base::in);
   FatalIf(!(*weightsStream), "Unable to open \"%s\" for reading.\n", weightsPath.c_str());

   long weightsFileSize = weightsStream->getFileSize();
   int frameCount = 0;
   while (weightsStream->getInPos() < weightsFileSize) {
      ++frameCount;

      // Read header and check important values
      BufferUtils::WeightHeader header;
      weightsStream->read(&header, sizeof(BufferUtils::WeightHeader));
      if (checkHeaderValue(header.baseHeader.nx, 1, weightsFileDesc.c_str(), "nx") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.baseHeader.ny, 1, weightsFileDesc.c_str(), "ny") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.baseHeader.nf, nfPre, weightsFileDesc.c_str(), "nf") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.nxp, 1, weightsFileDesc.c_str(), "nxp") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.nyp, 1, weightsFileDesc.c_str(), "nyp") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.nfp, nfPost, weightsFileDesc.c_str(), "nfp") == false) {
         status = PV_FAILURE;
      }
      FatalIf(
            status != PV_SUCCESS,
            "%s frame %ld does not have the correct header.\n",
            weightsFileDesc.c_str(), frameCount);

      // Read weights, sanity-checking patch headers as we go.
      // There should be nfPre patches, each with nfPost values.
      // Each patch should have a header with nx=1, ny=1, offset=0.
      std::vector<std::vector<float>> values(nfPre);
      for (auto &v : values) { v.resize(nfPost); }
      for (int patch = 0; patch < nfPre; ++patch) {
         std::string patchDesc("File \"#1\" frame #2, patch #3");
         patchDesc.replace(patchDesc.find("#3"), 2, std::to_string(patch));
         patchDesc.replace(patchDesc.find("#2"), 2, std::to_string(frameCount));
         patchDesc.replace(patchDesc.find("#1"), 2, weightsPath);

         Patch patchHeader;
         weightsStream->read(&patchHeader, sizeof(Patch));
         if (checkHeaderValue(patchHeader.nx, 1, patchDesc.c_str(), "nx") == false) {
            status = PV_FAILURE;
         }
         if (checkHeaderValue(patchHeader.ny, 1, patchDesc.c_str(), "nx") == false) {
            status = PV_FAILURE;
         }
         if (checkHeaderValue(patchHeader.offset, 0, patchDesc.c_str(), "nx") == false) {
            status = PV_FAILURE;
         }
         FatalIf(status != PV_SUCCESS, "%s does not have the correct patch header.\n", patchDesc.c_str());
         weightsStream->read(values[patch].data(), nfPost * sizeof(float));
      }
   
      // Check the weight values.
      for (int patch = 0; patch < nfPre; ++patch) {
         for (int elem = 0; elem < nfPost; ++elem) {
            float observed = values[patch][elem];
            float correct = static_cast<float>( (patch + 1) * (elem + 1) * nbatch * frameCount);
            if (observed != correct) {
               ErrorLog().printf(
                     "Frame %ld, patch %d, element %d is %f instead of correct value %f\n",
                     frameCount, patch, elem,
                     static_cast<double>(observed), static_cast<double>(correct));
               status = PV_FAILURE;
            }
         }
      }
      FatalIf(status != PV_SUCCESS, "%s does not have the correct values.\n", weightsFileDesc.c_str());
   }
   weightsStream = nullptr; // close the weights file

   std::string checkpointFileDesc("File \"" + checkpointPath + "\"");
   auto checkpointStream = std::make_shared<FileStream>(checkpointPath.c_str(), std::ios_base::in);
   FatalIf(!(*checkpointStream), "Unable to open \"%s\" for reading.\n", checkpointPath.c_str());
   {
      BufferUtils::WeightHeader header;
      checkpointStream->read(&header, sizeof(BufferUtils::WeightHeader));
      if (checkHeaderValue(header.baseHeader.nx, 1, checkpointFileDesc.c_str(), "nx") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.baseHeader.ny, 1, checkpointFileDesc.c_str(), "ny") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.baseHeader.nf, nfPre, checkpointFileDesc.c_str(), "nf") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.nxp, 1, checkpointFileDesc.c_str(), "nxp") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.nyp, 1, checkpointFileDesc.c_str(), "nyp") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(header.nfp, nfPost, checkpointFileDesc.c_str(), "nfp") == false) {
         status = PV_FAILURE;
      }
      FatalIf(
            status != PV_SUCCESS,
            "%s frame %ld does not have the correct header.\n",
            checkpointFileDesc.c_str(), frameCount);

      // Read weights, sanity-checking patch headers as we go.
      // There should be nfPre patches, each with nfPost values.
      // Each patch should have a header with nx=1, ny=1, offset=0.
      std::vector<std::vector<float>> values(nfPre);
      for (auto &v : values) { v.resize(nfPost); }
      for (int patch = 0; patch < nfPre; ++patch) {
         std::string patchDesc("File \"#1\" frame #2, patch #3");
         patchDesc.replace(patchDesc.find("#3"), 2, std::to_string(patch));
         patchDesc.replace(patchDesc.find("#2"), 2, std::to_string(frameCount));
         patchDesc.replace(patchDesc.find("#1"), 2, weightsPath);

         Patch patchHeader;
         checkpointStream->read(&patchHeader, sizeof(Patch));
         if (checkHeaderValue(patchHeader.nx, 1, patchDesc.c_str(), "nx") == false) {
            status = PV_FAILURE;
         }
         if (checkHeaderValue(patchHeader.ny, 1, patchDesc.c_str(), "nx") == false) {
            status = PV_FAILURE;
         }
         if (checkHeaderValue(patchHeader.offset, 0, patchDesc.c_str(), "nx") == false) {
            status = PV_FAILURE;
         }
         FatalIf(status != PV_SUCCESS, "%s does not have the correct patch header.\n", patchDesc.c_str());
         checkpointStream->read(values[patch].data(), nfPost * sizeof(float));
      }
   
      // Check the weight values.
      for (int patch = 0; patch < nfPre; ++patch) {
         for (int elem = 0; elem < nfPost; ++elem) {
            float observed = values[patch][elem];
            float correct = static_cast<float>( (patch + 1) * (elem + 1) * nbatch * frameCount);
            if (observed != correct) {
               ErrorLog().printf(
                     "Frame %ld, patch %d, element %d is %f instead of correct value %f\n",
                     frameCount, patch, elem,
                     static_cast<double>(observed), static_cast<double>(correct));
               status = PV_FAILURE;
            }
         }
      }
      FatalIf(status != PV_SUCCESS, "%s does not have the correct values.\n", checkpointFileDesc.c_str());
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

bool checkHeaderValue(int observed, int correct, char const *headerDesc, char const *valueDesc) {
   if (observed != correct) {
      ErrorLog().printf(
            "%s header has %s=%d instead of correct value %d\n",
            headerDesc, valueDesc, observed, correct);
      return false;
   }
   return true;
}
