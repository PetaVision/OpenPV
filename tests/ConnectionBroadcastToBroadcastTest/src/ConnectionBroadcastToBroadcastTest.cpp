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

int checkFrame(
      std::shared_ptr<FileStream> weightsStream,
      int nfPre,
      int nfPost,
      int nbatch,
      int frameNumber,
      int displayPeriod);
bool checkHeaderValue(int observed, int correct, char const *headerDesc, char const *valueDesc);
int checkResults(PV_Init &pv_initObj);
int runParams(PV_Init &pv_initObj, char const *paramsFile);

int main(int argc, char *argv[]) {
   // Run params file
   PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   FatalIf(pv_initObj.getParams() != nullptr, "%s must be run without a params file.\n", argv[0]);

   int status = PV_SUCCESS;

   char const *baseParamsFile = "input/ConnectionBroadcastToBroadcastTest.params";
   status = runParams(pv_initObj, baseParamsFile);
   FatalIf(status != PV_SUCCESS, "Run with params \"%s\" failed.\n", baseParamsFile);

   // Only root processes need to check the output
   auto outputFileManager = pv_initObj.getCommunicator()->getOutputFileManager();
   if (outputFileManager->isRoot()) {
      status = checkResults(pv_initObj);
   }
   FatalIf(
         status != PV_SUCCESS, "Checking output from params file \"%s\" failed.\n", baseParamsFile);



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

int checkFrame(
      std::shared_ptr<FileStream> weightsStream,
      int nfPre,
      int nfPost,
      int nbatch,
      int frameNumber,
      int displayPeriod) {
   int status = PV_SUCCESS;
   std::string fileDescription("File \"" + weightsStream->getFileName() + "\"");
   BufferUtils::WeightHeader header;
   weightsStream->read(&header, sizeof(BufferUtils::WeightHeader));
   if (checkHeaderValue(header.baseHeader.nx, 1, fileDescription.c_str(), "nx") == false) {
      status = PV_FAILURE;
   }
   if (checkHeaderValue(header.baseHeader.ny, 1, fileDescription.c_str(), "ny") == false) {
      status = PV_FAILURE;
   }
   if (checkHeaderValue(header.baseHeader.nf, nfPre, fileDescription.c_str(), "nf") == false) {
      status = PV_FAILURE;
   }
   if (checkHeaderValue(header.nxp, 1, fileDescription.c_str(), "nxp") == false) {
      status = PV_FAILURE;
   }
   if (checkHeaderValue(header.nyp, 1, fileDescription.c_str(), "nyp") == false) {
      status = PV_FAILURE;
   }
   if (checkHeaderValue(header.nfp, nfPost, fileDescription.c_str(), "nfp") == false) {
      status = PV_FAILURE;
   }
   FatalIf(
         status != PV_SUCCESS,
         "%s frame %ld does not have the correct header.\n",
         fileDescription.c_str(), frameNumber);

   // Read weights, sanity-checking patch headers as we go.
   // There should be nfPre patches, each with nfPost values.
   // Each patch should have a header with nx=1, ny=1, offset=0.
   std::vector<std::vector<float>> values(nfPre);
   for (auto &v : values) { v.resize(nfPost); }
   for (int patch = 0; patch < nfPre; ++patch) {
      std::string patchDesc("#1 frame #2, patch #3");
      patchDesc.replace(patchDesc.find("#3"), 2, std::to_string(patch));
      patchDesc.replace(patchDesc.find("#2"), 2, std::to_string(frameNumber));
      patchDesc.replace(patchDesc.find("#1"), 2, fileDescription);

      Patch patchHeader;
      weightsStream->read(&patchHeader, sizeof(Patch));
      if (checkHeaderValue(patchHeader.nx, 1, patchDesc.c_str(), "nx") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(patchHeader.ny, 1, patchDesc.c_str(), "ny") == false) {
         status = PV_FAILURE;
      }
      if (checkHeaderValue(patchHeader.offset, 0, patchDesc.c_str(), "offset") == false) {
         status = PV_FAILURE;
      }
      FatalIf(status != PV_SUCCESS, "%s does not have the correct patch header.\n", patchDesc.c_str());
      weightsStream->read(values[patch].data(), nfPost * sizeof(float));
   }
   
   // Check the weight values.
   for (int patch = 0; patch < nfPre; ++patch) {
      for (int elem = 0; elem < nfPost; ++elem) {
         float observed = values[patch][elem];
         float correct = static_cast<float>( (patch + 1) * (elem + 1) * nbatch * displayPeriod);
         if (observed != correct) {
            ErrorLog().printf(
                  "Frame %ld, patch %d, element %d is %f instead of correct value %f\n",
                  frameNumber, patch, elem,
                  static_cast<double>(observed), static_cast<double>(correct));
            status = PV_FAILURE;
         }
      }
   }
   FatalIf(status != PV_SUCCESS, "%s does not have the correct values.\n", fileDescription.c_str());
   return status;
}

int checkResults(PV_Init &pv_initObj) {
   int status = PV_SUCCESS;
   // Read params values we'll need later
   PVParams *params = pv_initObj.getParams();
   int nfPre  = params->valueInt("Pre", "nf");
   int nfPost = params->valueInt("Post", "nf");
   int nbatch = params->valueInt("column", "nbatch");

   // Check the weights file
   std::string weightsPath("output/PreToPost.pvp");
   std::string weightsFileDesc("File \"output/PreToPost.pvp\"");
   auto weightsStream = std::make_shared<FileStream>(weightsPath.c_str(), std::ios_base::in);
   FatalIf(!(*weightsStream), "Unable to open \"%s\" for reading.\n", weightsPath.c_str());

   long weightsFileSize = weightsStream->getFileSize();
   int frameCount = 0;
   while (weightsStream->getInPos() < weightsFileSize) {
      ++frameCount;
      status = checkFrame(weightsStream, nfPre, nfPost, nbatch, frameCount, frameCount);
   }
   weightsStream = nullptr; // close the weights file

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
   std::string checkpointFileDesc("File \"" + checkpointPath + "\"");
   auto checkpointStream = std::make_shared<FileStream>(checkpointPath.c_str(), std::ios_base::in);
   FatalIf(!(*checkpointStream), "Unable to open \"%s\" for reading.\n", checkpointPath.c_str());

   // Check the checkpointed file
   status = checkFrame(checkpointStream, nfPre, nfPost, nbatch, 1, frameCount);
   checkpointStream = nullptr; // close the checkpoints file

   return status;
}

int runParams(PV_Init &pv_initObj, char const *paramsFile) {
   pv_initObj.setParams("input/ConnectionBroadcastToBroadcastTest.params");
   int status = buildandrun(&pv_initObj, nullptr, nullptr);
   return status;
}
