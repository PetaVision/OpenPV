/*
 * RandomInputActivityTest.cpp
 *
 * Runs the params files
 *   RandomInputActivity0.params
 *   RandomInputActivity1.params
 *   RandomInputActivity2.params
 * in sequence. RandomInputAcitivityTest0.params contains a PvpLayer with batchMethod = Random.
 * It runs up to time t=20, with output written to the output0 directory.
 * RandomInputActivity1.params is the same, except it stops at t=10, and writes to the
 * output1 directory.
 * RandomInputActivity2.params is the same as RandomInputActivity1.params, except that
 * the ending time is t=20. This program runs RandomInputActivity2.params with the
 * CheckpointReadDirectory option set to output1/Checkpoints/Checkpoint10. Hence it continues the
 * run where RandomInputActivityTest1.params left off.
 *
 * Therefore, if everything is working, the Input_A.pvp files in checkpoint20 should be the same
 * in both output0 and output1. The checkOutput() function loads these two files and compares the
 * output. The test passes if they are equal. There is no resizing or rescaling or anything else
 * that would cause issues with floating point precision, so the values should be exactly equal.
 */

#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <structures/Buffer.hpp>
#include <utils/BufferUtilsPvp.hpp>
#include <utils/PVLog.hpp>

#include <string>

int checkOutput(PV_Init &pv_initObj);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;
   PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   FatalIf(
         pv_initObj.getParams() != nullptr,
         "%s needs to be run without a params file argument. "
         "The necessary params files are hard-coded.\n",
         argv[0]);
   std::string logFile = pv_initObj.getStringArgument("LogFile");

   if (status == PV_SUCCESS) {
      pv_initObj.setParams("input/RandomInputActivityTest0.params");
      status = buildandrun(&pv_initObj);
   }

   if (status == PV_SUCCESS) {
      pv_initObj.setParams("input/RandomInputActivityTest1.params");
      pv_initObj.setLogFile(logFile.c_str(), true /*appendFlag*/);
      status = buildandrun(&pv_initObj);
   }

   if (status == PV_SUCCESS) {
      pv_initObj.setParams("input/RandomInputActivityTest2.params");
      pv_initObj.setLogFile(logFile.c_str(), true /*appendFlag*/);
      pv_initObj.setStringArgument("CheckpointReadDirectory", "output1/Checkpoints/Checkpoint10");
      status = buildandrun(&pv_initObj);
   }

   if (status == PV_SUCCESS) {
      status = checkOutput(pv_initObj);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkOutput(PV_Init &pv_initObj) {
   if (pv_initObj.getCommunicator()->getIOMPIBlock()->getRank() != 0) {
      return PV_SUCCESS;
   }

   int status = PV_SUCCESS;
   std::string path0 = "output0/Checkpoints/Checkpoint20/Input_A.pvp";
   std::string path1 = "output1/Checkpoints/Checkpoint20/Input_A.pvp";
   Buffer<float> buffer0, buffer1;
   for (int b = 0; b < 2/*nbatch*/; ++b) {
      BufferUtils::readActivityFromPvp(path0.c_str(), &buffer0, b, nullptr /*sparseFileTable*/);
      BufferUtils::readActivityFromPvp(path1.c_str(), &buffer1, b, nullptr /*sparseFileTable*/);
      FatalIf(
            buffer0.getWidth() != 4 or buffer0.getHeight() != 4 or buffer0.getFeatures() != 1,
            "%s has dimensions %d-by-%d-by-%d instead of expected 4-by-4-by-1.\n",
            path0.c_str(), buffer0.getWidth(), buffer0.getHeight(), buffer0.getFeatures());
      FatalIf(
            buffer1.getWidth() != 4 or buffer1.getHeight() != 4 or buffer1.getFeatures() != 1,
            "%s has dimensions %d-by-%d-by-%d instead of expected 4-by-4-by-1.\n",
            path1.c_str(), buffer1.getWidth(), buffer1.getHeight(), buffer1.getFeatures());
      for (int n = 0; n < 16; ++n) {
         float v0 = buffer0.at(n);
         float v1 = buffer1.at(n);
         if (v1 != v0) {
            ErrorLog().printf(
                  "Batch element %d, LCA neuron %4d: expected %f, observed %f, discrepancy %g\n",
                  b, n, double(v0), double(v1), double(v1 - v0));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}
