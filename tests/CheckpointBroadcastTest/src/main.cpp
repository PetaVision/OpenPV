/*
 * main.cpp for CheckpointBroadcastTest
 *
 * Runs the params files
 *   CheckpointBroadcastTest0.params
 *   CheckpointBroadcastTest1.params
 *   CheckpointBroadcastTest2.params
 * in sequence. CheckpointBroadcastTest0.params performs the LCA algorithm with a broadcast layer
 * for V1, up to time t=20, with output written to the output0 directory.
 * CheckpointBroadcastTest1.params performs the same algorithm on the same input, up to time t=10,
 * with output written to the output1 directory.
 * CheckpointBroadcastTest2.params is the same as CheckpointBroadcastTest1.params, except that
 * the ending time is t=20. This program runs CheckpointBroadcastTest2.params with the
 * CheckpointReadDirectory option set to output1/checkpoints/Checkpoint10. Hence it continues the
 * run where CheckpointBroadcastTest1.params left off.
 *
 * Therefore, if restarting a broadcast layer from checkpoint is working (as well as everything
 * else the params files depend on), the V1_V.pvp files in Checkpoint20 should be the same in
 * both output0 and output1. The checkOutput() function loads these two files and compares the
 * output. The test passes if they are equal within a hard-coded tolerance of 1e-6
 * (TODO eliminate the hard-coding for the tolerance).
 *
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
      pv_initObj.setParams("input/CheckpointBroadcastTest0.params");
      status = buildandrun(&pv_initObj);
   }

   if (status == PV_SUCCESS) {
      pv_initObj.setParams("input/CheckpointBroadcastTest1.params");
      pv_initObj.setLogFile(logFile.c_str(), true /*appendFlag*/);
      status = buildandrun(&pv_initObj);
   }

   if (status == PV_SUCCESS) {
      pv_initObj.setParams("input/CheckpointBroadcastTest2.params");
      pv_initObj.setLogFile(logFile.c_str(), true /*appendFlag*/);
      pv_initObj.setStringArgument("CheckpointReadDirectory", "output1/checkpoints/Checkpoint10");
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
   std::string path0 = "output0/checkpoints/Checkpoint20/V1_V.pvp";
   std::string path1 = "output1/checkpoints/Checkpoint20/V1_V.pvp";
   Buffer<float> buffer0, buffer1;
   for (int b = 0; b < 4/*nbatch*/; ++b) {
      BufferUtils::readActivityFromPvp(path0.c_str(), &buffer0, b, nullptr /*sparseFileTable*/);
      BufferUtils::readActivityFromPvp(path1.c_str(), &buffer1, b, nullptr /*sparseFileTable*/);
      FatalIf(
            buffer0.getWidth() != 1 or buffer0.getHeight() != 1 or buffer0.getFeatures() != 1024,
            "%s has dimensions %d-by-%d-by-%d instead of expected 1-by-1-by-1024.\n",
            path0.c_str(), buffer0.getWidth(), buffer0.getHeight(), buffer0.getFeatures());
      FatalIf(
            buffer1.getWidth() != 1 or buffer1.getHeight() != 1 or buffer1.getFeatures() != 1024,
            "%s has dimensions %d-by-%d-by-%d instead of expected 1-by-1-by-1024.\n",
            path1.c_str(), buffer1.getWidth(), buffer1.getHeight(), buffer1.getFeatures());
      for (int n = 0; n < 1024; ++n) {
         float v0 = buffer0.at(n);
         float v1 = buffer1.at(n);
         float err = v1 - v0;
         if (std::fabs(v1 - v0) > 1.0e-6f) {
            ErrorLog().printf(
                  "Batch element %d, LCA neuron %4d: expected %f, observed %f, discrepancy %g\n",
                  b, n, double(v0), double(v1), double(err));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}
