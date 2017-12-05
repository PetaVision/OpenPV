/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>
#include <layers/HyPerLayer.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   FatalIf(hc == nullptr, "%s failed to build HyPerCol.\n", argv[0]);
   HyPerLayer *inputLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("Input"));
   FatalIf(inputLayer == nullptr, "%s failed to build input layer.\n", argv[0]);
   HyPerLayer *outputLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("Output"));
   FatalIf(outputLayer == nullptr, "%s failed to build output layer.\n", argv[0]);
   HyPerConn *connection = dynamic_cast<HyPerConn *>(hc->getObjectFromName("InputToOutput"));
   FatalIf(connection == nullptr, "%s failed to build connection.\n", argv[0]);
   return PV_SUCCESS;
}
