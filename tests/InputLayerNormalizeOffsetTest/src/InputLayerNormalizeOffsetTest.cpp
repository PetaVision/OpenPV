/*
 * InputRegionLayerTest.cpp
 *
 */

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <layers/InputLayer.hpp>
#include <layers/InputRegionLayer.hpp>
#include <structures/Buffer.hpp>
#include <utils/BufferUtilsMPI.hpp>

template <typename T>
T *getObjectFromName(std::string const &objectName, PV::HyPerCol *hc);

template <typename T>
char const *objectType();

void verifyLayerLocs(PV::HyPerLayer *layer1, PV::HyPerLayer *layer2);
PV::Buffer<float> gatherLayer(PV::HyPerLayer *layer, PV::Communicator *communicator);
void verifyActivity(
      PV::Buffer<float> *inputBuffer,
      std::string const &inputName,
      PV::Buffer<float> *validRegionBuffer,
      std::string const &validRegionName,
      PV::PVParams *params);

int main(int argc, char *argv[]) {
   PV::PV_Init *pv_init =
         new PV::PV_Init(&argc, &argv, false /* do not allow unrecognized arguments */);

   PV::HyPerCol *hc = new PV::HyPerCol(pv_init);
   hc->allocateColumn();

   auto *inputLayer       = getObjectFromName<PV::InputLayer>(std::string("Input"), hc);
   auto *validRegionLayer = getObjectFromName<PV::InputLayer>(std::string("ValidRegion"), hc);

   verifyLayerLocs(inputLayer, validRegionLayer);

   // Gather everything to rank-zero process; nonzero-rank processes are then done.
   PV::Communicator *communicator = hc->getCommunicator();
   auto inputBuffer               = gatherLayer(inputLayer, communicator);
   auto validRegionBuffer         = gatherLayer(validRegionLayer, communicator);

   if (communicator->commRank() == 0) {
      verifyActivity(
            &inputBuffer,
            std::string(inputLayer->getName()),
            &validRegionBuffer,
            std::string(validRegionLayer->getName()),
            hc->parameters());
   }

   delete hc;
   InfoLog() << "Test passed." << std::endl;
   delete pv_init;
   return EXIT_SUCCESS;
}

template <typename T>
T *getObjectFromName(std::string const &objectName, PV::HyPerCol *hc) {
   std::string const &paramsFile =
         hc->getPV_InitObj()->getStringArgument(std::string("ParamsFile"));
   auto *baseObject = hc->getObjectFromName(objectName);
   FatalIf(
         baseObject == nullptr,
         "No group named \"%s\" in %s",
         objectName.c_str(),
         paramsFile.c_str());
   auto *object = dynamic_cast<T *>(baseObject);
   FatalIf(
         object == nullptr,
         "No %s named \"%s\" in %s",
         objectType<T>(),
         objectName.c_str(),
         paramsFile.c_str());
   return object;
}

template <>
char const *objectType<PV::InputLayer>() {
   return "InputLayer";
}

void verifyLayerLocs(PV::HyPerLayer *layer1, PV::HyPerLayer *layer2) {
   PVLayerLoc const *loc1 = layer1->getLayerLoc();
   PVLayerLoc const *loc2 = layer2->getLayerLoc();
   FatalIf(
         loc1->nbatchGlobal != loc2->nbatchGlobal,
         "%s and %s nbatchGlobal values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->nbatchGlobal,
         loc2->nbatchGlobal);
   FatalIf(
         loc1->nxGlobal != loc2->nxGlobal,
         "%s and %s nxGlobal values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->nxGlobal,
         loc2->nxGlobal);
   FatalIf(
         loc1->nyGlobal != loc2->nyGlobal,
         "%s and %s nyGlobal values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->nyGlobal,
         loc2->nyGlobal);
   FatalIf(
         loc1->nf != loc2->nf,
         "%s and %s nf values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->nf,
         loc2->nf);
   FatalIf(
         loc1->halo.lt != loc2->halo.lt,
         "%s and %s halo.lt values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->halo.lt,
         loc2->halo.lt);
   FatalIf(
         loc1->halo.rt != loc2->halo.rt,
         "%s and %s halo.rt values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->halo.rt,
         loc2->halo.rt);
   FatalIf(
         loc1->halo.dn != loc2->halo.dn,
         "%s and %s halo.dn values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->halo.dn,
         loc2->halo.dn);
   FatalIf(
         loc1->halo.up != loc2->halo.up,
         "%s and %s halo.up values differ (%d versus %d).\n",
         layer1->getName(),
         layer2->getName(),
         loc1->halo.up,
         loc2->halo.up);
}

PV::Buffer<float> gatherLayer(PV::HyPerLayer *layer, PV::Communicator *communicator) {
   PVLayerLoc const *loc = layer->getLayerLoc();
   int nxExt             = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt             = loc->ny + loc->halo.dn + loc->halo.up;
   int const rootProc    = 0;
   PV::Buffer<float> buffer(layer->getLayerData(0), nxExt, nyExt, loc->nf);
   buffer = PV::BufferUtils::gather(
         communicator->getLocalMPIBlock(), buffer, loc->nx, loc->ny, 0 /*batch index*/, rootProc);
   return buffer;
}

void verifyActivity(
      PV::Buffer<float> *inputBuffer,
      std::string const &inputName,
      PV::Buffer<float> *validRegionBuffer,
      std::string const &validRegionName,
      PV::PVParams *params) {
   // Each neuron of ValidRegion should have either activity = 1 or activity = 0.
   FatalIf(
         inputBuffer->getTotalElements() != validRegionBuffer->getTotalElements(),
         "%s and %s have different total numbers of elements (%d versus %d)\n",
         inputName.c_str(),
         validRegionName.c_str(),
         inputBuffer->getTotalElements(),
         validRegionBuffer->getTotalElements());
   int const totalElements = inputBuffer->getTotalElements();

   // Verify each validRegionBuffer value is either 0.0 or 1.0, and that each appears somewhere.
   bool onePresent  = false;
   bool zeroPresent = false;
   for (int k = 0; k < totalElements; k++) {
      float const value = validRegionBuffer->at(k);
      FatalIf(
            value != 0.0f and value != 1.0f,
            "%s has activity that is not all ones or zeroes.\n",
            validRegionName.c_str());
      if (value == 0.0f) {
         zeroPresent = true;
      }
      if (value == 1.0f) {
         onePresent = true;
      }
   }
   FatalIf(!onePresent, "%s activity is all zeroes.\n", validRegionName.c_str());
   FatalIf(!zeroPresent, "%s activity is all ones.\n", validRegionName.c_str());

   int status = PV_SUCCESS;

   // Verify that where ValidRegion is zero, Input is the pad value.
   float padValue = (float)params->value(
         inputName.c_str(), "padValue", 0.0f /*default*/, false /*no warning if absent*/);
   for (int k = 0; k < totalElements; k++) {
      if (validRegionBuffer->at(k) == 0.0f and inputBuffer->at(k) != padValue) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "Neuron %d: %s has value zero but %s has value %f instead of the pad value %f.\n",
               k,
               validRegionName.c_str(),
               inputName.c_str(),
               (double)inputBuffer->at(k),
               (double)padValue);
      }
   }
   FatalIf(status != PV_SUCCESS, "Test failed.\n");

   // Verify that where ValidRegion is one, Input has been normalized.
   bool normalizeLuminance = params->value(
                                   inputName.c_str(),
                                   "normalizeLuminanceFlag",
                                   false /*default*/,
                                   false /*no warning if absent*/)
                             != 0;
   FatalIf(
         !normalizeLuminance,
         "%s has normalizeLuminanceFlag set to false. This test requires it to be true.\n",
         inputName.c_str());
   double mean = 0.0;
   int count   = 0;
   for (int k = 0; k < totalElements; k++) {
      if (validRegionBuffer->at(k) == 0.0f) {
         continue;
      }
      mean += (double)inputBuffer->at(k);
      count++;
   }
   FatalIf(count <= 0, "Computing mean: count is not positive.\n");
   mean /= (double)count;
   FatalIf(
         std::abs(mean) > 1.0e-6,
         "Mean of %s over valid region is %f instead of zero.\n",
         inputName.c_str(),
         mean);

   bool normalizeStdDev = params->value(
                                inputName.c_str(),
                                "normalizeStdDev",
                                false /*default*/,
                                false /*no warning if absent*/)
                          != 0;
   FatalIf(
         !normalizeLuminance,
         "%s has normalizeStdDev set to false. This test requires it to be true.\n",
         inputName.c_str());
   double stddev = 0.0;
   count         = 0;
   for (int k = 0; k < totalElements; k++) {
      if (validRegionBuffer->at(k) == 0.0f) {
         continue;
      }
      double dev = (double)inputBuffer->at(k) - mean;
      stddev += dev * dev;
      count++;
   }
   FatalIf(count <= 0, "Computing standard deviation: count is not positive.\n");
   stddev /= (double)count;
   stddev = std::sqrt(stddev);
   FatalIf(
         std::abs(stddev - 1.0) > 1.0e-6,
         "Mean of %s over valid region is %f instead of one.\n",
         inputName.c_str(),
         mean);
}
