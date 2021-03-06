/*
 * InputRegionLayerTest.cpp
 *
 */

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <components/BasePublisherComponent.hpp>
#include <layers/InputLayer.hpp>
#include <layers/InputRegionLayer.hpp>
#include <structures/Buffer.hpp>
#include <utils/BufferUtilsMPI.hpp>

template <typename T>
T *getObjectFromName(std::string const &objectName, PV::HyPerCol *hc);

template <typename T>
char const *objectType();

void verifyLayerLocs(
      PV::BasePublisherComponent *publisher1,
      PV::BasePublisherComponent *publisher2);
void compareLayers(
      PV::BasePublisherComponent *publisher1,
      PV::BasePublisherComponent *publisher2,
      PV::Communicator const *communicator);
PV::Buffer<float>
gatherLayer(PV::BasePublisherComponent *publisher, PV::Communicator const *communicator);
void dumpLayerActivity(
      PV::Buffer<float> &layerBuffer,
      PVLayerLoc const *loc,
      std::string const &description);

int main(int argc, char *argv[]) {
   PV::PV_Init *pv_init =
         new PV::PV_Init(&argc, &argv, false /* do not allow unrecognized arguments */);
   PV::HyPerCol *hc = new PV::HyPerCol(pv_init);
   int status       = PV_SUCCESS;
   hc->run();

   auto *inputLayer      = getObjectFromName<PV::InputLayer>(std::string("Input"), hc);
   auto *inputPublisher  = inputLayer->getComponentByType<PV::BasePublisherComponent>();
   auto *regionLayer     = getObjectFromName<PV::InputRegionLayer>(std::string("InputRegion"), hc);
   auto *regionPublisher = regionLayer->getComponentByType<PV::BasePublisherComponent>();
   auto *correctRegion   = getObjectFromName<PV::HyPerLayer>(std::string("CorrectInputRegion"), hc);
   auto *correctPublisher = correctRegion->getComponentByType<PV::BasePublisherComponent>();

   verifyLayerLocs(inputPublisher, regionPublisher);
   verifyLayerLocs(regionPublisher, correctPublisher);

   compareLayers(regionPublisher, correctPublisher, hc->getCommunicator());
   delete hc;
   delete pv_init;
   InfoLog() << "Test passed." << std::endl;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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
template <>
char const *objectType<PV::InputRegionLayer>() {
   return "InputRegionLayer";
}
template <>
char const *objectType<PV::HyPerLayer>() {
   return "HyPerLayer";
}

void verifyLayerLocs(
      PV::BasePublisherComponent *publisher1,
      PV::BasePublisherComponent *publisher2) {
   PVLayerLoc const *loc1 = publisher1->getLayerLoc();
   PVLayerLoc const *loc2 = publisher2->getLayerLoc();
   FatalIf(
         loc1->nbatchGlobal != loc2->nbatchGlobal,
         "%s and %s nbatchGlobal values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->nbatchGlobal,
         loc2->nbatchGlobal);
   FatalIf(
         loc1->nxGlobal != loc2->nxGlobal,
         "%s and %s nxGlobal values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->nxGlobal,
         loc2->nxGlobal);
   FatalIf(
         loc1->nyGlobal != loc2->nyGlobal,
         "%s and %s nyGlobal values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->nyGlobal,
         loc2->nyGlobal);
   FatalIf(
         loc1->nf != loc2->nf,
         "%s and %s nf values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->nf,
         loc2->nf);
   FatalIf(
         loc1->halo.lt != loc2->halo.lt,
         "%s and %s halo.lt values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->halo.lt,
         loc2->halo.lt);
   FatalIf(
         loc1->halo.rt != loc2->halo.rt,
         "%s and %s halo.rt values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->halo.rt,
         loc2->halo.rt);
   FatalIf(
         loc1->halo.dn != loc2->halo.dn,
         "%s and %s halo.dn values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->halo.dn,
         loc2->halo.dn);
   FatalIf(
         loc1->halo.up != loc2->halo.up,
         "%s and %s halo.up values differ (%d versus %d).\n",
         publisher1->getName(),
         publisher2->getName(),
         loc1->halo.up,
         loc2->halo.up);
}

void compareLayers(
      PV::BasePublisherComponent *publisher1,
      PV::BasePublisherComponent *publisher2,
      PV::Communicator const *communicator) {
   verifyLayerLocs(publisher1, publisher2);
   PV::Buffer<float> layer1buffer = gatherLayer(publisher1, communicator);
   PV::Buffer<float> layer2buffer = gatherLayer(publisher2, communicator);
   if (communicator->globalCommRank() == 0) {
      FatalIf(
            layer1buffer.getTotalElements() != layer2buffer.getTotalElements(),
            "Buffers from %s and %s do not have the same total number of elements.\n",
            publisher1->getDescription_c(),
            publisher2->getDescription_c());
      int const N = layer1buffer.getTotalElements();

      int status = PV_FAILURE;
      for (int n = 0; n < N; n++) {
         if (layer1buffer.at(n) != 0.0f) {
            status = PV_SUCCESS;
            break;
         }
      }
      if (status != PV_SUCCESS) {
         Fatal().printf(
               "Layer %s does not have any nonzero activity.\n", publisher1->getDescription_c());
      }

      pvAssert(status == PV_SUCCESS);
      for (int n = 0; n < N; n++) {
         if (layer1buffer.at(n) != layer2buffer.at(n)) {
            status = PV_FAILURE;
            break;
         }
      }
      if (status != PV_SUCCESS) {
         InfoLog().printf(
               "%s and %s do not agree.\n",
               publisher1->getDescription_c(),
               publisher2->getDescription_c());
         dumpLayerActivity(layer1buffer, publisher1->getLayerLoc(), publisher1->getDescription());
         dumpLayerActivity(layer2buffer, publisher2->getLayerLoc(), publisher2->getDescription());
         Fatal().printf(
               "Layers %s and %s do not agree.\n",
               publisher1->getDescription_c(),
               publisher2->getDescription_c());
      }
   }
}

PV::Buffer<float>
gatherLayer(PV::BasePublisherComponent *publisher, PV::Communicator const *communicator) {
   PVLayerLoc const *loc = publisher->getLayerLoc();
   int nxExt             = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt             = loc->ny + loc->halo.dn + loc->halo.up;
   int const rootProc    = 0;
   PV::Buffer<float> buffer(publisher->getLayerData(0), nxExt, nyExt, loc->nf);
   buffer = PV::BufferUtils::gather(
         communicator->getLocalMPIBlock(), buffer, loc->nx, loc->ny, 0 /*batch index*/, rootProc);
   return buffer;
}

void dumpLayerActivity(
      PV::Buffer<float> &layerBuffer,
      PVLayerLoc const *loc,
      std::string const &description) {
   int nxExtGlobal = loc->nxGlobal + loc->halo.lt + loc->halo.rt;
   int nyExtGlobal = loc->nyGlobal + loc->halo.dn + loc->halo.up;
   int nf          = loc->nf;
   FatalIf(
         layerBuffer.getTotalElements() != nxExtGlobal * nyExtGlobal * nf,
         "%s has the wrong number of elements.\n",
         description.c_str());
   InfoLog() << description << ":\n";
   for (int f = 0; f < loc->nf; f++) {
      InfoLog() << "    Feature index " << f << ":\n";
      for (int y = 0; y < nyExtGlobal; y++) {
         if (y == loc->halo.up or y == loc->nyGlobal + loc->halo.up) {
            InfoLog() << std::string(12, ' ') << std::string(loc->halo.lt * 5, '-') << '+'
                      << std::string(loc->nxGlobal * 5, '-') << '+'
                      << std::string(loc->halo.rt * 5, '-') << "\n";
         }
         InfoLog().printf("    y = %3d:", y - loc->halo.up);
         for (int x = 0; x < nxExtGlobal; x++) {
            if (x == loc->halo.lt or x == loc->nxGlobal + loc->halo.lt) {
               printf("|");
            }
            InfoLog().printf(" %3.1f ", (double)layerBuffer.at(x, y, f));
         }
         InfoLog() << "\n";
      }
   }
}
