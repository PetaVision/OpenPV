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
void dumpLayerActivity(PV::HyPerLayer *layer, PV::Communicator *communicator);

int main(int argc, char *argv[]) {
   PV::PV_Init *pv_init =
         new PV::PV_Init(&argc, &argv, false /* do not allow unrecognized arguments */);
   PV::HyPerCol *hc              = new PV::HyPerCol(pv_init);
   std::string const &paramsFile = pv_init->getStringArgument(std::string("ParamsFile"));
   int status                    = PV_SUCCESS;
   hc->allocateColumn();

   auto *inputLayer  = getObjectFromName<PV::InputLayer>(std::string("Input"), hc);
   auto *regionLayer = getObjectFromName<PV::InputRegionLayer>(std::string("InputRegion"), hc);

   verifyLayerLocs(inputLayer, regionLayer);

   dumpLayerActivity(inputLayer, pv_init->getCommunicator());
   dumpLayerActivity(regionLayer, pv_init->getCommunicator());

   MPI_Barrier(pv_init->getCommunicator()->globalCommunicator());
   delete hc;
   delete pv_init;
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

void dumpLayerActivity(PV::HyPerLayer *layer, PV::Communicator *communicator) {
   PVLayerLoc const *loc = layer->getLayerLoc();
   int nxExt             = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt             = loc->ny + loc->halo.dn + loc->halo.up;
   PV::Buffer<float> buffer(layer->getLayerData(0), nxExt, nyExt, loc->nf);
   int const rootProc = 0;
   buffer             = PV::BufferUtils::gather(
         communicator->getLocalMPIBlock(), buffer, loc->nx, loc->ny, 0 /*batch index*/, rootProc);
   if (communicator->commRank() == rootProc) {
      int nxExtGlobal = loc->nxGlobal + loc->halo.lt + loc->halo.rt;
      int nyExtGlobal = loc->nyGlobal + loc->halo.dn + loc->halo.up;
      InfoLog() << layer->getDescription() << "\n";
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
               InfoLog().printf(" %3d ", (int)buffer.at(x, y, f));
            }
            InfoLog() << "\n";
         }
      }
   }
}
