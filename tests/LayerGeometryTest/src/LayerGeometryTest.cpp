/*
 * pv.cpp
 *
 */

#include <columns/HyPerCol.hpp>
#include <columns/Messages.hpp>
#include <columns/PV_Init.hpp>
#include <components/LayerGeometry.hpp>
#include <layers/HyPerLayer.hpp>

PVLayerLoc makeCorrectLoc(PV::HyPerCol *hc);
void communicateInitInfo(PV::HyPerCol *hc);
int checkLayerLoc(PVLayerLoc const *loc, PVLayerLoc const *correct);

// This program reads the params file LayerGeometryTest.params, which contains the HyPerCol and a
// single HyPerLayer, "Layer". It then checks the LayerGeometry class in two ways:
// (1) by directly constructing a LayerGeometry object named "Layer".
// (2) by retrieving the LayerGeometry component from the HyPerLayer.
// In both cases, it retrieves the PVLayerLoc and compares it to the correct PVLayerLoc, built
// from the MPI configuration and the params.
//
// Note that LayerGeometry does not construct the PVLayerLoc until the CommunicateInitInfo stage.
// The LayerGeometry class's communicateInitInfo method does not have any dependencies,
// so for the directly-constructed LayerGeometry object, we only need to send the object
// a CommunicateInitInfo message.
//
// When using the LayerGeometry inside the HyPerLayer, we send a CommunicateInitInfo
// message to the HyPerCol's hierarchy. If working correctly, HyPerLayer will receive
// this message and send it along to its components, including LayerGeometry.
//

int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   auto *hc = new PV::HyPerCol(&pv_initObj);

   PVLayerLoc correctLoc = makeCorrectLoc(hc);

   PV::LayerGeometry *lg = nullptr;
   PVLayerLoc const *loc = nullptr;
   int status            = PV_SUCCESS;

   // Test direct construction of the LayerGeometry component.
   lg = new PV::LayerGeometry("Layer", hc->parameters(), hc->getCommunicator());

   auto objectTable           = hc->getAllObjectsFlat();
   auto communicateMessagePtr = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hc->getDeltaTime(),
         hc->getNxGlobal(),
         hc->getNyGlobal(),
         hc->getNBatchGlobal(),
         hc->getNumThreads());
   lg->respond(communicateMessagePtr);

   loc    = lg->getLayerLoc();
   status = checkLayerLoc(loc, &correctLoc);
   FatalIf(status != PV_SUCCESS, "Stand-alone construction of LayerGeometry failed.\n");
   delete lg;

   // Test whether HyPerLayer builds the LayerGeometry component correctly.
   auto *layer = dynamic_cast<PV::HyPerLayer *>(hc->getObjectFromName(std::string("Layer")));
   FatalIf(layer == nullptr, "Params file does not contain a layer named \"Layer\"\n");
   communicateInitInfo(hc);
   lg = layer->getComponentByType<PV::LayerGeometry>();
   FatalIf(lg == nullptr, "Layer does not contain a LayerGeometry component.\n");
   loc = lg->getLayerLoc();

   status = checkLayerLoc(loc, &correctLoc);
   FatalIf(status != PV_SUCCESS, "Construction of LayerGeometry within HyPerLayer failed.\n");
   delete hc;

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

PVLayerLoc makeCorrectLoc(PV::HyPerCol *hc) {
   // Read parameters directly
   auto *params = hc->parameters();
   int nx       = params->value(hc->getName(), "nx");
   int ny       = params->value(hc->getName(), "ny");
   int nbatch   = params->value(hc->getName(), "nbatch");

   float nxScale   = params->value("Layer", "nxScale");
   float nyScale   = params->value("Layer", "nyScale");
   int numFeatures = params->value("Layer", "nf");

   // Get location in MPI configuration
   auto comm          = hc->getCommunicator();
   int numRows        = comm->numCommRows();
   int numColumns     = comm->numCommColumns();
   int commBatchWidth = comm->numCommBatches();
   int rowIndex       = comm->commRow();
   int columnIndex    = comm->commColumn();
   int batchIndex     = comm->commBatch();

   // Use the params to build the correct loc
   PVLayerLoc correctLoc;
   correctLoc.nbatchGlobal = nbatch;
   correctLoc.nxGlobal     = (int)std::nearbyint((float)nx * nxScale);
   correctLoc.nyGlobal     = (int)std::nearbyint((float)ny * nyScale);
   correctLoc.nbatch       = correctLoc.nbatchGlobal / commBatchWidth;
   correctLoc.nx           = correctLoc.nxGlobal / numColumns;
   correctLoc.ny           = correctLoc.nyGlobal / numRows;
   correctLoc.nf           = numFeatures;
   correctLoc.kb0          = correctLoc.nbatch * batchIndex;
   correctLoc.kx0          = correctLoc.nx * columnIndex;
   correctLoc.ky0          = correctLoc.ny * rowIndex;
   correctLoc.halo.lt      = 0;
   correctLoc.halo.rt      = 0;
   correctLoc.halo.dn      = 0;
   correctLoc.halo.up      = 0;

   return correctLoc;
}

void communicateInitInfo(PV::HyPerCol *hc) {
   auto objectTable = hc->getAllObjectsFlat();
   auto messagePtr  = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hc->getDeltaTime(),
         hc->getNxGlobal(),
         hc->getNyGlobal(),
         hc->getNBatchGlobal(),
         hc->getNumThreads());

   int maxcount = 0;
   PV::Response::Status status;
   do {
      status = PV::Response::SUCCESS;
      for (auto &obj : objectTable) {
         status = status + obj->respond(messagePtr);
      }
      maxcount++;
   } while (status != PV::Response::SUCCESS and maxcount < 10);
   FatalIf(
         status != PV::Response::SUCCESS,
         "communicateInitInfo(\"%s\") failed.\n",
         messagePtr->getMessageType().c_str());
}

int checkLayerLoc(PVLayerLoc const *loc, PVLayerLoc const *correct) {
   int status = PV_SUCCESS;
   if (loc->nbatchGlobal != correct->nbatchGlobal) {
      ErrorLog().printf(
            "loc->nbatchGlobal should be %d, but is %d\n",
            correct->nbatchGlobal,
            loc->nbatchGlobal);
      status = PV_FAILURE;
   }
   if (loc->nxGlobal != correct->nxGlobal) {
      ErrorLog().printf(
            "loc->nxGlobal should be %d, but is %d\n", correct->nxGlobal, loc->nxGlobal);
      status = PV_FAILURE;
   }
   if (loc->nyGlobal != correct->nyGlobal) {
      ErrorLog().printf(
            "loc->nyGlobal should be %d, but is %d\n", correct->nyGlobal, loc->nyGlobal);
      status = PV_FAILURE;
   }
   if (loc->nbatch != correct->nbatch) {
      ErrorLog().printf("loc->nbatch should be %d, but is %d\n", correct->nbatch, loc->nbatch);
      status = PV_FAILURE;
   }
   if (loc->nx != correct->nx) {
      ErrorLog().printf("loc->nx should be %d, but is %d\n", correct->nx, loc->nx);
      status = PV_FAILURE;
   }
   if (loc->ny != correct->ny) {
      ErrorLog().printf("loc->ny should be %d, but is %d\n", correct->ny, loc->ny);
      status = PV_FAILURE;
   }
   if (loc->nf != correct->nf) {
      ErrorLog().printf("loc->nf should be %d, but is %d\n", correct->nf, loc->nf);
      status = PV_FAILURE;
   }
   if (loc->kb0 != correct->kb0) {
      ErrorLog().printf("loc->kb0 should be %d, but is %d\n", correct->kb0, loc->kb0);
      status = PV_FAILURE;
   }
   if (loc->kx0 != correct->kx0) {
      ErrorLog().printf("loc->kx0 should be %d, but is %d\n", correct->kx0, loc->kx0);
      status = PV_FAILURE;
   }
   if (loc->ky0 != correct->ky0) {
      ErrorLog().printf("loc->ky0 should be %d, but is %d\n", correct->ky0, loc->ky0);
      status = PV_FAILURE;
   }
   if (loc->halo.lt != correct->halo.lt) {
      ErrorLog().printf("loc->halo.lt should be %d, but is %d\n", correct->halo.lt, loc->halo.lt);
      status = PV_FAILURE;
   }
   if (loc->halo.rt != correct->halo.rt) {
      ErrorLog().printf("loc->halo.rt should be %d, but is %d\n", correct->halo.rt, loc->halo.rt);
      status = PV_FAILURE;
   }
   if (loc->halo.dn != correct->halo.dn) {
      ErrorLog().printf("loc->halo.dn should be %d, but is %d\n", correct->halo.dn, loc->halo.dn);
      status = PV_FAILURE;
   }
   if (loc->halo.up != correct->halo.up) {
      ErrorLog().printf("loc->halo.up should be %d, but is %d\n", correct->halo.up, loc->halo.up);
      status = PV_FAILURE;
   }
   return status;
}
