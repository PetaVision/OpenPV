/*
 * pv.cpp
 *
 */

#include <columns/HyPerCol.hpp>
#include <columns/Messages.hpp>
#include <columns/PV_Init.hpp>
#include <components/LayerGeometry.hpp>

void initGeometries(
      PV::HyPerCol *hc,
      PV::LayerGeometry **haloA,
      PV::LayerGeometry **haloB,
      PV::LayerGeometry **haloC);
void checkHalo(PVHalo const *halo, int xmargin, int ymargin, char const *name);

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
   PV::HyPerCol *hc         = nullptr;
   PV::LayerGeometry *geomA = nullptr, *geomB = nullptr, *geomC = nullptr;
   PVHalo const *haloA = nullptr, *haloB = nullptr, *haloC = nullptr;

   // ============= // Create layers, and change x-margins

   hc = new PV::HyPerCol(&pv_initObj);
   initGeometries(hc, &geomA, &geomB, &geomC);
   haloA = &geomA->getLayerLoc()->halo;
   haloB = &geomB->getLayerLoc()->halo;
   haloC = &geomC->getLayerLoc()->halo;

   checkHalo(haloA, 0, 0, "A");
   checkHalo(haloB, 0, 0, "B");
   checkHalo(haloC, 0, 0, "C");

   geomA->requireMarginWidth(2, 'x');
   checkHalo(haloA, 2, 0, "A");

   PV::LayerGeometry::synchronizeMarginWidths(geomA, geomB);
   checkHalo(haloA, 2, 0, "A");
   checkHalo(haloA, 2, 0, "B");

   geomA->requireMarginWidth(3, 'x');
   checkHalo(haloA, 3, 0, "A");
   checkHalo(haloA, 3, 0, "B");

   geomB->requireMarginWidth(4, 'x');
   checkHalo(haloA, 4, 0, "A");
   checkHalo(haloA, 4, 0, "B");

   geomB->requireMarginWidth(1, 'x');
   checkHalo(haloA, 4, 0, "A");
   checkHalo(haloA, 4, 0, "B");

   geomC->requireMarginWidth(5, 'x');
   checkHalo(haloA, 4, 0, "A");
   checkHalo(haloB, 4, 0, "B");
   checkHalo(haloC, 5, 0, "C");

   PV::LayerGeometry::synchronizeMarginWidths(geomA, geomC);
   checkHalo(haloA, 5, 0, "A");
   checkHalo(haloB, 5, 0, "B");
   checkHalo(haloC, 5, 0, "C");

   geomB->requireMarginWidth(6, 'x');
   checkHalo(haloA, 6, 0, "A");
   checkHalo(haloB, 6, 0, "B");
   checkHalo(haloC, 6, 0, "C");

   // At the end, check y-margins; they should all be synchronized.

   geomA->requireMarginWidth(2, 'y');
   checkHalo(haloA, 6, 2, "A");
   checkHalo(haloA, 6, 2, "B");
   checkHalo(haloA, 6, 2, "C");

   delete hc;

   // ============= // Reset, and do y-margin first, then x-margin at the end

   hc = new PV::HyPerCol(&pv_initObj);
   initGeometries(hc, &geomA, &geomB, &geomC);
   haloA = &geomA->getLayerLoc()->halo;
   haloB = &geomB->getLayerLoc()->halo;
   haloC = &geomC->getLayerLoc()->halo;

   checkHalo(haloA, 0, 0, "A");
   checkHalo(haloB, 0, 0, "B");
   checkHalo(haloC, 0, 0, "C");

   geomA->requireMarginWidth(2, 'y');
   checkHalo(haloA, 0, 2, "A");

   PV::LayerGeometry::synchronizeMarginWidths(geomA, geomB);
   checkHalo(haloA, 0, 2, "A");
   checkHalo(haloA, 0, 2, "B");

   geomA->requireMarginWidth(3, 'y');
   checkHalo(haloA, 0, 3, "A");
   checkHalo(haloA, 0, 3, "B");

   geomB->requireMarginWidth(4, 'y');
   checkHalo(haloA, 0, 4, "A");
   checkHalo(haloA, 0, 4, "B");

   geomB->requireMarginWidth(1, 'y');
   checkHalo(haloA, 0, 4, "A");
   checkHalo(haloA, 0, 4, "B");

   geomC->requireMarginWidth(5, 'y');
   checkHalo(haloA, 0, 4, "A");
   checkHalo(haloB, 0, 4, "B");
   checkHalo(haloC, 0, 5, "C");

   PV::LayerGeometry::synchronizeMarginWidths(geomA, geomC);
   checkHalo(haloA, 0, 5, "A");
   checkHalo(haloB, 0, 5, "B");
   checkHalo(haloC, 0, 5, "C");

   geomB->requireMarginWidth(6, 'y');
   checkHalo(haloA, 0, 6, "A");
   checkHalo(haloB, 0, 6, "B");
   checkHalo(haloC, 0, 6, "C");

   geomA->requireMarginWidth(2, 'x');
   checkHalo(haloA, 2, 6, "A");
   checkHalo(haloA, 2, 6, "B");
   checkHalo(haloA, 2, 6, "C");

   delete hc;

   return EXIT_SUCCESS;
}

void initGeometries(
      PV::HyPerCol *hc,
      PV::LayerGeometry **geomA,
      PV::LayerGeometry **geomB,
      PV::LayerGeometry **geomC) {
   auto objectTable           = hc->getAllObjectsFlat();
   auto communicateMessagePtr = std::make_shared<PV::CommunicateInitInfoMessage>(
         &objectTable,
         hc->getDeltaTime(),
         hc->getNxGlobal(),
         hc->getNyGlobal(),
         hc->getNBatchGlobal(),
         hc->getNumThreads());

   *geomA = nullptr;
   *geomB = nullptr;
   *geomC = nullptr;

   *geomA = new PV::LayerGeometry("A", hc->parameters(), hc->getCommunicator());
   (*geomA)->respond(communicateMessagePtr);

   *geomB = new PV::LayerGeometry("B", hc->parameters(), hc->getCommunicator());
   (*geomB)->respond(communicateMessagePtr);

   *geomC = new PV::LayerGeometry("C", hc->parameters(), hc->getCommunicator());
   (*geomC)->respond(communicateMessagePtr);
}

void checkHalo(PVHalo const *halo, int xmargin, int ymargin, char const *name) {
   int status = PV_SUCCESS;
   if (halo->lt != xmargin) {
      ErrorLog().printf(
            "%s expected left margin %d, observed value %d.\n", name, xmargin, halo->lt);
      status = PV_FAILURE;
   }
   if (halo->rt != xmargin) {
      ErrorLog().printf(
            "%s expected right margin %d, observed value %d.\n", name, xmargin, halo->rt);
      status = PV_FAILURE;
   }
   if (halo->dn != ymargin) {
      ErrorLog().printf(
            "%s expected bottom margin %d, observed value %d.\n", name, ymargin, halo->dn);
      status = PV_FAILURE;
   }
   if (halo->up != ymargin) {
      ErrorLog().printf("%s expected top margin %d, observed value %d.\n", name, ymargin, halo->up);
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "SynchronizeMarginWidth test failed.\n");
}
