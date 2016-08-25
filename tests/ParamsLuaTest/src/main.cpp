/*
 * main.cpp for ParamsLuaTest
 *
 *  Created on: Jul 1, 2016
 *      Author: peteschultz
 */


#include "ColumnArchive.hpp"
#include <cmath>
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <columns/HyPerCol.hpp>
#include <layers/HyPerLayer.hpp>
#include <connections/HyPerConn.hpp>

bool compareColumns(PV::HyPerCol * hc1, PV::HyPerCol * hc2, pvdata_t layerTolerance, pvwdata_t weightTolerance);
bool compareLayers(PV::HyPerLayer * layer1, PV::HyPerLayer * layer2, pvwdata_t tolerance);
bool compareConns(PV::HyPerConn * conn1, PV::HyPerConn * conn2, pvwdata_t tolerance);

int main(int argc, char * argv[]) {
   float tolerance = 1.0e-5;

   PV::PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   int rank = pv_initObj.getWorldRank();
   int status = PV_SUCCESS;
   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the params file argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt_str(argc, argv, "-c", NULL, NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (pv_getopt(argc, argv, "-r", NULL)==0) {
      if (rank==0) {
         pvErrorNoExit().printf("%s should be run without the restart flag.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank==0) {
         pvErrorNoExit().printf("This test uses compares a hard-coded .params.lua file with a hard-coded .params file, and the results of the two runs are compared.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   std::string paramsfile("input/ParamsLuaTest.params");
   std::string paramsluafile("output/pv.params.lua");
   pv_initObj.setParams(paramsfile.c_str());
   PV::HyPerCol * hc1 = createHyPerCol(&pv_initObj);
   if (hc1==nullptr) { pvError() << "setParams(\"" << paramsfile << "\") failed.\n"; }
   status = hc1->run();
   if (status != PV_SUCCESS) { pvError() << "Running with \"" << paramsfile << "\" failed.\n"; }
   ColumnArchive columnArchive1(hc1, tolerance, tolerance); // Archive the layer and connection data since changing the params file is destructive.

   pv_initObj.setParams(paramsluafile.c_str());
   PV::HyPerCol * hc2 = createHyPerCol(&pv_initObj);
   if (hc2==nullptr) { pvError() << "setParams(\"" << paramsluafile << "\") failed.\n"; }
   status = hc2->run();
   if (status != PV_SUCCESS) { pvError() << "Running with \"" << paramsluafile << "\" failed.\n"; }
   ColumnArchive columnArchive2(hc2, tolerance, tolerance);

   return columnArchive1==columnArchive2 ? EXIT_SUCCESS : EXIT_FAILURE;
}

bool compareColumns(PV::HyPerCol * hc1, PV::HyPerCol * hc2, pvdata_t layerTolerance, pvwdata_t weightTolerance) {
   int numLayers = hc1->numberOfLayers();
   if (numLayers != hc2->numberOfLayers()) {
      pvErrorNoExit() << "hc1 and hc2 have different numbers of layers\n";
      return false;
   }
   int numConnections = hc1->numberOfConnections();
   if (numConnections != hc2->numberOfConnections()) {
      pvErrorNoExit() << "hc1 and hc2 have different numbers of connections\n";
   }
   bool areEqual = true;

   for (int layerindex = 0; layerindex < numLayers; layerindex++) {
      PV::HyPerLayer * l1 = hc1->getLayer(layerindex);
      PV::HyPerLayer * l2 = hc2->getLayerFromName(l1->getName());
      if (!compareLayers(l1, l2, layerTolerance)) { areEqual = false; }
   }

   for (int connindex = 0; connindex < numConnections; connindex++) {
      PV::BaseConnection * b1 = hc1->getConnection(connindex);
      if (b1==nullptr) {
         pvErrorNoExit() << "hc1 connection " << connindex << " is null.\n";
         areEqual = false;
      }
      PV::HyPerConn * c1 = dynamic_cast<PV::HyPerConn*>(b1);
      if (c1==nullptr) {
         pvErrorNoExit() << "hc1 connection \"" << b1->getName() << "\" is not a HyPerConn.\n";
         areEqual = false;
      }
      PV::BaseConnection * b2 = hc1->getConnection(connindex);
      if (b2==nullptr) {
         pvErrorNoExit() << "hc2 connection " << connindex << " is null.\n";
         areEqual = false;
      }
      PV::HyPerConn * c2 = dynamic_cast<PV::HyPerConn*>(b2);
      if (c2==nullptr) {
         pvErrorNoExit() << "hc2 connection \"" << b2->getName() << "\" is not a HyPerConn.\n";
         areEqual = false;
      }
      if (!areEqual) {
         return false;
      }
      if (!compareConns(c1, c2, weightTolerance)) { areEqual = false; }
   }
   return areEqual;
}

bool compareLayers(PV::HyPerLayer * layer1, PV::HyPerLayer * layer2, pvwdata_t tolerance) {
   PVLayerLoc const * loc1 = layer1->getLayerLoc();
   PVLayerLoc const * loc2 = layer2->getLayerLoc();
   if (loc1->nx != loc2->nx || loc1->ny != loc2->ny || loc1->nf != loc2->nf) {
      pvErrorNoExit() << "Layers \"" << layer1->getName() << "\" and \"" << layer2->getName() << "\" have different dimensions.\n";
      return false;
   }
   if (layer1->getNumNeurons() != loc1->nx * loc1->ny * loc1->nf) {
      pvError() << "Number of neurons in layer \"" << layer1->getName() << "\" has the wrong number of neurons.\n";
   }
   if (layer2->getNumNeurons() != loc2->nx * loc2->ny * loc2->nf) {
      pvError() << "Number of neurons in layer \"" << layer2->getName() << "\" has the wrong number of neurons.\n";
   }
   int const N = layer1->getNumNeurons(); pvAssert(N==layer2->getNumNeurons());
   int const nx = loc1->nx; pvErrorIf(nx!=loc2->nx, "Test failed.\n");
   int const ny = loc1->ny; pvErrorIf(ny!=loc2->ny, "Test failed.\n");
   int const nf = loc1->nf; pvErrorIf(nf!=loc2->nf, "Test failed.\n");
   PVHalo const * halo1 = &loc1->halo;
   PVHalo const * halo2 = &loc2->halo;

   bool layersEqual = true; // Will become false if any nonequal neurons are encountered.
   for (int k=0; k<N; k++) {
      int kExt1 = kIndexExtended(k, nx, ny, nf, halo1->lt, halo1->rt, halo1->dn, halo1->up);
      int kExt2 = kIndexExtended(k, nx, ny, nf, halo2->lt, halo2->rt, halo2->dn, halo2->up);
      pvdata_t const a1 = layer1->getLayerData()[kExt1];
      pvdata_t const a2 = layer2->getLayerData()[kExt2];
      if (a1 != a2 && std::fabs(a2-a1) > tolerance) {
         pvErrorNoExit() << "Layers \"" << layer1->getName() << "\" and \"" << layer2->getName() <<
               "\" disagree in restricted index " << k << " (" << a1 << " versus " << a2 << ")\n";
         layersEqual = false;
      }
   }
   
   return layersEqual;
}

bool compareConns(PV::HyPerConn * conn1, PV::HyPerConn * conn2, pvwdata_t tolerance) {
   int const numArbors = conn1->numberOfAxonalArborLists();
   if (numArbors != conn2->numberOfAxonalArborLists()) {
      pvErrorNoExit() << "Connections \"" << conn1->getName() << "\" and \"" << conn2->getName() << "\" have different numbers of axonal arbors.\n";
      return false;
   }
   if (conn1->usingSharedWeights() != conn2->usingSharedWeights()) {
      pvErrorNoExit() << "Connections \"" << conn1->getName() << "\" and \"" << conn2->getName() << "\" have different shared weights flags.\n";
      return false;
   }
   int const numPatches = conn1->getNumDataPatches(); // conn1->getNumPatches();
   if (numPatches != conn2->getNumDataPatches()) {
      pvErrorNoExit() << "Connections \"" << conn1->getName() << "\" and \"" << conn2->getName() << "\" have different numbers of data patches.\n";
      return false;
   }
   int const nxp = conn1->xPatchSize();
   if (nxp != conn2->xPatchSize()) {
      pvErrorNoExit() << "Connections \"" << conn1->getName() << "\" and \"" << conn2->getName() << "\" have different values of nxp.\n";
      return false;
   }
   int const nyp = conn1->xPatchSize();
   if (nyp != conn2->yPatchSize()) {
      pvErrorNoExit() << "Connections \"" << conn1->getName() << "\" and \"" << conn2->getName() << "\" have different values of nyp.\n";
      return false;
   }
   int const nfp = conn1->xPatchSize();
   if (nfp != conn2->fPatchSize()) {
      pvErrorNoExit() << "Connections \"" << conn1->getName() << "\" and \"" << conn2->getName() << "\" have different values of nfp.\n";
      return false;
   }
   int const numWeights = nxp*nyp*nfp*numPatches;
   bool connsEqual = true; // Will become false if any nonequal weights are encountered.
   for (int arbor=0; arbor < numArbors; arbor++) {
      pvwdata_t const * buffer1 = conn1->get_wDataStart(arbor);
      pvwdata_t const * buffer2 = conn2->get_wDataStart(arbor);
      for (int widx = 0; widx < numWeights; widx++) {
         pvwdata_t const w1 = buffer1[widx];
         pvwdata_t const w2 = buffer2[widx];
         if (w1 != w2 && std::fabs(w2-w1) > tolerance) {
            int const patchSize = nxp*nyp*nfp;
            int const patchIdx = widx / patchSize; // integer arithmetic
            int const weightIndex = widx % patchSize;
            int x = kxPos(widx, nxp, nyp, nfp);
            int y = kyPos(widx, nxp, nyp, nfp);
            int f = featureIndex(widx, nxp, nyp, nfp);
            pvErrorNoExit() << "Patch index " << patchIdx << ": weights at x=" << x << ", y=" << y << ", p=" << f << " differ:" <<
                  w1 << " versus " << w2 << ".\n";
            connsEqual = false;
         }
      }
   }
   return connsEqual;
}
