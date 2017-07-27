/*
 * ColumnArchive.cpp
 *
 *  Created on: Jul 5, 2016
 *      Author: pschultz
 */

#include "ColumnArchive.hpp"
#include <cmath>
// Definitions of constructor and destructor are both empty, defined in header

bool LayerArchive::operator==(LayerArchive const &comparison) const {
   bool areEqual = true;
   areEqual &= compareFields("Layers", "names", name, comparison.name);
   areEqual &=
         compareFields("Layers", "nbatch values", layerLoc.nbatch, comparison.layerLoc.nbatch);
   areEqual &= compareFields("Layers", "nx values", layerLoc.nx, comparison.layerLoc.nx);
   areEqual &= compareFields("Layers", "ny values", layerLoc.ny, comparison.layerLoc.ny);
   areEqual &= compareFields("Layers", "nf values", layerLoc.nf, comparison.layerLoc.nf);
   areEqual &= compareFields("Layers", "kb0 values", layerLoc.kb0, comparison.layerLoc.kb0);
   areEqual &= compareFields("Layers", "kx0 values", layerLoc.kx0, comparison.layerLoc.kx0);
   areEqual &= compareFields("Layers", "ky0 values", layerLoc.ky0, comparison.layerLoc.ky0);
   // Can have different halos.
   int const nx        = layerLoc.nx;
   int const ny        = layerLoc.ny;
   int const nf        = layerLoc.nf;
   PVHalo const &halo1 = layerLoc.halo;
   PVHalo const &halo2 = comparison.layerLoc.halo;
   float const *dat1   = data.data();
   float const *dat2   = comparison.data.data();
   if (areEqual) {
      for (int b = 0; b < layerLoc.nbatch; b++) {
         int const N = layerLoc.nx * layerLoc.ny * layerLoc.nf;
         for (int n = 0; n < N; n++) {
            int nExt1 = kIndexExtended(n, nx, ny, nf, halo1.lt, halo1.rt, halo1.dn, halo1.up);
            int nExt2 = kIndexExtended(n, nx, ny, nf, halo2.lt, halo2.rt, halo2.dn, halo2.up);
            if (std::fabs(dat1[nExt1] - dat2[nExt2]) > tolerance) {
               int const x = kxPos(n, nx, ny, nf) + layerLoc.kx0;
               int const y = kyPos(n, nx, ny, nf) + layerLoc.ky0;
               int const f = featureIndex(n, nx, ny, nf);
               std::stringstream fieldstream("");
               fieldstream << "values in batch element " << b << " at x=" << x << ",y=" << y
                           << ",f=";
               compareFields("Activities", fieldstream.str().c_str(), dat1[nExt1], dat2[nExt2]);
               areEqual = false;
            }
         }
      }
   }
   return areEqual;
}

bool ConnArchive::operator==(ConnArchive const &comparison) const {
   bool areEqual = true;
   areEqual &= compareFields("Connections", "names", name, comparison.name);
   areEqual &= compareFields(
         "Connections", "sharedWeights flags", sharedWeights, comparison.sharedWeights);
   areEqual &= compareFields(
         "Connections", "numbers of data patches", numDataPatches, comparison.numDataPatches);
   areEqual &= compareFields("Connections", "nx values", nxp, comparison.nxp);
   areEqual &= compareFields("Connections", "ny values", nyp, nyp);
   areEqual &= compareFields("Connections", "nf values", nfp, nfp);
   if (areEqual) {
      int const patchSize = nxp * nyp * nfp;
      for (int arborIdx = 0; arborIdx < numArbors; arborIdx++) {
         std::vector<float> const &arbor1 = data.at(arborIdx);
         std::vector<float> const &arbor2 = comparison.data.at(arborIdx);
         for (int patchIdx = 0; patchIdx < numDataPatches; patchIdx++) {
            float const *patchdata1 = &arbor1.data()[patchIdx * patchSize];
            float const *patchdata2 = &arbor2.data()[patchIdx * patchSize];
            for (int widx; widx < patchSize; widx++) {
               if (std::fabs(patchdata1[widx] - patchdata2[widx]) > tolerance) {
                  int const x = kxPos(widx, nxp, nyp, nfp);
                  int const y = kyPos(widx, nxp, nyp, nfp);
                  int const f = featureIndex(widx, nxp, nyp, nfp);
                  std::stringstream fieldstream("");
                  fieldstream << "values in data patch " << patchIdx << " at x=" << x << ",y=" << y
                              << ",f=";
                  compareFields(
                        "Weights", fieldstream.str().c_str(), patchdata1[widx], patchdata2[widx]);
                  areEqual = false;
               }
            }
         }
      }
   }
   return areEqual;
}

bool ColumnArchive::operator==(ColumnArchive const &comparison) const {
   bool areEqual = true;
   areEqual &= compareFields(
         "The columns", "numbers of layers", m_layerdata.size(), comparison.m_layerdata.size());
   for (auto const &layer1 : m_layerdata) {
      bool found = false;
      for (auto const &layer2 : comparison.m_layerdata) {
         if (layer2.name == layer1.name) {
            found = true;
            if (layer1 != layer2) {
               areEqual = false; // diagnostics get printed by LayerArchive::operator==
            }
            break;
         }
      }
      if (!found) {
         ErrorLog() << "Column 1 has a layer \"" << layer1.name << " \" but column 2 does not.\n";
         areEqual = false;
      }
   }

   areEqual &= compareFields(
         "The columns", "numbers of connections", m_conndata.size(), comparison.m_conndata.size());
   int const nc = m_conndata.size();
   for (auto const &conn1 : m_conndata) {
      bool found = false;
      for (auto const &conn2 : comparison.m_conndata) {
         if (conn2.name == conn1.name) {
            found = true;
            if (conn1 != conn2) {
               areEqual = false; // diagnostics get printed by LayerArchive::operator==
            }
            break;
         }
      }
      if (!found) {
         ErrorLog() << "Column 1 has a connection \"" << conn1.name
                    << " \" but column 2 does not.\n";
         areEqual = false;
      }
   }
   return areEqual;
}

void ColumnArchive::addLayer(PV::HyPerLayer *layer, float layerTolerance) {
   std::vector<LayerArchive>::size_type sz = m_layerdata.size();
   m_layerdata.resize(sz + 1);
   LayerArchive &latestLayer = m_layerdata.at(sz);
   latestLayer.name          = layer->getName();
   latestLayer.layerLoc      = layer->getLayerLoc()[0];
   float const *ldatastart   = layer->getLayerData();
   float const *ldataend     = &ldatastart[layer->getNumExtendedAllBatches()];
   latestLayer.data          = std::vector<float>(ldatastart, ldataend);
   latestLayer.tolerance     = layerTolerance;
}

void ColumnArchive::addConn(PV::HyPerConn *conn, float connTolerance) {
   std::vector<ConnArchive>::size_type sz = m_conndata.size();
   m_conndata.resize(sz + 1);
   ConnArchive &latestConnection = m_conndata.at(sz);
   latestConnection.name         = conn->getName();
   int const numArbors           = conn->numberOfAxonalArborLists();

   latestConnection.numArbors = numArbors;
   latestConnection.nxp       = conn->xPatchSize();
   latestConnection.nyp       = conn->yPatchSize();
   latestConnection.nfp       = conn->fPatchSize();
   latestConnection.data.resize(numArbors);
   int const datasize = latestConnection.nxp * latestConnection.nyp * latestConnection.nfp
                        * latestConnection.numDataPatches;
   for (int arbor = 0; arbor < numArbors; arbor++) {
      float const *cdatastart         = conn->get_wDataStart(arbor);
      float const *cdataend           = &cdatastart[datasize];
      latestConnection.data.at(arbor) = std::vector<float>(cdatastart, cdataend);
   }
   latestConnection.tolerance = connTolerance;
}

void ColumnArchive::addCol(PV::HyPerCol *hc, float layerTolerance, float connTolerance) {
   PV::Observer *firstObject = hc->getNextObject(nullptr);
   for (PV::Observer *obj = firstObject; obj != nullptr; obj = hc->getNextObject(obj)) {
      PV::HyPerLayer *layer = dynamic_cast<PV::HyPerLayer *>(obj);
      if (layer != nullptr) {
         addLayer(layer, layerTolerance);
      }
      PV::HyPerConn *conn = dynamic_cast<PV::HyPerConn *>(obj);
      if (conn != nullptr) {
         addConn(conn, connTolerance);
      }
   }
}
