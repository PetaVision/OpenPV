/*
 * ColumnArchive.hpp
 *
 *  Created on: Jul 5, 2016
 *      Author: pschultz
 */

#ifndef COLUMNARCHIVE_HPP_
#define COLUMNARCHIVE_HPP_

#include <columns/ComponentBasedObject.hpp>
#include <columns/HyPerCol.hpp>
#include <components/PublisherComponent.hpp>
#include <include/PVLayerLoc.h>

struct LayerArchive {
   std::string name;
   PVLayerLoc layerLoc;
   std::vector<float> data;
   bool operator==(LayerArchive const &comparison) const;
   bool operator!=(LayerArchive const &comparison) const { return !(operator==(comparison)); }
   float tolerance;
};

struct ConnArchive {
   std::string name;
   bool sharedWeights;
   int numArbors;
   int numDataPatches;
   int nxp;
   int nyp;
   int nfp;
   std::vector<std::vector<float>> data; // outer index is arbors, inner index is individual
   // weights, with nxp,nyp,nfp,numDataPatches squashed.
   bool operator==(ConnArchive const &comparison) const;
   bool operator!=(ConnArchive const &comparison) const { return !(operator==(comparison)); }
   float tolerance;
};

class ColumnArchive {
  public:
   ColumnArchive(PV::HyPerCol *hc, float layerTolerance, float connTolerance) {
      addCol(hc, layerTolerance, connTolerance);
   }
   virtual ~ColumnArchive() {}

   std::vector<LayerArchive>::size_type getNumLayers() const { return m_layerdata.size(); }
   std::vector<LayerArchive> const &getLayerData() const { return m_layerdata; }
   LayerArchive const &getLayerData(int l) const { return m_layerdata.at(l); }

   std::vector<ConnArchive>::size_type getNumConns() const { return m_conndata.size(); }
   std::vector<ConnArchive> const &getConnData() const { return m_conndata; }
   ConnArchive const &getConnData(int c) const { return m_conndata.at(c); }

   bool operator==(ColumnArchive const &comparison) const;
   bool operator!=(ColumnArchive const &comparison) const { return !(operator==(comparison)); }

  private:
   ColumnArchive() {}
   void addCol(PV::HyPerCol *hc, float layerTolerance, float connTolerance);
   void addLayer(PV::PublisherComponent *layer, float layerTolerance);
   void addConn(PV::ComponentBasedObject *conn, float connTolerance);

  private:
   std::vector<LayerArchive> m_layerdata;
   std::vector<ConnArchive> m_conndata;
};

template <typename T>
bool compareFields(char const *type, char const *field, T val1, T val2) {
   if (val1 != val2) {
      ErrorLog() << type << " have different " << field << ": " << val1 << " versus " << val2
                 << ".\n";
      return false;
   }
   else {
      return true;
   }
}

#endif /* COLUMNARCHIVE_HPP_ */
