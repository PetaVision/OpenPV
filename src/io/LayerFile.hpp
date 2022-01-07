/*
 * LayerFile.hpp
 *
 *  Created on: May 3, 2021
 *      Author: peteschultz
 */

#ifndef LAYERFILE_HPP_
#define LAYERFILE_HPP_

#include "include/PVLayerLoc.h"
#include "io/FileManager.hpp"
#include "io/LayerBatchGatherScatter.hpp"
#include "io/LayerIO.hpp"

#include <memory>
#include <string>

namespace PV {

/**
 * A class to manage nonsparse activity PVP files. It internally handles all MPI gather/scatter
 * operations, M-to-N communication, and PVP file format details. All file operations treat
 * the layer state, i.e. the data of all batch elements at a single timestep, as a unit.
 */
class LayerFile {
  public:
   LayerFile(
      std::shared_ptr <FileManager const> fileManager,
      std::string const &path,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      bool readOnlyFlag,
      bool verifyWrites);

   LayerFile() = delete;

   ~LayerFile();

   void read();
   void read(double &timestamp);
   void write(double timestamp);

   void truncate(int index);

   int getIndex() const { return mIndex; }
   void setIndex(int index);

   std::string const &getPath() const { return mPath; }

   float const *getDataLocation(int index) const { return mDataLocations.at(index); }
   float *getDataLocation(int index) { return mDataLocations.at(index); }
   void setDataLocation(float *location, int index) { mDataLocations.at(index) = location; }

  private:
   void initializeGatherScatter();
   void initializeLayerIO();

   bool isRoot() { return mFileManager->isRoot(); }

   void readInternal(double &timestamp, bool checkTimestampConsistency);

  private:
   std::shared_ptr<FileManager const> mFileManager = nullptr;
   std::string mPath;
   PVLayerLoc mLayerLoc;
   bool mDataExtended;
   bool mFileExtended;
   bool mReadOnly;
   bool mVerifyWrites;

   int mIndex = 0;
   std::vector<float*> mDataLocations;

   std::unique_ptr<LayerBatchGatherScatter> mGatherScatter;
   std::unique_ptr<LayerIO> mLayerIO;
};

} // namespacePV

#endif // LAYERFILE_HPP_
