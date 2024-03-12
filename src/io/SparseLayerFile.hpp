/*
 * SparseLayerFile.hpp
 *
 *  Created on: May 13, 2021
 *      Author: peteschultz
 */

#ifndef SPARSELAYERFILE_HPP_
#define SPARSELAYERFILE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "include/PVLayerLoc.hpp"
#include "io/FileManager.hpp"
#include "io/SparseLayerBatchGatherScatter.hpp"
#include "io/SparseLayerIO.hpp"
#include "structures/SparseList.hpp"

#include <memory>
#include <string>

namespace PV {

/**
 * A class to manage sparse activity PVP files. It internally handles all MPI gather/scatter
 * operations, M-to-N communication, and PVP file format details. All file operations treat
 * the layer state, i.e. the data of all batch elements at a single timestep, as a unit.
 */
class SparseLayerFile : public CheckpointerDataInterface {
  public:
   SparseLayerFile(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &path,
         PVLayerLoc const &layerLoc,
         bool dataExtendedFlag,
         bool fileExtendedFlag,
         bool readOnlyFlag,
         bool clobberFlag,
         bool verifyWrites);

   SparseLayerFile() = delete;

   ~SparseLayerFile();

   void read();
   void read(double &timestamp);
   void write(double timestamp);

   void truncate(int index);

   int getIndex() const { return mIndex; }
   void setIndex(int index);

   std::string const &getPath() const { return mPath; }

   SparseList<float> *getListLocation(int index) const { return mSparseListLocations.at(index); }
   void setListLocation(SparseList<float> *listPtr, int index) {
      mSparseListLocations.at(index) = listPtr;
   }

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status processCheckpointRead(double simTime) override;

  private:
   int initializeCheckpointerDataInterface();
   void initializeGatherScatter();
   void initializeSparseLayerIO(bool clobberFlag);

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
   std::vector<SparseList<float> *> mSparseListLocations;

   std::unique_ptr<SparseLayerBatchGatherScatter> mGatherScatter;
   std::unique_ptr<SparseLayerIO> mSparseLayerIO;

   // The following values are written during checkpointing.
   // It would be more logical to write the value of mIndex, but for reasons of
   // backward compatibility, we continue to write the old values.
   int mNumFramesSparse = 0; // number of batch elements handled by an MPIBlock;
   // that is, Index * MPIBlock->BatchDimension * loc->nbatch;
   long mFileStreamReadPos  = 0L; // Input file position of the SparseLayerIO's FileStream
   long mFileStreamWritePos = 0L; // Output file position of the SparseLayerIO's FileStream
};

} // namespace PV

#endif // SPARSELAYERFILE_HPP_
