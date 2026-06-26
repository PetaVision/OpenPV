#ifndef WEIGHTDATA_HPP_
#define WEIGHTDATA_HPP_

#include <string>
#include <vector>
#include "utils/PVLog.hpp"

namespace PV {

class WeightData {

  public:
   WeightData(
         std::string const &allocationMessage,
         int numArbors,
         int patchSizeX, int patchSizeY, int patchSizeF,
         int numDataPatchesX, int numDataPatchesY, int numDataPatchesF);
   WeightData() = delete;
   ~WeightData();

   void calcExtremeWeights(float &minWeight, float &maxWeight) const;

   float *getData(int arbor) { return mData.at(arbor).data(); }
   float const *getData(int arbor) const { return mData.at(arbor).data(); }

   float *getDataFromDataIndex(int arbor, long int dataIndex);
   float const *getDataFromDataIndex(int arbor, long int dataIndex) const;

   float *getDataFromXYF(int arbor, int indexX, int indexY, int indexF);
   float const *getDataFromXYF(int arbor, int indexX, int indexY, int indexF) const;

   // accessor function members (get-methods)
   int getNumArbors() const { return mNumArbors; }
   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const { return mPatchSizeOverall; }
   int getNumDataPatchesX() const { return mNumDataPatchesX; }
   int getNumDataPatchesY() const { return mNumDataPatchesY; }
   int getNumDataPatchesF() const { return mNumDataPatchesF; }
   long getNumDataPatchesOverall() const { return mNumDataPatchesOverall; }
   long getNumValuesPerArbor() const { return getPatchSizeOverall() * getNumDataPatchesOverall(); }

  private:
   void initializeData(std::string const &allocationMessage);

  private:
   int mNumArbors;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   long mPatchSizeOverall;
   int mNumDataPatchesX;
   int mNumDataPatchesY;
   int mNumDataPatchesF;
   long mNumDataPatchesOverall;

   std::vector<std::vector<float>> mData;
   std::string mAllocationMessage;

}; // class WeightData

} // namespace PV

#endif // WEIGHTDATA_HPP_
