#ifndef WEIGHTDATA_HPP_
#define WEIGHTDATA_HPP_

#include <vector>

namespace PV {

class WeightData {

  public:
   WeightData(
         int numArbors,
         int patchSizeX, int patchSizeY, int patchSizeF,
         int numDataPatchesX, int numDataPatchesY, int numDataPatchesF);
   WeightData() = delete;
   ~WeightData() {};

   float *getData(int arbor) { return mData.at(arbor).data(); }
   float const *getData(int arbor) const { return mData.at(arbor).data(); }

   float *getDataFromXYF(int arbor, int indexX, int indexY, int indexF);
   float const *getDataFromXYF(int arbor, int indexX, int indexY, int indexF) const;

   inline float *getDataFromDataIndex(int arbor, int dataIndex) {
      auto &a = mData[arbor];
      long offset = static_cast<long>(dataIndex) * mPatchSizeOverall;
      return &a[offset]; 
   }

   inline float const *getDataFromDataIndex(int arbor, int dataIndex) const {
      auto &a = mData[arbor];
      long offset = static_cast<long>(dataIndex) * mPatchSizeOverall;
      return &a[offset]; 
   }

   // accessor function members (get-methods)
   int getNumArbors() const { return mNumArbors; }
   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const { return mPatchSizeOverall; }
   int getNumDataPatchesX() const { return mNumDataPatchesX; }
   int getNumDataPatchesY() const { return mNumDataPatchesY; }
   int getNumDataPatchesF() const { return mNumDataPatchesF; }
   long getNumDataPatchesOverall() const {
      return static_cast<long>(getNumDataPatchesX() * getNumDataPatchesY() * getNumDataPatchesF());
   }
   long getNumValuesPerArbor() const { return getPatchSizeOverall() * getNumDataPatchesOverall(); }

  private:
   void initializeData();

  private:
   int mNumArbors;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   long mPatchSizeOverall;
   int mNumDataPatchesX;
   int mNumDataPatchesY;
   int mNumDataPatchesF;

   std::vector<std::vector<float>> mData;

}; // class WeightData

} // namespace PV

#endif // WEIGHTDATA_HPP_
