#include "WeightData.hpp"

#include "utils/conversions.hpp"

namespace PV {

WeightData::WeightData(
      int numArbors,
      int patchSizeX, int patchSizeY, int patchSizeF,
      int numDataPatchesX, int numDataPatchesY, int numDataPatchesF) {
   mNumArbors = numArbors;
   mPatchSizeX = patchSizeX;
   mPatchSizeY = patchSizeY;
   mPatchSizeF = patchSizeF;
   mPatchSizeOverall = static_cast<long>(patchSizeX * patchSizeY * patchSizeF);
   mNumDataPatchesX = numDataPatchesX;
   mNumDataPatchesY = numDataPatchesY;
   mNumDataPatchesF = numDataPatchesF;

   initializeData();
}

float *WeightData::getDataFromXYF(int arbor, int indexX, int indexY, int indexF) {
   int dataIndex = kIndex(
         indexX, indexY, indexF, getNumDataPatchesX(), getNumDataPatchesY(), getNumDataPatchesF());
   return getDataFromDataIndex(arbor, dataIndex);
}

float const *WeightData::getDataFromXYF(int arbor, int indexX, int indexY, int indexF) const {
   int dataIndex = kIndex(
         indexX, indexY, indexF, getNumDataPatchesX(), getNumDataPatchesY(), getNumDataPatchesF());
   return getDataFromDataIndex(arbor, dataIndex);
}

void WeightData::initializeData() {
   mData.resize(getNumArbors());
   for (auto &a : mData) {
      a.resize(getPatchSizeOverall() * getNumDataPatchesOverall());
   }
}

} // namespace PV
