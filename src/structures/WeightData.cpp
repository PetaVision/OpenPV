#include "WeightData.hpp"

#include "utils/conversions.hpp"

#include <limits>

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

void WeightData::calcExtremeWeights(float &minWeight, float &maxWeight) const {
   float minW = std::numeric_limits<float>::max();
   float maxW = -std::numeric_limits<float>::max();
   int const numArbors = getNumArbors();
   int const numValues = getNumValuesPerArbor();
   for (int a = 0; a < numArbors; ++a) {
      float const *arborPtr = getData(a);
      for (int k = 0; k < numValues; ++k) {
         float const value = arborPtr[k];
         minW = value < minW ? value : minW;
         maxW = value > maxW ? value : maxW;
      }
   }
   minWeight = minW;
   maxWeight = maxW;
}

float *WeightData::getDataFromDataIndex(int arbor, int dataIndex) {
   auto &a = mData[arbor];
   long offset = static_cast<long>(dataIndex) * mPatchSizeOverall;
   return &a[offset]; 
}

float const *WeightData::getDataFromDataIndex(int arbor, int dataIndex) const {
   auto &a = mData[arbor];
   long offset = static_cast<long>(dataIndex) * mPatchSizeOverall;
   return &a[offset]; 
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
