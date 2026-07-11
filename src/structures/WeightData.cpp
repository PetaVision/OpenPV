#include "WeightData.hpp"

#include "utils/conversions.hpp"

#include <limits>

namespace PV {

WeightData::WeightData(
      std::string const &allocationMessage,
      int numArbors,
      int patchSizeX, int patchSizeY, int patchSizeF,
      int numDataPatchesX, int numDataPatchesY, int numDataPatchesF) {
   mAllocationMessage     = allocationMessage;
   mNumArbors             = numArbors;
   mPatchSizeX            = patchSizeX;
   mPatchSizeY            = patchSizeY;
   mPatchSizeF            = patchSizeF;
   mPatchSizeOverall      = (long)patchSizeX * (long)patchSizeY * (long)patchSizeF;
   mNumDataPatchesX       = numDataPatchesX;
   mNumDataPatchesY       = numDataPatchesY;
   mNumDataPatchesF       = numDataPatchesF;
   mNumDataPatchesOverall = (long)numDataPatchesX * (long)numDataPatchesY * (long)numDataPatchesF;

   initializeData(allocationMessage);
}

WeightData::~WeightData() {
   long int numArborsL    = getNumArbors();
   long int patchSizeL    = getPatchSizeOverall();
   long int numPatchesL   = getNumDataPatchesOverall();
   long int bytesPerValue = sizeof(float);
   long int allocated     = numArborsL * patchSizeL * numPatchesL * bytesPerValue;
   InfoLog().printf(
           "Deallocation %ld bytes: \"%s\"\n", allocated, mAllocationMessage.c_str());
}

void WeightData::calcExtremeWeights(float &minWeight, float &maxWeight) const {
   float minW = std::numeric_limits<float>::max();
   float maxW = -std::numeric_limits<float>::max();
   int const numArbors = getNumArbors();
   long int const numValues = getNumValuesPerArbor();
   for (int a = 0; a < numArbors; ++a) {
      float const *arborPtr = getData(a);
      for (long int k = 0; k < numValues; ++k) {
         float const value = arborPtr[k];
         minW = value < minW ? value : minW;
         maxW = value > maxW ? value : maxW;
      }
   }
   minWeight = minW;
   maxWeight = maxW;
}

float *WeightData::getDataFromDataIndex(int arbor, long int dataIndex) {
   auto &a = mData[arbor];
   long offset = dataIndex * mPatchSizeOverall;
   return &a[offset]; 
}

float const *WeightData::getDataFromDataIndex(int arbor, long int dataIndex) const {
   auto &a = mData[arbor];
   long offset = dataIndex * mPatchSizeOverall;
   return &a[offset]; 
}

float *WeightData::getDataFromXYF(int arbor, int indexX, int indexY, int indexF) {
   long dataIndex = kIndex(
         indexX, indexY, indexF, getNumDataPatchesX(), getNumDataPatchesY(), getNumDataPatchesF());
   return getDataFromDataIndex(arbor, dataIndex);
}

float const *WeightData::getDataFromXYF(int arbor, int indexX, int indexY, int indexF) const {
   long dataIndex = kIndex(
         indexX, indexY, indexF, getNumDataPatchesX(), getNumDataPatchesY(), getNumDataPatchesF());
   return getDataFromDataIndex(arbor, dataIndex);
}

void WeightData::initializeData(std::string const &allocationMessage) {
   long int numArborsL    = getNumArbors();
   long int patchSizeL    = getPatchSizeOverall();
   long int numPatchesL   = getNumDataPatchesOverall();
   mData.resize(numArborsL);
   for (auto &a : mData) {
      a.resize(patchSizeL * numPatchesL);
   }
   long int bytesPerValue = static_cast<long int>(sizeof(float));
   long int allocated     = numArborsL * patchSizeL * numPatchesL * bytesPerValue;
   InfoLog().printf(
           "Allocation %ld bytes: "
           "\"%s\", %ld arbors, %ld patches, patch size %ld, %ld-byte values.\n",
           allocated,
           mAllocationMessage.c_str(),
           numArborsL,
           numPatchesL,
           patchSizeL,
           bytesPerValue);
}

} // namespace PV
