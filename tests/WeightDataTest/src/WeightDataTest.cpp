/*
 * WeightsClassTest.cpp
 */

#include <include/pv_common.h> // 
#include <structures/WeightData.hpp>
#include <utils/PathComponents.hpp> // baseName
#include <utils/PVLog.hpp>
#include <utils/conversions.hpp> // kxPos, kyPos, featureIndex

#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE

using namespace PV;

struct Dimensions {
   int patchSizeX;
   int patchSizeY;
   int patchSizeF;
   int numPatchesX;
   int numPatchesY;
   int numPatchesF;
};

WeightData makeTestWeights(Dimensions const &dimensions) {
   WeightData weightData(
         1 /*numArbors*/,
         dimensions.patchSizeX, dimensions.patchSizeY, dimensions.patchSizeF,
         dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
   return weightData;
}

int testDimensions() {
   int status = PV_SUCCESS;
   Dimensions dimensions{7, 5, 4, 8, 6, 3};
   auto weightData = makeTestWeights(dimensions);
   if (weightData.getNumArbors() != 1) {
      ErrorLog().printf(
            "Observed NumArbors is %d instead of expected NumArbors %d\n",
            weightData.getNumArbors(),
            1);
      status = PV_FAILURE;
   }
   if (weightData.getPatchSizeX() != dimensions.patchSizeX) {
      ErrorLog().printf(
            "Observed PatchSizeX is %d instead of expected PatchSizeX %d\n",
            weightData.getPatchSizeX(),
            dimensions.numPatchesX);
      status = PV_FAILURE;
   }
   if (weightData.getPatchSizeY() != dimensions.patchSizeY) {
      ErrorLog().printf(
            "Observed PatchSizeY is %d instead of expected PatchSizeY %d\n",
            weightData.getPatchSizeY(),
            dimensions.numPatchesY);
      status = PV_FAILURE;
   }
   if (weightData.getPatchSizeF() != dimensions.patchSizeF) {
      ErrorLog().printf(
            "Observed PatchSizeF is %d instead of expected PatchSizeF %d\n",
            weightData.getPatchSizeF(),
            dimensions.numPatchesF);
      status = PV_FAILURE;
   }
   long patchSizeOverall =
         static_cast<long>(dimensions.patchSizeX * dimensions.patchSizeY * dimensions.patchSizeF);
   if (weightData.getPatchSizeOverall() != patchSizeOverall) {
      ErrorLog().printf(
            "Observed PatchSizeOverall is %d instead of expected PatchSizeOverall %d\n",
            weightData.getPatchSizeOverall(),
            patchSizeOverall);
      status = PV_FAILURE;
   }
   if (weightData.getNumDataPatchesX() != dimensions.numPatchesX) {
      ErrorLog().printf(
            "Observed NumPatchesX is %d instead of expected NumPatchesX %d\n",
            weightData.getNumDataPatchesX(),
            dimensions.numPatchesX);
      status = PV_FAILURE;
   }
   if (weightData.getNumDataPatchesY() != dimensions.numPatchesY) {
      ErrorLog().printf(
            "Observed NumPatchesY is %d instead of expected NumPatchesY %d\n",
            weightData.getNumDataPatchesY(),
            dimensions.numPatchesY);
      status = PV_FAILURE;
   }
   if (weightData.getNumDataPatchesF() != dimensions.numPatchesF) {
      ErrorLog().printf(
            "Observed NumPatchesF is %d instead of expected NumPatchesF %d\n",
            weightData.getNumDataPatchesF(),
            dimensions.numPatchesF);
      status = PV_FAILURE;
   }
   long numPatchesOverall = static_cast<long>(
         dimensions.numPatchesX * dimensions.numPatchesY * dimensions.numPatchesF);
   if (weightData.getNumDataPatchesOverall() != numPatchesOverall) {
      ErrorLog().printf(
            "Observed NumPatchesOverall is %d instead of expected NumPatchesOverall %d\n",
            weightData.getNumDataPatchesOverall(),
            numPatchesOverall);
      status = PV_FAILURE;
   }
   long numValues = patchSizeOverall * numPatchesOverall;
   if (weightData.getNumValuesPerArbor() != numValues) {
      ErrorLog().printf(
            "Observed NumPatchesOverall is %d instead of expected NumPatchesOverall %d\n",
            weightData.getNumDataPatchesOverall(),
            numPatchesOverall);
      status = PV_FAILURE;
   }
   return status;
}

int testGetData() {
   int status = PV_SUCCESS;
   Dimensions dimensions{7, 5, 4, 8, 6, 3};
   auto weightData = makeTestWeights(dimensions);
   float *writePointer = weightData.getData(0);
   long numValues = weightData.getNumValuesPerArbor();
   for (long k = 0; k < numValues; ++k) {
      writePointer[k] = static_cast<float>(2L * k + 1L);
   }

   float const *readPointer = weightData.getData(0);
   for (long k = numValues - 1L; k >= 0L; --k) {
      double expected = static_cast<double>(2*k + 1);
      double observed = static_cast<double>(readPointer[k]);
      if (observed != expected) {
         ErrorLog().printf(
               "WeightData::getData() failed: index %ld expected %f, observed %f\n",
               k, expected, observed);
         status = PV_FAILURE;
      }
   }
   return status;
}

int testGetDataFromDataIndex() {
   int status = PV_SUCCESS;
   Dimensions dimensions{7, 5, 4, 8, 6, 3};
   auto weightData = makeTestWeights(dimensions);
   int numPatches = static_cast<int>(weightData.getNumDataPatchesOverall());
   int numValues  = static_cast<int>(weightData.getPatchSizeOverall());
   for (int p = 0; p < numPatches; ++p) {
      float *writePointer = weightData.getDataFromDataIndex(0 /*arbor*/, p);
      for (int k = 0; k < numValues; ++k) {
         writePointer[k] = static_cast<float>(3 * (p * numValues + k) + 2);
      }
   }
   for (int p = 0; p < numPatches; ++p) {
      float const *readPointer = weightData.getDataFromDataIndex(0 /*arbor*/, p);
      for (int k = 0L; k < numValues; ++k) {
         double expected = static_cast<double>(3 * (p * numValues + k) + 2);
         double observed = static_cast<double>(readPointer[k]);
         if (observed != expected) {
            ErrorLog().printf(
                  "WeightData::getDataFromDataIndex() failed: "
                  "patch index %ld, value %d, expected %f, observed %f\n",
                  p, k, expected, observed);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int testGetDataFromXYF() {
   int status = PV_SUCCESS;
   Dimensions dimensions{7, 5, 4, 8, 6, 3};
   auto weightData = makeTestWeights(dimensions);
   int numPatches  = static_cast<int>(weightData.getNumDataPatchesOverall());
   int numValues   = static_cast<int>(weightData.getPatchSizeOverall());
   for (int p = 0; p < numPatches; ++p) {
      int xIndex = kxPos(p, dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
      int yIndex = kyPos(p, dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
      int fIndex =
            featureIndex(p, dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
      float *writePointer = weightData.getDataFromXYF(0 /*arbor*/, xIndex, yIndex, fIndex);
      for (int k = 0; k < numValues; ++k) {
         writePointer[k] = static_cast<float>(4 * (p * numValues + k) - 2);
      }
   }
   for (int p = 0; p < numPatches; ++p) {
      int xIndex = kxPos(p, dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
      int yIndex = kyPos(p, dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
      int fIndex =
            featureIndex(p, dimensions.numPatchesX, dimensions.numPatchesY, dimensions.numPatchesF);
      float const *readPointer = weightData.getDataFromXYF(0 /*arbor*/, xIndex, yIndex, fIndex);
      for (int k = 0; k < numValues; ++k) {
         double expected = static_cast<double>(4 * (p * numValues + k) - 2);
         double observed = static_cast<double>(readPointer[k]);
         if (observed != expected) {
            ErrorLog().printf(
                  "WeightData::getDataFromXYF() failed: "
                  "patch (%d,%d,%d), value %d, expected %f, observed %f\n",
                  xIndex, yIndex, fIndex, k, expected, observed);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int main(int argc, char *argv[]) {
   std::string programName = baseName(argv[0]);
   std::string logFilename = programName + "_1.log";
   setLogFile(logFilename);

   int status = PV_SUCCESS;
   if (testDimensions() != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testGetData() != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testGetDataFromDataIndex() != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testGetDataFromXYF() != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      InfoLog() << programName << " passed.\n";
      return EXIT_SUCCESS;
   }
   else {
      ErrorLog() << programName << " FAILED.\n";
      return EXIT_FAILURE;
   }
}
