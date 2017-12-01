//============================================================================
// Name        : NormalizeSystemTest.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>
#include <normalizers/NormalizeBase.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) { return buildandrun(argc, argv, NULL, customexit); }

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   float tol = 1e-4;

   // check normalizeSum
   Observer *baseObject;
   baseObject                            = hc->getObjectFromName("NormalizeSumConnection");
   NormalizeBase *normalizeSumNormalizer = hc->getNormalizerFromName("NormalizeSumConnection");
   FatalIf(!(normalizeSumNormalizer), "Test failed.\n");
   float normalizeSumStrength    = normalizeSumNormalizer->getStrength();
   baseObject                    = hc->getObjectFromName("NormalizeSumCheck");
   HyPerLayer *normalizeSumCheck = dynamic_cast<HyPerLayer *>(baseObject);
   float normalizeSumValue       = normalizeSumCheck->getLayerData()[0];
   FatalIf(fabsf(normalizeSumValue - normalizeSumStrength) >= tol, "Test failed.\n");

   // check normalizeL2
   baseObject                           = hc->getObjectFromName("NormalizeL2Connection");
   NormalizeBase *normalizeL2Normalizer = hc->getNormalizerFromName("NormalizeL2Connection");
   FatalIf(!(normalizeL2Normalizer), "Test failed.\n");
   float normalizeL2Strength    = normalizeL2Normalizer->getStrength();
   baseObject                   = hc->getObjectFromName("NormalizeL2Check");
   HyPerLayer *normalizeL2Check = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(!(normalizeL2Check), "Test failed.\n");
   float normalizeL2Value = sqrtf(normalizeL2Check->getLayerData()[0]);
   FatalIf(fabsf(normalizeL2Value - normalizeL2Strength) >= tol, "Test failed.\n");

   // check normalizeMax
   baseObject                            = hc->getObjectFromName("NormalizeMaxConnection");
   NormalizeBase *normalizeMaxNormalizer = hc->getNormalizerFromName("NormalizeMaxConnection");
   FatalIf(!(normalizeMaxNormalizer), "Test failed.\n");
   float normalizeMaxStrength    = normalizeMaxNormalizer->getStrength();
   baseObject                    = hc->getObjectFromName("NormalizeMaxCheck");
   HyPerLayer *normalizeMaxCheck = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(!(normalizeMaxCheck), "Test failed.\n");
   float normalizeMaxValue = -FLT_MAX;
   for (int k = 0; k < normalizeMaxCheck->getNumExtended(); k++) {
      float layerData = normalizeMaxCheck->getLayerData()[k];
      if (normalizeMaxValue < layerData) {
         normalizeMaxValue = layerData;
      }
   }
   FatalIf(fabsf(normalizeMaxValue - normalizeMaxStrength) >= tol, "Test failed.\n");

   // check normalizeContrastZeroMean.
   baseObject = hc->getObjectFromName("NormalizeContrastZeroMeanConnection");
   HyPerConn *normalizeContrastZeroMeanConn = dynamic_cast<HyPerConn *>(baseObject);
   FatalIf(!normalizeContrastZeroMeanConn, "Test failed.\n");
   NormalizeBase *normalizeContrastZeroMeanNormalizer =
         hc->getNormalizerFromName("NormalizeContrastZeroMeanConnection");
   FatalIf(!(normalizeContrastZeroMeanNormalizer), "Test failed.\n");
   float normalizeContrastZeroMeanStrength = normalizeContrastZeroMeanNormalizer->getStrength();
   int numNeurons = normalizeContrastZeroMeanConn->getPost()->getNumGlobalNeurons();
   baseObject     = hc->getObjectFromName("NormalizeContrastZeroMeanCheckMean");
   HyPerLayer *normalizeContrastZeroMeanCheckMean = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(!normalizeContrastZeroMeanCheckMean, "Test failed.\n");
   float normalizeContrastZeroMeanValue =
         normalizeContrastZeroMeanCheckMean->getLayerData()[0] / numNeurons;
   FatalIf(fabsf(normalizeContrastZeroMeanValue) >= tol, "Test failed.\n");
   baseObject = hc->getObjectFromName("NormalizeContrastZeroMeanCheckVariance");
   HyPerLayer *normalizeContrastZeroMeanCheckVariance = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(!normalizeContrastZeroMeanCheckVariance, "Test failed.\n");
   float normalizeContrastZeroMeanStDev =
         sqrtf(normalizeContrastZeroMeanCheckVariance->getLayerData()[0] / numNeurons);
   FatalIf(
         fabsf(normalizeContrastZeroMeanStDev - normalizeContrastZeroMeanStrength) >= tol,
         "Test failed.\n");

   return PV_SUCCESS;
}
