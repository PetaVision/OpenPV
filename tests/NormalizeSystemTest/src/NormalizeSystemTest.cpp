//============================================================================
// Name        : NormalizeSystemTest.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <columns/buildandrun.hpp>
#include <normalizers/NormalizeBase.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) { return buildandrun(argc, argv, NULL, customexit); }

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   float tol = 1e-4;

   // check normalizeSum
   BaseConnection *baseConn;
   baseConn                    = hc->getConnFromName("NormalizeSumConnection");
   HyPerConn *normalizeSumConn = dynamic_cast<HyPerConn *>(baseConn);
   FatalIf(!(normalizeSumConn), "Test failed.\n");
   NormalizeBase *normalizeSumNormalizer = normalizeSumConn->getNormalizer();
   FatalIf(!(normalizeSumNormalizer), "Test failed.\n");
   float normalizeSumStrength    = normalizeSumNormalizer->getStrength();
   HyPerLayer *normalizeSumCheck = hc->getLayerFromName("NormalizeSumCheck");
   float normalizeSumValue       = normalizeSumCheck->getLayerData()[0];
   FatalIf(!(fabsf(normalizeSumValue - normalizeSumStrength) < tol), "Test failed.\n");

   // check normalizeL2
   baseConn                   = hc->getConnFromName("NormalizeL2Connection");
   HyPerConn *normalizeL2Conn = dynamic_cast<HyPerConn *>(baseConn);
   FatalIf(!(normalizeL2Conn), "Test failed.\n");
   NormalizeBase *normalizeL2Normalizer = normalizeL2Conn->getNormalizer();
   FatalIf(!(normalizeL2Normalizer), "Test failed.\n");
   float normalizeL2Strength    = normalizeL2Normalizer->getStrength();
   HyPerLayer *normalizeL2Check = hc->getLayerFromName("NormalizeL2Check");
   FatalIf(!(normalizeL2Check), "Test failed.\n");
   float normalizeL2Value = sqrtf(normalizeL2Check->getLayerData()[0]);
   FatalIf(!(fabsf(normalizeL2Value - normalizeL2Strength) < tol), "Test failed.\n");

   // check normalizeMax
   baseConn                    = hc->getConnFromName("NormalizeMaxConnection");
   HyPerConn *normalizeMaxConn = dynamic_cast<HyPerConn *>(baseConn);
   FatalIf(!(normalizeMaxConn), "Test failed.\n");
   NormalizeBase *normalizeMaxNormalizer = normalizeMaxConn->getNormalizer();
   FatalIf(!(normalizeMaxNormalizer), "Test failed.\n");
   float normalizeMaxStrength    = normalizeMaxNormalizer->getStrength();
   HyPerLayer *normalizeMaxCheck = hc->getLayerFromName("NormalizeMaxCheck");
   FatalIf(!(normalizeMaxCheck), "Test failed.\n");
   float normalizeMaxValue = -FLT_MAX;
   for (int k = 0; k < normalizeMaxCheck->getNumExtended(); k++) {
      float layerData = normalizeMaxCheck->getLayerData()[k];
      if (normalizeMaxValue < layerData) {
         normalizeMaxValue = layerData;
      }
   }
   FatalIf(!(fabsf(normalizeMaxValue - normalizeMaxStrength) < tol), "Test failed.\n");

   // check normalizeContrastZeroMean.
   baseConn = hc->getConnFromName("NormalizeContrastZeroMeanConnection");
   HyPerConn *normalizeContrastZeroMeanConn = dynamic_cast<HyPerConn *>(baseConn);
   FatalIf(!(normalizeContrastZeroMeanConn), "Test failed.\n");
   NormalizeBase *normalizeContrastZeroMeanNormalizer =
         normalizeContrastZeroMeanConn->getNormalizer();
   FatalIf(!(normalizeContrastZeroMeanNormalizer), "Test failed.\n");
   float normalizeContrastZeroMeanStrength = normalizeContrastZeroMeanNormalizer->getStrength();
   int numNeurons = normalizeContrastZeroMeanConn->postSynapticLayer()->getNumGlobalNeurons();
   HyPerLayer *normalizeContrastZeroMeanCheckMean =
         hc->getLayerFromName("NormalizeContrastZeroMeanCheckMean");
   FatalIf(!(normalizeContrastZeroMeanCheckMean), "Test failed.\n");
   float normalizeContrastZeroMeanValue =
         normalizeContrastZeroMeanCheckMean->getLayerData()[0] / numNeurons;
   FatalIf(!(fabsf(normalizeContrastZeroMeanValue) < tol), "Test failed.\n");
   HyPerLayer *normalizeContrastZeroMeanCheckVariance =
         hc->getLayerFromName("NormalizeContrastZeroMeanCheckVariance");
   FatalIf(!(normalizeContrastZeroMeanCheckVariance), "Test failed.\n");
   float normalizeContrastZeroMeanStDev =
         sqrtf(normalizeContrastZeroMeanCheckVariance->getLayerData()[0] / numNeurons);
   FatalIf(
         !(fabsf(normalizeContrastZeroMeanStDev - normalizeContrastZeroMeanStrength) < tol),
         "Test failed.\n");

   return PV_SUCCESS;
}
