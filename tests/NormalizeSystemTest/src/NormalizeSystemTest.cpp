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
   HyPerConn *hyperconn;
   baseObject                            = hc->getObjectFromName("NormalizeSumConnection");
   hyperconn                             = dynamic_cast<HyPerConn *>(baseObject);
   NormalizeBase *normalizeSumNormalizer = hyperconn->getComponentByType<NormalizeBase>();
   FatalIf(!normalizeSumNormalizer, "%s has no normalizer.\n", hyperconn->getDescription_c());
   float normalizeSumStrength    = normalizeSumNormalizer->getStrength();
   baseObject                    = hc->getObjectFromName("NormalizeSumCheck");
   HyPerLayer *normalizeSumCheck = dynamic_cast<HyPerLayer *>(baseObject);
   float normalizeSumValue       = normalizeSumCheck->getLayerData()[0];
   FatalIf(fabsf(normalizeSumValue - normalizeSumStrength) >= tol, "Test failed.\n");

   // check normalizeL2
   baseObject                           = hc->getObjectFromName("NormalizeL2Connection");
   hyperconn                            = dynamic_cast<HyPerConn *>(baseObject);
   NormalizeBase *normalizeL2Normalizer = hyperconn->getComponentByType<NormalizeBase>();
   FatalIf(!normalizeL2Normalizer, "%s has no normalizer.\n", hyperconn->getDescription_c());
   float normalizeL2Strength    = normalizeL2Normalizer->getStrength();
   baseObject                   = hc->getObjectFromName("NormalizeL2Check");
   HyPerLayer *normalizeL2Check = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(!(normalizeL2Check), "Test failed.\n");
   float normalizeL2Value = sqrtf(normalizeL2Check->getLayerData()[0]);
   FatalIf(fabsf(normalizeL2Value - normalizeL2Strength) >= tol, "Test failed.\n");

   // check normalizeMax
   baseObject                            = hc->getObjectFromName("NormalizeMaxConnection");
   hyperconn                             = dynamic_cast<HyPerConn *>(baseObject);
   NormalizeBase *normalizeMaxNormalizer = hyperconn->getComponentByType<NormalizeBase>();
   FatalIf(!normalizeMaxNormalizer, "%s has no normalizer.\n", hyperconn->getDescription_c());
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
   hyperconn  = dynamic_cast<HyPerConn *>(baseObject);
   FatalIf(!hyperconn, "Test failed.\n");
   NormalizeBase *normalizeContrastZeroMeanNormalizer =
         hyperconn->getComponentByType<NormalizeBase>();
   FatalIf(
         !normalizeContrastZeroMeanNormalizer,
         "%s has no normalizer.\n",
         hyperconn->getDescription_c());
   float normalizeContrastZeroMeanStrength = normalizeContrastZeroMeanNormalizer->getStrength();
   int numNeurons                          = hyperconn->getPost()->getNumGlobalNeurons();
   baseObject = hc->getObjectFromName("NormalizeContrastZeroMeanCheckMean");
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
