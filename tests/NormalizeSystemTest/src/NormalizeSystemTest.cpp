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

ComponentBasedObject *getObject(HyPerCol *hc, char const *objectName) {
   return dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(objectName));
}

int main(int argc, char *argv[]) { return buildandrun(argc, argv, NULL, customexit); }

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   float tol = 1e-4;
   ComponentBasedObject *baseObject;

   // check normalizeSum
   baseObject                            = getObject(hc, "NormalizeSumConnection");
   NormalizeBase *normalizeSumNormalizer = baseObject->getComponentByType<NormalizeBase>();
   FatalIf(!normalizeSumNormalizer, "%s has no normalizer.\n", baseObject->getDescription_c());
   float normalizeSumStrength    = normalizeSumNormalizer->getStrength();
   baseObject                    = getObject(hc, "NormalizeSumCheck");
   HyPerLayer *normalizeSumCheck = dynamic_cast<HyPerLayer *>(baseObject);
   auto *normalizeSumCheckData   = normalizeSumCheck->getComponentByType<PublisherComponent>();
   float normalizeSumValue       = normalizeSumCheckData->getLayerData()[0];
   FatalIf(fabsf(normalizeSumValue - normalizeSumStrength) >= tol, "Test failed.\n");

   // check normalizeL2
   baseObject                           = getObject(hc, "NormalizeL2Connection");
   NormalizeBase *normalizeL2Normalizer = baseObject->getComponentByType<NormalizeBase>();
   FatalIf(!normalizeL2Normalizer, "%s has no normalizer.\n", baseObject->getDescription_c());
   float normalizeL2Strength    = normalizeL2Normalizer->getStrength();
   baseObject                   = getObject(hc, "NormalizeL2Check");
   HyPerLayer *normalizeL2Check = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(normalizeL2Check == nullptr, "No layer named NormalizeL2Check.\n");
   auto *normalizeL2CheckData = normalizeL2Check->getComponentByType<PublisherComponent>();
   float normalizeL2Value     = sqrtf(normalizeL2CheckData->getLayerData()[0]);
   FatalIf(fabsf(normalizeL2Value - normalizeL2Strength) >= tol, "Test failed.\n");

   // check normalizeMax
   baseObject                            = getObject(hc, "NormalizeMaxConnection");
   NormalizeBase *normalizeMaxNormalizer = baseObject->getComponentByType<NormalizeBase>();
   FatalIf(!normalizeMaxNormalizer, "%s has no normalizer.\n", baseObject->getDescription_c());
   float normalizeMaxStrength    = normalizeMaxNormalizer->getStrength();
   baseObject                    = getObject(hc, "NormalizeMaxCheck");
   HyPerLayer *normalizeMaxCheck = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(normalizeMaxCheck == nullptr, "No layer named NormalizeMaxCheck.\n");
   auto *normalizeMaxCheckData = normalizeMaxCheck->getComponentByType<PublisherComponent>();
   float normalizeMaxValue     = -FLT_MAX;
   for (int k = 0; k < normalizeMaxCheck->getNumExtended(); k++) {
      float layerData = normalizeMaxCheckData->getLayerData()[k];
      if (normalizeMaxValue < layerData) {
         normalizeMaxValue = layerData;
      }
   }
   FatalIf(fabsf(normalizeMaxValue - normalizeMaxStrength) >= tol, "Test failed.\n");

   // check normalizeContrastZeroMean.
   baseObject = getObject(hc, "NormalizeContrastZeroMeanConnection");
   FatalIf(!baseObject, "No connection named \"NormalizeContrastZeroMeanConnection\".\n");
   NormalizeBase *normalizeContrastZeroMeanNormalizer =
         baseObject->getComponentByType<NormalizeBase>();
   FatalIf(
         !normalizeContrastZeroMeanNormalizer,
         "%s has no normalizer.\n",
         baseObject->getDescription_c());
   float normalizeContrastZeroMeanStrength = normalizeContrastZeroMeanNormalizer->getStrength();
   auto *connData                          = baseObject->getComponentByType<ConnectionData>();
   FatalIf(!connData, "%s has no ConnectionData component.\n", baseObject->getDescription_c());
   int numNeurons = connData->getPost()->getNumGlobalNeurons();

   // check normalizeConstransZeroMean mean
   baseObject = getObject(hc, "NormalizeContrastZeroMeanCheckMean");
   HyPerLayer *normalizeContrastZeroMeanCheckMean = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(
         !normalizeContrastZeroMeanCheckMean,
         "No layer named \"NormalizeContrastZeroMeanCheckMean\".\n");
   auto *checkDataMean =
         normalizeContrastZeroMeanCheckMean->getComponentByType<PublisherComponent>();
   float normalizeContrastZeroMeanValue = checkDataMean->getLayerData()[0] / numNeurons;
   FatalIf(fabsf(normalizeContrastZeroMeanValue) >= tol, "Test failed.\n");

   // check normalizeConstransZeroMean variance
   baseObject = getObject(hc, "NormalizeContrastZeroMeanCheckVariance");
   HyPerLayer *normalizeContrastZeroMeanCheckVariance = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(
         !normalizeContrastZeroMeanCheckVariance,
         "No layer named \"NormalizeContrastZeroMeanCheckVariance\".\n");
   auto *checkDataVariance =
         normalizeContrastZeroMeanCheckVariance->getComponentByType<PublisherComponent>();
   float normalizeContrastZeroMeanStDev = sqrtf(checkDataVariance->getLayerData()[0] / numNeurons);
   FatalIf(
         fabsf(normalizeContrastZeroMeanStDev - normalizeContrastZeroMeanStrength) >= tol,
         "Test failed.\n");

   return PV_SUCCESS;
}
