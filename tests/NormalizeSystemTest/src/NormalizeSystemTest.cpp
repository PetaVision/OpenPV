//============================================================================
// Name        : NormalizeSystemTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <columns/buildandrun.hpp>
#include <normalizers/NormalizeBase.hpp>

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   return buildandrun(argc, argv, NULL, customexit);
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   float tol = 1e-5;

   // check normalizeSum
   BaseConnection * baseConn;
   baseConn = hc->getConnFromName("NormalizeSumConnection");
   HyPerConn * normalizeSumConn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(normalizeSumConn), "Test failed.\n");
   NormalizeBase * normalizeSumNormalizer = normalizeSumConn->getNormalizer();
   pvErrorIf(!(normalizeSumNormalizer), "Test failed.\n");
   float normalizeSumStrength = normalizeSumNormalizer->getStrength();
   HyPerLayer * normalizeSumCheck = hc->getLayerFromName("NormalizeSumCheck");
   float normalizeSumValue = normalizeSumCheck->getLayerData()[0];
   pvErrorIf(!(fabsf(normalizeSumValue - normalizeSumStrength)<tol), "Test failed.\n");

   // check normalizeL2
   baseConn = hc->getConnFromName("NormalizeL2Connection");
   HyPerConn * normalizeL2Conn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(normalizeL2Conn), "Test failed.\n");
   NormalizeBase * normalizeL2Normalizer = normalizeL2Conn->getNormalizer();
   pvErrorIf(!(normalizeL2Normalizer), "Test failed.\n");
   float normalizeL2Strength = normalizeL2Normalizer->getStrength();
   HyPerLayer * normalizeL2Check = hc->getLayerFromName("NormalizeL2Check");
   pvErrorIf(!(normalizeL2Check), "Test failed.\n");
   float normalizeL2Value = sqrtf(normalizeL2Check->getLayerData()[0]);
   pvErrorIf(!(fabsf(normalizeL2Value - normalizeL2Strength)<tol), "Test failed.\n");

   // check normalizeMax
   baseConn = hc->getConnFromName("NormalizeMaxConnection");
   HyPerConn * normalizeMaxConn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(normalizeMaxConn), "Test failed.\n");
   NormalizeBase * normalizeMaxNormalizer = normalizeMaxConn->getNormalizer();
   pvErrorIf(!(normalizeMaxNormalizer), "Test failed.\n");
   float normalizeMaxStrength = normalizeMaxNormalizer->getStrength();
   HyPerLayer * normalizeMaxCheck = hc->getLayerFromName("NormalizeMaxCheck");
   pvErrorIf(!(normalizeMaxCheck), "Test failed.\n");
   float normalizeMaxValue = -FLT_MAX;
   for (int k=0; k<normalizeMaxCheck->getNumExtended(); k++) {
      pvadata_t layerData = normalizeMaxCheck->getLayerData()[k];
      if (normalizeMaxValue < layerData) {normalizeMaxValue = layerData;}
   }
   pvErrorIf(!(fabsf(normalizeMaxValue - normalizeMaxStrength)<tol), "Test failed.\n");

   // check normalizeContrastZeroMean.
   baseConn = hc->getConnFromName("NormalizeContrastZeroMeanConnection");
   HyPerConn * normalizeContrastZeroMeanConn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(normalizeContrastZeroMeanConn), "Test failed.\n");
   NormalizeBase * normalizeContrastZeroMeanNormalizer = normalizeContrastZeroMeanConn->getNormalizer();
   pvErrorIf(!(normalizeContrastZeroMeanNormalizer), "Test failed.\n");
   float normalizeContrastZeroMeanStrength = normalizeContrastZeroMeanNormalizer->getStrength();
   int numNeurons = normalizeContrastZeroMeanConn->postSynapticLayer()->getNumGlobalNeurons();
   HyPerLayer * normalizeContrastZeroMeanCheckMean = hc->getLayerFromName("NormalizeContrastZeroMeanCheckMean");
   pvErrorIf(!(normalizeContrastZeroMeanCheckMean), "Test failed.\n");
   float normalizeContrastZeroMeanValue = normalizeContrastZeroMeanCheckMean->getLayerData()[0]/numNeurons;
   pvErrorIf(!(fabsf(normalizeContrastZeroMeanValue)<tol), "Test failed.\n");
   HyPerLayer * normalizeContrastZeroMeanCheckVariance = hc->getLayerFromName("NormalizeContrastZeroMeanCheckVariance");
   pvErrorIf(!(normalizeContrastZeroMeanCheckVariance), "Test failed.\n");
   float normalizeContrastZeroMeanStDev = sqrtf(normalizeContrastZeroMeanCheckVariance->getLayerData()[0]/numNeurons);
   pvErrorIf(!(fabsf(normalizeContrastZeroMeanStDev - normalizeContrastZeroMeanStrength)<tol), "Test failed.\n");

   return PV_SUCCESS;
}
