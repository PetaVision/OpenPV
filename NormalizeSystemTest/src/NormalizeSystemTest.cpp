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
   baseConn = hc->getConnFromName("normalizeSum connection");
   HyPerConn * normalizeSumConn = dynamic_cast<HyPerConn *>(baseConn);
   assert(normalizeSumConn);
   NormalizeBase * normalizeSumNormalizer = normalizeSumConn->getNormalizer();
   assert(normalizeSumNormalizer);
   float normalizeSumStrength = normalizeSumNormalizer->getStrength();
   HyPerLayer * normalizeSumCheck = hc->getLayerFromName("normalizeSum check");
   float normalizeSumValue = normalizeSumCheck->getLayerData()[0];
   assert(fabsf(normalizeSumValue - normalizeSumStrength)<tol);

   // check normalizeL2
   baseConn = hc->getConnFromName("normalizeL2 connection");
   HyPerConn * normalizeL2Conn = dynamic_cast<HyPerConn *>(baseConn);
   assert(normalizeL2Conn);
   NormalizeBase * normalizeL2Normalizer = normalizeL2Conn->getNormalizer();
   assert(normalizeL2Normalizer);
   float normalizeL2Strength = normalizeL2Normalizer->getStrength();
   HyPerLayer * normalizeL2Check = hc->getLayerFromName("normalizeL2 check");
   assert(normalizeL2Check);
   float normalizeL2Value = sqrtf(normalizeL2Check->getLayerData()[0]);
   assert(fabsf(normalizeL2Value - normalizeL2Strength)<tol);

   // check normalizeMax
   baseConn = hc->getConnFromName("normalizeMax connection");
   HyPerConn * normalizeMaxConn = dynamic_cast<HyPerConn *>(baseConn);
   assert(normalizeMaxConn);
   NormalizeBase * normalizeMaxNormalizer = normalizeMaxConn->getNormalizer();
   assert(normalizeMaxNormalizer);
   float normalizeMaxStrength = normalizeMaxNormalizer->getStrength();
   HyPerLayer * normalizeMaxCheck = hc->getLayerFromName("normalizeMax check");
   assert(normalizeMaxCheck);
   float normalizeMaxValue = -FLT_MAX;
   for (int k=0; k<normalizeMaxCheck->getNumExtended(); k++) {
      pvadata_t layerData = normalizeMaxCheck->getLayerData()[k];
      if (normalizeMaxValue < layerData) {normalizeMaxValue = layerData;}
   }
   assert(fabsf(normalizeMaxValue - normalizeMaxStrength)<tol);

   // check normalizeContrastZeroMean.
   baseConn = hc->getConnFromName("normalizeContrastZeroMean connection");
   HyPerConn * normalizeContrastZeroMeanConn = dynamic_cast<HyPerConn *>(baseConn);
   assert(normalizeContrastZeroMeanConn);
   NormalizeBase * normalizeContrastZeroMeanNormalizer = normalizeContrastZeroMeanConn->getNormalizer();
   assert(normalizeContrastZeroMeanNormalizer);
   float normalizeContrastZeroMeanStrength = normalizeContrastZeroMeanNormalizer->getStrength();
   int numNeurons = normalizeContrastZeroMeanConn->postSynapticLayer()->getNumGlobalNeurons();
   HyPerLayer * normalizeContrastZeroMeanCheckMean = hc->getLayerFromName("normalizeContrastZeroMean check mean");
   assert(normalizeContrastZeroMeanCheckMean);
   float normalizeContrastZeroMeanValue = normalizeContrastZeroMeanCheckMean->getLayerData()[0]/numNeurons;
   assert(fabsf(normalizeContrastZeroMeanValue)<tol);
   HyPerLayer * normalizeContrastZeroMeanCheckVariance = hc->getLayerFromName("normalizeContrastZeroMean check variance");
   assert(normalizeContrastZeroMeanCheckVariance);
   float normalizeContrastZeroMeanStDev = sqrtf(normalizeContrastZeroMeanCheckVariance->getLayerData()[0]/numNeurons);
   assert(fabsf(normalizeContrastZeroMeanStDev - normalizeContrastZeroMeanStrength)<tol);

   return PV_SUCCESS;
}
