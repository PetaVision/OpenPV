//============================================================================
// Name        : NormalizeSystemTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <columns/buildandrun.hpp>
#include "NormalizeL3.hpp"

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("normalizeL3", Factory::create<NormalizeL3>);
   int status = buildandrun(&pv_initObj, NULL, customexit);
   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   float tol = 1e-5;

   BaseConnection * baseConn;
   baseConn = hc->getConnFromName("normalizeL3 connection");
   HyPerConn * normalizeL3Conn = dynamic_cast<HyPerConn *>(baseConn);
   pvErrorIf(!(normalizeL3Conn), "Test failed.\n");
   NormalizeBase * normalizeL3Normalizer = normalizeL3Conn->getNormalizer();
   pvErrorIf(!(normalizeL3Normalizer), "Test failed.\n");
   float normalizeL3Strength = normalizeL3Normalizer->getStrength();
   float correctValue = powf(normalizeL3Strength, 3.0f);
   HyPerLayer * normalizeL3Check = hc->getLayerFromName("normalizeL3 check");
   float normalizeL3Value = normalizeL3Check->getLayerData()[0];
   pvErrorIf(!(fabsf(normalizeL3Value - correctValue)<tol), "Test failed.\n");

   return PV_SUCCESS;
}
