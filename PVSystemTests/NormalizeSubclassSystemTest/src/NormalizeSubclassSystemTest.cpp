//============================================================================
// Name        : NormalizeSystemTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <columns/buildandrun.hpp>
#include "CustomGroupHandler.hpp"

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   ParamGroupHandler * normalizeL3GroupHandler = new CustomGroupHandler();
   int status = buildandrun(argc, argv, NULL, customexit, &normalizeL3GroupHandler, 1/*numGroupHandlers*/);
   delete normalizeL3GroupHandler;
   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   float tol = 1e-5;

   BaseConnection * baseConn;
   baseConn = hc->getConnFromName("normalizeL3 connection");
   HyPerConn * normalizeL3Conn = dynamic_cast<HyPerConn *>(baseConn);
   assert(normalizeL3Conn);
   NormalizeBase * normalizeL3Normalizer = normalizeL3Conn->getNormalizer();
   assert(normalizeL3Normalizer);
   float normalizeL3Strength = normalizeL3Normalizer->getStrength();
   float correctValue = powf(normalizeL3Strength, 3.0f);
   HyPerLayer * normalizeL3Check = hc->getLayerFromName("normalizeL3 check");
   float normalizeL3Value = normalizeL3Check->getLayerData()[0];
   assert(fabsf(normalizeL3Value - correctValue)<tol);

   return PV_SUCCESS;
}
