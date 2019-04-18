//============================================================================
// Name        : NormalizeSystemTest.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "NormalizeL3.hpp"
#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("normalizeL3", Factory::create<NormalizeL3>);
   int status = buildandrun(&pv_initObj, NULL, customexit);
   return status;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   float tol = 1e-5;

   Observer *baseObject;

   baseObject      = hc->getObjectFromName("NormalizeL3Connection");
   auto *hyperconn = dynamic_cast<HyPerConn *>(baseObject);
   FatalIf(hyperconn == nullptr, "Connection \"NormalizeL3Connection\" does not exist.\n");
   NormalizeBase *normalizeL3Normalizer = hyperconn->getComponentByType<NormalizeBase>();
   FatalIf(normalizeL3Normalizer == nullptr, "NormalizeL3Connection has no normalizer.\n");
   float normalizeL3Strength = normalizeL3Normalizer->getStrength();
   float correctValue        = powf(normalizeL3Strength, 3.0f);

   baseObject                   = hc->getObjectFromName("NormalizeL3Check");
   HyPerLayer *normalizeL3Check = dynamic_cast<HyPerLayer *>(baseObject);
   FatalIf(normalizeL3Check == nullptr, "Layer \"NormalizeL3Check\" does not exist.\n");
   float normalizeL3Value = normalizeL3Check->getLayerData()[0];
   FatalIf(
         fabsf(normalizeL3Value - correctValue) >= tol,
         "Result %f differs from %f by more than allowed tolerance.\n",
         (double)normalizeL3Value,
         (double)correctValue);

   return PV_SUCCESS;
}
