/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <columns/HyPerCol.hpp>
#include <components/ActivityComponent.hpp>
#include <layers/HyPerLayer.hpp>

using namespace PV;

int checkAnswer(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, &checkAnswer);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkAnswer(HyPerCol *hc, int argc, char *argv[]) {
   char const *layerName = "Output";
   auto *layer           = dynamic_cast<HyPerLayer *>(hc->getObjectFromName(layerName));
   FatalIf(!layer, "No layer named \"%s\".\n", layerName);
   auto *activity = layer->getComponentByType<ActivityComponent>();
   FatalIf(
         !activity,
         "%s does not have an ActivityComponent.\n",
         layer->getDescription_c());
   FatalIf(
         activity->getNumExtendedAcrossBatch() != 1,
         "%s has %d neurons instead of expected number 1.\n",
         activity->getNumExtendedAcrossBatch());
   float A = activity->getActivity()[0];
   float correct = 73.0f;
   FatalIf(
         A != correct,
         "%s has incorrect activity: expected %f, observed %f (discrepancy %g).\n",
         layer->getDescription_c(),
         double(correct),
         double(A),
         double(A - correct));
   return PV_SUCCESS;
}
