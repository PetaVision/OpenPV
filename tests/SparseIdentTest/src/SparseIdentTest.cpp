/*
 * SparseIdentTest.cpp
 *
 */

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <layers/HyPerLayer.hpp>

using namespace PV;

HyPerLayer *getLayerFromName(std::string const &name, HyPerCol *hc);
void compare(char const *inputname, char const *outputname, HyPerCol *hc);

int main(int argc, char *argv[]) {
   auto *pv_init_obj = new PV_Init(&argc, &argv, false /* do not allow unrecognized args */);
   auto *hc          = new HyPerCol(pv_init_obj);

   HyPerLayer *inputLayer = getLayerFromName(std::string("Input"), hc);
   pvAssert(inputLayer); // Was tested in getLayerFromName().

   auto *geometry = inputLayer->getComponentByType<LayerGeometry>();
   FatalIf(
         geometry == nullptr,
         "%s does not have a LayerGeometry component.\n",
         inputLayer->getDescription_c());
   int margin = 2;
   geometry->requireMarginWidth(margin, 'x');
   geometry->requireMarginWidth(margin, 'y');
   PVHalo const &halo = inputLayer->getLayerLoc()->halo;
   FatalIf(
         halo.lt != margin or halo.rt != margin or halo.dn != margin or halo.up != margin,
         "%s failed to set halo.\n",
         inputLayer->getDescription_c());

   int status = hc->run();

   compare("Input", "OutputIdent", hc);
   compare("Input", "OutputRescale", hc);

   delete hc;
   delete pv_init_obj;
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

HyPerLayer *getLayerFromName(std::string const &name, HyPerCol *hc) {
   auto *object = hc->getObjectFromName(name);
   FatalIf(object == nullptr, "There is no object named \"%s\" in the column.\n", name.c_str());
   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(object);
   FatalIf(layer == nullptr, "%s is not a layer in the column.\n", layer->getDescription_c());
   return layer;
}

void compare(char const *inputname, char const *outputname, HyPerCol *hc) {
   HyPerLayer *inputLayer  = getLayerFromName(std::string(inputname), hc);
   HyPerLayer *outputLayer = getLayerFromName(std::string(outputname), hc);
   pvAssert(inputLayer and outputLayer); // getLayerFromName() checks result for non-null.

   auto *inputActivityComponent  = inputLayer->getComponentByType<ActivityComponent>();
   auto *inputActivityBuffer     = inputActivityComponent->getComponentByType<ActivityBuffer>();
   auto *outputActivityComponent = outputLayer->getComponentByType<ActivityComponent>();
   auto *outputActivityBuffer    = outputActivityComponent->getComponentByType<ActivityBuffer>();
   PVLayerLoc const *inLoc       = inputActivityBuffer->getLayerLoc();
   PVLayerLoc const *outLoc      = outputActivityBuffer->getLayerLoc();

   ComponentBuffer::checkDimensionsEqual(inputActivityBuffer, outputActivityBuffer);

   auto *inputPublisher  = inputLayer->getComponentByType<PublisherComponent>();
   auto *outputPublisher = outputLayer->getComponentByType<PublisherComponent>();

   int nbatch         = inLoc->nbatch;
   int numNeurons     = inLoc->nx * inLoc->ny * inLoc->nf;
   int numInExtended  = inputPublisher->getNumExtended();
   int numOutExtended = outputPublisher->getNumExtended();

   float const *inData  = inputPublisher->getLayerData();
   float const *outData = outputPublisher->getLayerData();
   bool failed          = false;
   for (int b = 0; b < nbatch; b++) {
      for (int k = 0; k < numNeurons; k++) {
         int kInExt = kIndexExtended(
               k,
               inLoc->nx,
               inLoc->ny,
               inLoc->nf,
               inLoc->halo.lt,
               inLoc->halo.rt,
               inLoc->halo.dn,
               inLoc->halo.up);
         int kOutExt = kIndexExtended(
               k,
               outLoc->nx,
               outLoc->ny,
               outLoc->nf,
               outLoc->halo.lt,
               outLoc->halo.rt,
               outLoc->halo.dn,
               outLoc->halo.up);
         float inValue  = inData[b * numInExtended + kInExt];
         float outValue = outData[b * numOutExtended + kOutExt];

         if (inValue != outValue) {
            ErrorLog().printf(
                  "b=%d, k=%d, %s=%f, %s=%f.\n",
                  b,
                  k,
                  inputLayer->getDescription_c(),
                  (double)inValue,
                  outputLayer->getDescription_c(),
                  (double)outValue);
            failed = true;
         }
      }
   }
   FatalIf(
         failed,
         "%s and %s do not agree.\n",
         inputLayer->getDescription_c(),
         outputLayer->getDescription_c());
}
