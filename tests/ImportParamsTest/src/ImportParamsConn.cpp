#include "ImportParamsConn.hpp"

namespace PV {

ImportParamsConn::ImportParamsConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer){
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}

int ImportParamsConn::initialize_base()
{
   return PV_SUCCESS;
}

int ImportParamsConn::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer)
{
   KernelConn::initialize(name, hc, weightInitializer, weightNormalizer);

   PVParams * params = parent->parameters();
   //Test grabbed array value
   int size;
   const float * delayVals = params->arrayValues(name, "delay", &size);
   if(strcmp(name, "origConn") == 0){
      assert(size == 2);
      assert(delayVals[0] == 0);
      assert(delayVals[1] == 1);
      assert(strcmp(preLayerName, "orig") == 0);
   }
   else{
      assert(size == 3);
      assert(delayVals[0] == 3);
      assert(delayVals[1] == 4);
      assert(delayVals[2] == 5);
      assert(strcmp(preLayerName, "copy") == 0);
   }

   return PV_SUCCESS;
}

int ImportParamsConn::communicateInitInfo() {
   int status = KernelConn::communicateInitInfo();
   return status;
}

int ImportParamsConn::allocateDataStructures() {
   int status = KernelConn::allocateDataStructures();
   return status;
}

BaseObject * createImportParamsConn(const char * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   InitWeights * weightInitializer = getWeightInitializer(name, hc);
   NormalizeBase * weightNormalizer = getWeightNormalizer(name, hc);
   return new HyPerConn(name, hc, weightInitializer, weightNormalizer);
}


} /* namespace PV */
