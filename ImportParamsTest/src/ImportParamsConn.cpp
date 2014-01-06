#include "ImportParamsConn.hpp"

namespace PV {

ImportParamsConn::ImportParamsConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char* post_layer_name){
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name);
}

int ImportParamsConn::initialize_base()
{
   return PV_SUCCESS;
}

int ImportParamsConn::initialize(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name)
{
   KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, NULL, NULL);

   PVParams * params = parent->parameters();
   //Test grabbed array value
   int size;
   const float * delayVals = params->arrayValues(name, "delay", &size);
   if(strcmp(name, "origConn") == 0){
      assert(size == 2);
      assert(delayVals[0] == 0);
      assert(delayVals[1] == 1);
   }
   else{
      assert(size == 3);
      assert(delayVals[0] == 3);
      assert(delayVals[1] == 4);
      assert(delayVals[2] == 5);
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

} /* namespace PV */
