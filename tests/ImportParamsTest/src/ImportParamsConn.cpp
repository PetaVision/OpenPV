#include "ImportParamsConn.hpp"

namespace PV {

ImportParamsConn::ImportParamsConn(const char * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

int ImportParamsConn::initialize_base()
{
   return PV_SUCCESS;
}

int ImportParamsConn::initialize(const char * name, HyPerCol * hc)
{
   HyPerConn::initialize(name, hc);

   PVParams * params = parent->parameters();
   //Test grabbed array value
   int size;
   const float * delayVals = params->arrayValues(name, "delay", &size);
   if(strcmp(name, "origConn") == 0){
      pvErrorIf(!(size == 2), "Test failed.\n");
      pvErrorIf(!(delayVals[0] == 0), "Test failed.\n");
      pvErrorIf(!(delayVals[1] == 1), "Test failed.\n");
      pvErrorIf(!(strcmp(preLayerName, "orig") == 0), "Test failed.\n");
   }
   else{
      pvErrorIf(!(size == 3), "Test failed.\n");
      pvErrorIf(!(delayVals[0] == 3), "Test failed.\n");
      pvErrorIf(!(delayVals[1] == 4), "Test failed.\n");
      pvErrorIf(!(delayVals[2] == 5), "Test failed.\n");
      pvErrorIf(!(strcmp(preLayerName, "copy") == 0), "Test failed.\n");
   }

   return PV_SUCCESS;
}

int ImportParamsConn::communicateInitInfo() {
   int status = HyPerConn::communicateInitInfo();
   return status;
}

int ImportParamsConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   return status;
}


} /* namespace PV */
