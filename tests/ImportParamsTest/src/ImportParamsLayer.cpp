#include "ImportParamsLayer.hpp"

namespace PV {

ImportParamsLayer::ImportParamsLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int ImportParamsLayer::initialize_base()
{
   return PV_SUCCESS;
}

int ImportParamsLayer::initialize(const char * name, HyPerCol * hc)
{
   ANNLayer::initialize(name, hc);

   PVParams * params = parent->parameters();
   if(strcmp(name, "orig") == 0){
      //Test grabbed value
      assert(params->value(name, "nxScale") == 1);
      //Test grabbed filename
      assert(strcmp(params->stringValue(name, "Vfilename"), "input/a0.pvp") == 0);
   }
   else{
      //Test overwritten value
      assert(params->value(name, "nxScale") == 2);
      //Test overwritten filename
      assert(strcmp(params->stringValue(name, "Vfilename"), "input/a1.pvp") == 0);
   }

   return PV_SUCCESS;
}

int ImportParamsLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   return status;
}

int ImportParamsLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();
   return status;
}

BaseObject * createImportParamsLayer(char const * name, HyPerCol * hc) {
   return hc ? new ImportParamsLayer(name, hc) : NULL;
}

} /* namespace PV */
