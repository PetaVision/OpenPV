#include "ImportParamsLayer.hpp"

namespace PV {

ImportParamsLayer::ImportParamsLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}

ImportParamsLayer::ImportParamsLayer(const char * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc, MAX_CHANNELS);
}

int ImportParamsLayer::initialize_base()
{
   return PV_SUCCESS;
}

int ImportParamsLayer::initialize(const char * name, HyPerCol * hc, int num_channels)
{
   ANNLayer::initialize(name, hc, num_channels);

   PVParams * params = parent->parameters();
   if(strcmp(name, "orig") == 0){
      //Test grabbed value
      assert(params->value(name, "nxScale") == 1);
      //Test grabbed string
      assert(strcmp(params->stringValue(name, "InitVType"), "ZeroV") == 0);
      assert(strcmp(params->stringValue(name, "Vfilename"), "/nh/compneuro/Data/a.pvp") == 0);
   }
   else{
      //Test overwritten value
      assert(params->value(name, "nxScale") == 2);
      //Test overwritten string
      assert(strcmp(params->stringValue(name, "InitVType"), "ConstantV") == 0);
      assert(strcmp(params->stringValue(name, "Vfilename"), "/nh/compneuro/Data/test.pvp") == 0);
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

} /* namespace PV */
