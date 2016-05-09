#ifndef IMPORTPARAMSLAYER_HPP_ 
#define IMPORTPARAMSLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ImportParamsLayer: public PV::ANNLayer {
public:
	ImportParamsLayer(const char* name, HyPerCol * hc);
   virtual int communicateInitInfo();
	virtual int allocateDataStructures();

private:
    int initialize(const char * name, HyPerCol * hc);
    int initialize_base();
};

BaseObject * createImportParamsLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
