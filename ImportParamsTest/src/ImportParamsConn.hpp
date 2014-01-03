#ifndef IMPORTPARAMSCONN_HPP_ 
#define IMPORTPARAMSCONN_HPP_

#include <connections/KernelConn.hpp>

namespace PV {

class ImportParamsConn: public PV::KernelConn{
public:
	ImportParamsConn(const char* name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name);
   virtual int communicateInitInfo();
	virtual int allocateDataStructures();

private:
   int initialize(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name);
   int initialize_base();
};

} /* namespace PV */
#endif
