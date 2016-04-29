#ifndef IMPORTPARAMSCONN_HPP_ 
#define IMPORTPARAMSCONN_HPP_

#include <connections/KernelConn.hpp>

namespace PV {

class ImportParamsConn: public PV::KernelConn{
public:
   ImportParamsConn(const char* name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

private:
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   int initialize_base();
};

BaseObject * createImportParamsConn(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
