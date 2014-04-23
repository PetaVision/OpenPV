/*
 * MLPOutputLayer.hpp
 * Author: slundquist
 */

#ifndef MLPOUTPUTLAYER_HPP_ 
#define MLPOUTPUTLAYER_HPP_ 
#include "SigmoidLayer.hpp"

namespace PV{

class MLPOutputLayer : public PV::SigmoidLayer{
public:
   MLPOutputLayer(const char * name, HyPerCol * hc);
   virtual ~MLPOutputLayer();
   virtual int updateState(double timef, double dt);
   virtual int allocateDataStructures();
protected:
   MLPOutputLayer();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_LocalTarget(enum ParamsIOFlag ioFlag);
private:
   bool localTarget;
   pvdata_t * classBuffer;
   int initialize_base();

};

}
#endif 
