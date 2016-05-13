#ifndef BINNINGTESTLAYER_HPP_ 
#define BINNINGTESTLAYER_HPP_

#include <layers/BinningLayer.hpp>

namespace PV {

class BinningTestLayer: public PV::BinningLayer{
public:
	BinningTestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
};

BaseObject * createBinningTestLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
