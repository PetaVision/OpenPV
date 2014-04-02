/*
 * SparsityProbeTest.cpp
 * Author: slundquist
 */

#include "SparsityProbeTest.hpp"
#include <assert.h>

namespace PV {

SparsityProbeTest::SparsityProbeTest(const char * name, HyPerCol * hc)
{
   SparsityLayerProbe::initLayerProbe(name , hc);
}

int SparsityProbeTest::outputState(double time){
   //Update probe
   SparsityLayerProbe::outputState(time);
   //Sparsity should be .5
   assert(getSparsity() - .5 < .0001);
}

}
