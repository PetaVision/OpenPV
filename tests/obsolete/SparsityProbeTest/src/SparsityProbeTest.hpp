/*
 * TriggerTestLayer.hpp
 * Author: slundquist
 */

#ifndef SPARSITYPROBETEST_HPP_ 
#define SPARSITYPROBETEST_HPP_
#include <io/SparsityLayerProbe.hpp>

namespace PV{

class SparsityProbeTest : public PV::SparsityLayerProbe{
public:
   SparsityProbeTest(const char * name, HyPerCol * hc);
   virtual int outputState(double time);
};

}
#endif /* IMAGETESTPROBE_HPP */
