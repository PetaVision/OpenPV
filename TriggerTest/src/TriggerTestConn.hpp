/*
 * TriggerTestConn.hpp
 * Author: slundquist
 */

#ifndef TRIGGERTESTCONN_HPP_
#define TRIGGERTESTCONN_HPP_ 
#include <connections/HyPerConn.hpp>

namespace PV{

class TriggerTestConn: public PV::HyPerConn{
public:
   TriggerTestConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name);
   int virtual updateStateWrapper (double time, double dt);
};

}
#endif /* IMAGETESTPROBE_HPP */
