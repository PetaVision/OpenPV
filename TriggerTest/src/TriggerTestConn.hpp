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
   TriggerTestConn(const char * name, HyPerCol * hc);
   int virtual updateStateWrapper (double time, double dt);
};

}
#endif /* IMAGETESTPROBE_HPP */
