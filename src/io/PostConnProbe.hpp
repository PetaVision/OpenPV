/*
 * PostConnProbe.hpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#ifndef POSTCONNPROBE_HPP_
#define POSTCONNPROBE_HPP_

#include "ConnectionProbe.hpp"

namespace PV {

class PostConnProbe: public PV::ConnectionProbe {
public:
   PostConnProbe(int kPost);
   PostConnProbe(int kxPost, int kyPost, int kfPost);
   PostConnProbe(const char * filename, int kPost);
   PostConnProbe(const char * filename, int kxPost, int kyPost, int kfPost);

   virtual int outputState(float time, HyPerConn * c);

protected:
   int kPost;   // index of post-synaptic neuron
   int kxPost, kyPost, kfPost;
};

}

#endif /* POSTCONNPROBE_HPP_ */
