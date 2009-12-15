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

   virtual int outputState(float time, HyPerConn * c);

   void setOutputIndices(bool flag)   {outputIndices = flag;}

protected:
   int kPost;   // index of post-synaptic neuron
   int kxPost, kyPost, kfPost;
   bool outputIndices;
};

}

#endif /* POSTCONNPROBE_HPP_ */
