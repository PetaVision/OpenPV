/*
 * LinearPostConnProbe.hpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#ifndef LINEARPOSTCONNPROBE_HPP_
#define LINEARPOSTCONNPROBE_HPP_

#include "src/io/PostConnProbe.hpp"
#include "src/io/LinearActivityProbe.hpp"

namespace PV {

class LinearPostConnProbe: public PV::PostConnProbe {
public:
   LinearPostConnProbe(PVDimType dim, int loc, int f);
   LinearPostConnProbe(int kPost);
   LinearPostConnProbe(const char * filename, int kPost);

   virtual int outputState(float time, HyPerConn * c);

protected:
   int kPost;  // index of post-synaptic neuron
   HyPerCol * parent;
   PVDimType dim;
   int loc;
   int f;
};

}

#endif /* LINEARPOSTCONNPROBE_HPP_ */
