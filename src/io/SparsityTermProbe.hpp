/*
 * SparsityTermProbe.hpp
 *
 *  Created on: Nov 18, 2010
 *      Author: pschultz
 */

#ifndef SPARSITYTERMPROBE_HPP_
#define SPARSITYTERMPROBE_HPP_

#include "LayerFunctionProbe.hpp"
#include "SparsityTermFunction.hpp"

namespace PV {

class SparsityTermProbe : public LayerFunctionProbe {
public:
   SparsityTermProbe(HyPerLayer * layer, const char * msg);
   SparsityTermProbe(const char * filename, HyPerLayer * layer, const char * msg);
   virtual ~SparsityTermProbe();
   virtual int outputState(float timef);

protected:
   int initSparsityTermProbe(const char * filename, HyPerLayer * layer, const char * msg);
};

}  // end namespace PV

#endif /* SPARSITYTERMPROBE_HPP_ */
