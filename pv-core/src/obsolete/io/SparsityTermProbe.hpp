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
   SparsityTermProbe(const char * probeName, HyPerCol * hc);
   virtual ~SparsityTermProbe();
   virtual int outputState(double timef);

protected:
   SparsityTermProbe();
   int initSparsityTermProbe(const char * probeName, HyPerCol * hc);
   virtual void initFunction();

private:
   int initSparsityTermProbe_base() { return PV_SUCCESS; }
};

}  // end namespace PV

#endif /* SPARSITYTERMPROBE_HPP_ */
