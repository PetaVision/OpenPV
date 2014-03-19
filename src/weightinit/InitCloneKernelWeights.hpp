/*
 * InitCloneKernelWeights.hpp
 *
 *  Created on: Oct 2, 2011
 *      Author: pschultz
 */

#ifndef INITCLONEKERNELWEIGHTS_HPP_
#define INITCLONEKERNELWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitCloneKernelWeights: public PV::InitWeights {
public:
   InitCloneKernelWeights();
   virtual ~InitCloneKernelWeights();
   virtual int calcWeights(/*PVPatch * patch*/ pvdata_t * dataStart, int patchIndex, int arborId);
protected:
   virtual int initialize_base();
}; // end class InitCloneKernelWeights

}  /* namespace PV */


#endif /* INITCLONEKERNELWEIGHTS_HPP_ */
