/*
 * L2ConnProbe.hpp
 *
 *  Created on: July 24th, 2015
 *      Author: Kendall Stewart
 */

#ifndef L2CONNPROBE_HPP_
#define L2CONNPROBE_HPP_

#include "KernelProbe.hpp"

namespace PV {

class L2ConnProbe : public KernelProbe {

   // Methods
  public:
   L2ConnProbe(const char *probename, HyPerCol *hc);
   virtual ~L2ConnProbe();
   virtual Response::Status outputState(double timef) override;

  protected:
   L2ConnProbe();

}; // end of class L2ConnProbe block

} // end of namespace PV block

#endif /* L2CONNPROBE_HPP_ */
