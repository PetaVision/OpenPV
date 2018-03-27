/*
 * ANNWhitenedLayer.hpp
 *
 *  Created on: Feb 15, 2013
 *      Author: garkenyon
 */

#ifndef ANNWHITENEDLAYER_HPP_
#define ANNWHITENEDLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ANNWhitenedLayer : public PV::ANNLayer {
  public:
   ANNWhitenedLayer(const char *name, HyPerCol *hc);
   virtual ~ANNWhitenedLayer();

  protected:
   ANNWhitenedLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status updateState(double time, double dt) override;

  private:
   int initialize_base();
}; // class ANNWhitenedLayer

} /* namespace PV */
#endif /* ANNWHITENEDLAYER_HPP_ */
