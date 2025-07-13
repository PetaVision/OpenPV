/*
 * ANNWhitenedLayer.hpp
 *
 *  Created on: Feb 15, 2013
 *      Author: garkenyon
 */

#ifndef ANNWHITENEDLAYER_HPP_
#define ANNWHITENEDLAYER_HPP_

// ANNWhitenedLayer was deprecated on Aug 15, 2018.

#include "layers/ANNLayer.hpp"

namespace PV {

class ANNWhitenedLayer : public ANNLayer {
  public:
   ANNWhitenedLayer(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~ANNWhitenedLayer();

  protected:
   ANNWhitenedLayer();
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual Response::Status updateState(double time, double dt) override;

  private:
   int initialize_base();
}; // class ANNWhitenedLayer

} /* namespace PV */
#endif /* ANNWHITENEDLAYER_HPP_ */
