/*
 * ISTALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef ISTALAYER_HPP__
#define ISTALAYER_HPP__

#include "HyPerLayer.hpp"

namespace PV {

class ISTALayer : public HyPerLayer {
  public:
   ISTALayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~ISTALayer();

  protected:
   ISTALayer() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif /* ISTALAYER_HPP_ */
