/*
 * Retina.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef RETINA_HPP_
#define RETINA_HPP_

#include "layers/HyPerLayer.hpp"

namespace PV {

class Retina : public HyPerLayer {
  public:
   Retina(const char *name, PVParams *params, Communicator *comm);
   virtual ~Retina();

  protected:
   Retina();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual ActivityComponent *createActivityComponent() override;

}; // class Retina

} // namespace PV

#endif /* RETINA_HPP_ */
