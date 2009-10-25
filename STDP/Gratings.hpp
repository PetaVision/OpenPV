/*
 * Gratings.hpp
 *
 *  Created on: Oct 23, 2009
 *      Author: travel
 */

#ifndef GRATINGS_HPP_
#define GRATINGS_HPP_

#include <src/layers/Image.hpp>

namespace PV {

class Gratings : public Image  {
public:
   Gratings(const char * name, HyPerCol * hc);
   virtual ~Gratings();

   virtual bool updateImage(float time_step, float dt);

};

}

#endif /* GRATINGS_HPP_ */
