/*
 * TestImage.hpp
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#ifndef TESTIMAGE_HPP_
#define TESTIMAGE_HPP_

#include "../src/layers/Image.hpp"

namespace PV {

class TestImage : public Image {
public:
   TestImage(const char * name, HyPerCol * hc, pvdata_t val);

   virtual bool updateImage(float time, float dt);

   int setData(pvdata_t val);

   pvdata_t * getData()       {return data;}

protected:

};

}

#endif /* TESTIMAGE_HPP_ */
