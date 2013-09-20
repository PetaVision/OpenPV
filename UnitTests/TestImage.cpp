/*
 * TestImage.cpp
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#include "TestImage.hpp"

namespace PV {

TestImage::TestImage(const char * name, HyPerCol * hc, pvdata_t val)
          : Image(name, hc)
{
   initialize_data(&loc);
   setData(val);
}

bool TestImage::updateImage(float time, float dt)
{
   return false;
}

int TestImage::setData(pvdata_t val)
{
   const int N = (loc.nx + 2*loc.nPad) * (loc.ny + 2*loc.nPad) * loc.nBands;
   for (int kext = 0; kext < N; kext++) {
      data[kext] = val;
   }
   return 0;
}

} // namespace PV
