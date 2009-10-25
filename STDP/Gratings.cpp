/*
 * Gratings.cpp
 *
 *  Created on: Oct 23, 2009
 *      Author: travel
 */

#include "Gratings.hpp"
#include <src/include/pv_common.h>  // for PI

namespace PV {

Gratings::Gratings(const char * name, HyPerCol * hc) : Image(name, hc)
{
   initialize_data(&loc);
   updateImage(0.0, 0.0);
}

Gratings::~Gratings() {
}

bool Gratings::updateImage(float time_step, float dt)
{
   // extended frame
   const int nx = loc.nx + 2*loc.nPad;
   const int ny = loc.ny + 2*loc.nPad;
   const int sx = 1;
   const int sy = sx * nx;

   const float kx  = 2.0*PI/4.0;   // wavenumber
   const float phi = PI;           // phase

   for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
         float x = (float) ix;
         data[ix*sx + iy*sy] = sin(kx * x + phi);
      }
   }

   float * buf = data;

   return false;  // not updating dynamically yet
}

} // namespace PV
