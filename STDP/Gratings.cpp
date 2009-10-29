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

   PVParams * params = hc->parameters();
   float freq = params->value(name, "burstFreq", 40.0);

   period = 25;
   if (freq > 0.0) {
	   period = 1000/freq;
   }

   float dt = hc->getDeltaTime();

   calcPhase(0.0, dt);  // time==0.0 initializes random phase, don't delete
   calcPhase(hc->simulationTime(), dt);

   updateImage(0.0, 0.0);
}

Gratings::~Gratings() {
}

bool Gratings::updateImage(float time, float dt)
{
   // extended frame
   const int nx = loc.nx + 2*loc.nPad;
   const int ny = loc.ny + 2*loc.nPad;
   const int sx = 1;
   const int sy = sx * nx;

   const float kx  = 2.0*PI/4.0;   // wavenumber
   const float phi = calcPhase(time, dt);

   for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
         float x = (float) ix;
         data[ix*sx + iy*sy] = sin(kx * x + phi);
      }
   }

   return true;
}

float Gratings::calcPhase(float time, float dt)
{
	int iperiod = (int) (period/dt);
	int itime   = (int) (time/dt);

    if (itime%iperiod == 0) {
    	double p = ((double) rand()) / (double) RAND_MAX;
    	if (p < 0.25)       phase = 0.0 * PI;
    	else if (p < 0.50)  phase = 0.5 * PI;
    	else if (p < 0.75)  phase = 1.0 * PI;
    	else                phase = 1.5 * PI;
    }

	return phase;
}


} // namespace PV
