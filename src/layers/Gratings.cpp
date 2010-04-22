/*
 * Gratings.cpp
 *
 *  Created on: Oct 23, 2009
 *      Author: Craig Rasmussen
 */

#include "Gratings.hpp"
#include "../include/pv_common.h"  // for PI
#include "../utils/pv_random.h"

namespace PV {

Gratings::Gratings(const char * name, HyPerCol * hc) : Image(name, hc)
{
   initialize_data(&loc);

   // initialize to unused phase to trigger update
   //
   this->phase = 0.5 * PI;
   this->lastPhase = phase;

   // set moving (random phase) probability
   pMove = 0;

   PVParams * params = hc->parameters();
   float freq = params->value(name, "burstFreq", 40.0);

   // check for explicit parameters in params.stdp
   if (params->present(name, "pMove")) {
      pMove = params->value(name, "pMove");
      //printf("pMove = %f\n", pMove);
   }

   // set parameters that controls writing of new images
    writeImages = params->value(name, "writeImages",0);

   period = 25;
   if (freq > 0.0) {
      period = 1000/freq;
   }

   float dt = hc->getDeltaTime();

   calcPhase(0.0, dt);  // time==0.0 initializes random phase, don't delete
   calcPhase(hc->simulationTime(), dt);

   updateImage(0.0, 0.0);
}

Gratings::~Gratings()
{
}

/**
 * NOTES:
 *    - Retina calls updateImage(float time, float dt) and expects a bool variable
 *    in return.
 *    - If true, the image has been changed; if false the image has not been
 * changed.
 *    - If true, the retina also calls copyFromImageBuffer() to copy the Image
 *    data buffer into the V buffer (it also normalizes the V buffer so that V <= 1).
 *
 */
bool Gratings::updateImage(float time, float dt)
{
   // extended frame
   const int nx = loc.nx + 2 * loc.nPad;
   const int ny = loc.ny + 2 * loc.nPad;
   const int sx = 1;
   const int sy = sx * nx;
   float phi;
   char basicfilename[128] = { 0 };

   const float kx = 2.0 * PI / 8.0; // wavenumber

   double p_move = 1.0 * rand() / RAND_MAX;

   if (p_move < pMove) {
      calcPhase(time, dt);
   }
   else {
      phase = lastPhase;
   }

   for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
         float x = (float) ix;
         data[ix * sx + iy * sy] = 1.0 + sinf(kx * x + phase);
      }
   }

   if (lastPhase != phase) {
      lastPhase = phase;
      lastUpdateTime = time;
      if (writeImages) {
         snprintf(basicfilename, 127, "Gratings_%.2f", time);
         write(basicfilename);
      }
      return true;
   }
   else {
      return false;
   }

   return true;
}

float Gratings::calcPhase(float time, float dt)
{
	int iperiod = (int) (period/dt);
	int itime   = (int) (time/dt);

    if (itime%iperiod == 0) {
    	double p = pv_random_prob();
    	if (p < 0.25)       phase = 0.0 * PI;
    	else if (p < 0.50)  phase = 0.5 * PI;
    	else if (p < 0.75)  phase = 1.0 * PI;
    	else                phase = 1.5 * PI;
    }

	return phase;
}

} // namespace PV
