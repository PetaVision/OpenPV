/*
 * RetinaGrating.cpp
 *
 *  Created on: Jun 7, 2009
 *      Author: rasmussn
 */

#include "RetinaGrating.hpp"
/*
 * int opt determines the type of grating created:
 *    opt == 1 creates a horizontal grating
 *    opt == 2 creates a vertical grating
 *    opt == 3 creates a horizontal grating with a small cutout of the vertical
 *             grating in the center
 *    opt == 4 creates a close-up vertical grating
 */
namespace PV {

RetinaGrating::RetinaGrating(const char * name, HyPerCol * hc, int opt)
   : Retina(name, hc)
{
   // must call here because call from Retina constructor not virtual
   createImage(clayer->V, opt);
}

int RetinaGrating::createImage(pvdata_t * buf, int opt) {
   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;

   PVParams * params = parent->parameters();

   float lambda = 2.0;
   if (params->present(getName(), "lambda")) lambda = params->value(getName(), "lambda");

   for (int k = 0; k < clayer->numNeurons; k++) {

	  float kx = kxPos(k, nx, ny, nf);
	  float ky = kyPos(k, nx, ny, nf);

	  if (opt == 1)       buf[k] = cos(2 * PI * kx / lambda);
	  else if (opt == 2)  buf[k] = cos(2 * PI * ky / lambda );
	  else if (opt == 3) {
		  if( (kx > nx/3) && (kx < 2 * nx/3) ) {
			  if( (ky > ny/3) && (ky < 2 * ny/3) ) {
				  buf[k] = cos(2 * PI * ky / lambda );
			  }
			  else buf[k] = cos(2 * PI * kx / lambda);
		  }
		  else     buf[k] = cos(2 * PI * kx / lambda);
	  }
	  else if(opt == 4) {
		  lambda = nx / 1.5;
		  buf[k] = sin(2 * PI * ky / lambda);
	  }
   }

   return 0;
}

int RetinaGrating::updateState(float time, float dt)
{
   int start;

   fileread_params * params = (fileread_params *) clayer->params;

   pvdata_t * V = clayer->V;
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < clayer->numNeurons; k++) {
      float probStim = params->poissonEdgeProb * V[k];
      float prob = params->poissonBlankProb;
//      float sinStatus = sin( 2 * PI * time * params->burstFreq / 1000. );
      int flag = spike(time, dt, prob, probStim, &start);
      activity[k] = (flag) ? 1.0 : 0.0;
//      activity[k] = flag;
//      V[k] = V[k] * sinStatus;
//      activity[k] = (flag > params->poissonBlankProb) ? 1.0 : 0.0;
   }

   return 0;
}

} // namespace PV
