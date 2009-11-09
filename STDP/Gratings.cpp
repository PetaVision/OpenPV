/*
 * Gratings.cpp
 *
 *  Created on: Oct 23, 2009
 *      Author: travel
 */

#include "Gratings.hpp"
#include <src/include/pv_common.h>  // for PI
namespace PV {

//! A derived Image class

/*!
 * Sets the period is sync with the burst frequency of the Retina.
 * Sets the phase of the gratings (is this designed to work with restarted
 * simulations?)
 */
Gratings::Gratings(const char * name, HyPerCol * hc) :
	Image(name, hc) {
	initialize_data(&loc);

	PVParams * params = hc->parameters();
	float freq = params->value(name, "burstFreq", 40.0);

	period = 25;
	if (freq > 0.0) {
		period = 1000 / freq; // in miliseconds
	}

	float dt = hc->getDeltaTime();

	calcPhase(0.0, dt); // time==0.0 initializes random phase, don't delete
	calcPhase(hc->simulationTime(), dt);

	updateImage(0.0, 0.0);
}

Gratings::~Gratings() {
}

//! Updates the image

/*!
 * REMARKS:
 * 	- We work in the extended frame.
 *	- The data buffer is modulated by a sinusoid with
 *     wavelength kx.
 *	- The phase of this sinusoid is set
 *     by a call to calcPhase(). It stays constant during
 *     a time that equals the burstPeriod.
 *	- Comment on how are negative data values handled!
 *	.
 *
 * NOTES:
 *	- We should make this time longer by modifying
 *        calcPhase().
 *
 *	.
 */
bool Gratings::updateImage(float time, float dt) {
	// extended frame
	const int nx = loc.nx + 2 * loc.nPad;
	const int ny = loc.ny + 2 * loc.nPad;
	const int sx = 1;
	const int sy = sx * nx;

	int marginWidth = loc.nPad;

	const float kx = 2.0 * PI / 4.0; // wavenumber
	const float phi = calcPhase(time, dt);

	for (int iy = 0; iy < ny; iy++) {
		//	   int ix = 6 + marginWidth;
		//      for (int ix = 0; ix < nx; ix++) {
		//         float x = (float) (ix + marginWidth);
		//         data[ix*sx + iy*sy] = sin(kx * x + phi);
		//      }
		for (int ix = 6; ix < nx; ix += 6) {
			data[ix * sx + iy * sy] = 1;
		}

	}

	return true;
}

//! Calculates phase of the gratings

/*!
 * When time is a multiple of period we assign one of four possible phases.
 *
 *
 */

float Gratings::calcPhase(float time, float dt) {
	int iperiod = (int) (period / dt);
	int itime = (int) (time / dt);

	if (itime % iperiod == 0) {
		double p = ((double) rand()) / (double) RAND_MAX;
		if (p < 0.25)
			phase = 0.0 * PI;
		else if (p < 0.50)
			phase = 0.5 * PI;
		else if (p < 0.75)
			phase = 1.0 * PI;
		else
			phase = 1.5 * PI;
	}

	return phase;
}

} // namespace PV
