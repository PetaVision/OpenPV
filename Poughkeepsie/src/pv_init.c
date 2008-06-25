#include "pv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * initialize variables
 *
 * phi - membrane potential
 * x  - x position of neuron
 * y  - y position of neuron
 * th - orientation of neuron
 * I  - input pixel (line) image
 * f  - firing event mask
 *
 */
void pv_init(PVState* s, int nx, int ny, int no)
{
  float xc, yc, oc;
  int i, j, k, t; 
  float r;

  float dx, dy, r2;
  float X0 = (s->n_cols * NX)/2.;
  float Y0 = (s->n_rows * NY)/2.;

  float pi = 2.0*acos(0.0);
  float deg_rad = pi/180.0;

  float x0 = s->loc.x0;
  float y0 = s->loc.y0;

  float* x = s->loc.x;
  float* y = s->loc.y;
  float* o = s->loc.o;

  float* phi = s->phi;
  float* V   = s->V;
  float* I   = s->I;

  const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;
  const float clutter_prob = (.0001) * 1*0.01; // prob of pixel being "on"
  // Imax: 0.25->65Hz, 0.5>110Hz, 1.0>315Hz (for noise_amp = 0.75)
  const float Imax = 1.0*0.5*V_TH_0;  // maximum value of input image
  float r_circle = NX*s->n_cols / (float) 4;
  float r_tolerance = 0.04; // percent deviation from radius, 0.05 = max for non-sloppy circle 
  float r2_min = r_circle * r_circle * (1 - r_tolerance) * (1 - r_tolerance);
  float r2_max = r_circle * r_circle * (1 + r_tolerance) * (1 + r_tolerance);

  k = 0;
  for (j = 0; j < ny; j++) {
    yc = y0 + j*DY;
    for (i = 0; i < nx; i++) {
      xc = x0 + i*DX;
      for (t = 0; t < no; t++) {
	oc = t*DTH;
	
	x[k] = xc - x0;
	y[k] = yc - y0;
	o[k] = oc;
	
	phi[k]  = 0.0;
	V[k]    = 0.0;
	
	/* turn on random edges */
	r = rand() * INV_RAND_MAX;	
	I[k] = (r < clutter_prob) ? Imax : 0.0;
	
	/* turn on circle pixels */
	dx = (xc - X0);
        dy = (yc - Y0);
	r2 = dx*dx + dy*dy;
	if (r2 > r2_min && r2 < r2_max) {
	  int kk;
	  float a = 90.0 + (180./pi)*atanf(dy/dx);
	  a = fmod(a,180.);
	  kk = 0.5 + (a / DTH);
	  kk = kk % NO;
	  if (t == kk && I[k] == 0.0) {
	    I[k] = Imax;//0.0;
	    //printf("t=%d kk=%d k=%d o = %f r = %f (%d, %d) (%f, %f)\n", t, kk, k, a, sqrt(r2), i, j, dx, dy);
	  }
	}
	
	/**********************/
	/*         I[k] = 0.0; */
	/*         if (i == 48 && j == 28 && t == 8) { */
	/* 	  I[k] = 1.0; */
	/* 	  printf(" setting lone value (%d,%d) k=%d\n", i, j, k); */
	/* 	} */
	/***********************/
	
	k = k + 1;
	
      } // t < no
    } // i < nx
  } // j < ny
  
  /* turn on lone pixel in center at orientation t */
      j = ny/2; 
     i = nx/2; 
     t = 0; 
      k = t + j * no + i * nx * no; 
      phi[k]  = 1.0; 
      V[k]    = 1.0; 
      I[k] = I[k];//1.0; 

}



int random_16()
{
  return (random() & 0x0000000f);
}

int random_256()
{
  return (random() & 0x000000ff);
}

int random_4096()
{
  return (random() & 0x00000fff);
}


/****** old crap ******/

//   o[k] = fmodf(i*DTH, 180.);
//   f[k] =  fmodf((float)k, (float)2);

	/* 2 sides of equilateral triangle filling most of the image */

/******
	if (i > 15 && i < 80) {
	  if (j == 16 && t == 0) {
	    I[k] = 1.0;		// horizontal line
	  }
	  if (i == j && t == 2) {
	    I[k] = 1.0;		// 40 deg angle
	  }
	}

	if (j > 15 && j < 80) {
	  if (i == 80 && t == 4) {
	    I[k] = 1.0;		// 80 deg angle
	  }
	}

*******/
