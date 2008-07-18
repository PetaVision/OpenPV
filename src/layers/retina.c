#include "pv.h"
#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * Constructor for a Retina (PVLayer).
 * 
 * This layer just contains the firing events for an input image (currently a circle with clutter).
 * 
 */

PVLayer* pv_new_layer_retina(PVHyperCol* hc, int index, int nx, int ny, int no)
  {
    int k;
    float dx, dy, r, r2;

    PVLayer* l = pv_new_layer(hc, index, nx, ny, no);

    eventtype_t* f = l->f;

    char filename[75];

    // If the user passed a filename on the command line use that.
    if (hc->input_filename[0] != 0)
        strcpy(filename, hc->input_filename);
    else
        strcpy(filename, INPUT_PATH);
    
    FILE* input= fopen(filename, "rb");
    if (input == NULL)
      {
	printf("Error: input file specified was not found.\n Please make sure file exists or use an input constructor to make it.\n Using default random clutter.\n");
	int k;
	for(k=0; k < l->n_neurons; k++) 
	  {
	    const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;
	    r = rand() * INV_RAND_MAX; 
	    f[k] = (r < CLUTTER_PROB) ? I_MAX : 0.0;
	  }
      }
    else
      fread(f,sizeof(eventtype_t),l->n_neurons,input);
       
    fclose(input);
return l;
  }



/* PVLayer* pv_new_layer_retina(PVHyperCol* hc, int index, int nx, int ny, int no) */
/*   { */
/*     int k; */
/*     float dx, dy, r, r2; */

/*     PVLayer* l = pv_new_layer(hc, index, nx, ny, no); */

/*     eventtype_t* f = l->f; */

/*     float* x = l->x; */
/*     float* y = l->y; */

/*     float X0 = (hc->n_cols * NX)/2.; */
/*     printf("X0=%f",X0); */
/*     float Y0 = (hc->n_rows * NY)/2.; */
/*     printf("Y0=%f",Y0); */
/*     float x0 = hc->x0; */

/*     float y0 = hc->y0; */

/*     float pi = 2.0*acos(0.0); */

/*     const float INV_RAND_MAX = 1.0 / (float) RAND_MAX; */
/*     const float clutter_prob = CLUTTER_PROB; // prob of pixel being "on" */
/*     // const float clutter_prob = (.0001) * 1*0.01; // prob of pixel being "on" */
/*     // Imax: 0.25->65Hz, 0.5>110Hz, 1.0>315Hz (for noise_amp = 0.75) */

/*     const float Imax = I_MAX; // maximum value of input image */
   
/*     // const float Imax = 1.0*0.5*V_TH_0; // maximum value of input image */

/*     float r_circle= NX*hc->n_cols / (float) 4; */
/*     float r_tolerance = 0.04; // percent deviation from radius, 0.05 = max for non-sloppy circle  */
/*     float r2_min = r_circle * r_circle * (1 - r_tolerance) * (1 - r_tolerance); */
/*     float r2_max = r_circle * r_circle * (1 + r_tolerance) * (1 + r_tolerance); */
    
/*     int u=0; */

/*     char filename_circlein[64]; */
/*     char number_of_in[64]; */
/*     strcpy(filename_circlein, OUTPUT_PATH); */
/*     strncat(filename_circlein, "/", 2); */
/*     strncat(filename_circlein, "ind",4); */
/*     strncat(filename_circlein, ".bin", 5); */
/*     strcpy(number_of_in, OUTPUT_PATH); */
/*     strncat(number_of_in, "/", 2); */
/*     strncat(number_of_in, "num",4); */
/*     strncat(number_of_in, ".bin", 5); */
    

/*     FILE* fo = fopen(filename_circlein, "ab"); */
   
/*     for (k = 0; k < l->n_neurons; k++) */
/*       { */
/*         float xc = x[k] + x0; */
/*         float yc = y[k] + y0; */

/*         turn on random edges  */
/*         r = rand() * INV_RAND_MAX; */
/*         f[k] = (r < CLUTTER_PROB) ? I_MAX : 0.0; */

/*         turn on circle pixels */
/*         dx = (xc - X0); */
/*         dy = (yc - Y0); */
/*         r2 = dx*dx + dy*dy; */
/*         if (r2> r2_min && r2 < r2_max) */
/*           { */
/*             int t, kk; */
/*             float a = 90.0 + (180./pi)*atanf(dy/dx); */
/*             a = fmod(a, 180.); */
/*             kk = 0.5 + (a / DTH); */
/*             kk = kk % NO; */
/* 	    // kk = rand() * NO; //randomize orientations */
/*             t = k % NO;  t is the orientation index  */
/*             if (t == kk) // && f[k] == 0.0) */
/*               { */
/*                 f[k] += I_MAX; */
/* 		if (fo != NULL) */
/* 		  fwrite(&k , sizeof(float), 1 , fo );  */
/* 		u++; */
/*                 //printf("t=%d kk=%d k=%d o = %f r = %f (%d, %d) (%f, %f)\n", t, kk, k, a, sqrt(r2), i, j, dx, dy); */
/*               } */
/* 	  } */
	


/*       } */
/*     fclose(fo); */
/*     FILE* num = fopen(number_of_in, "ab"); */
/*     fwrite(&u, sizeof(int), 1, num); */
/*     fclose(num); */
/*     return l; */
/*   } */
