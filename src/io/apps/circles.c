#include "src/include/pv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARTIAL_OUTPUT 1


int main(int argc, char* argv[])
  {
    const int NUM_CIRC = 2;
    const float REL_POS_X0[2]={1.0/4.0,3.0/4.0};
    const float REL_POS_Y0[2]={1.0/2.0,1.0/2.0};
    const float REL_RADIUS[2]={1.0/5.0,1.0/5.0};
    const char input_path[64] = "./input/circle2_";

    int k, i;
    int u[NUM_CIRC];
    int j, jj, jjj,p; //indices for x and y initialization

    eventtype_t f[N];
    float dx, dy, r, r2;
    int nrows = (int) 1;
    int ncols = (int) 1/nrows;
    float xa, ya;
    float x[N], y[N];

    float X0, Y0;
    float x0 = 0.0;
    float y0 = 0.0;


    p=0;
    for (j = 0; j < NY; j++)
      {
        ya = j*DY;
        for (jj = 0; jj < NX; jj++)
          {
            xa = jj*DX;

            for (jjj= 0; jjj < NO; jjj++)
              {


                x[p] = xa;

                y[p] = ya;
                p++;

              } // jjj < no
          } // jj < nx
      } // j < ny
   /*  for(i=0;i<NUM_CIRC;i++) */


    float pi = 2.0*acos(0.0);

    const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;
    const float clutter_prob = CLUTTER_PROB; // prob of pixel being "on"

    float r_circle;
    float r_tolerance = 0.04; // percent deviation from radius, 0.05 = max for non-sloppy circle
    float r2_min, r2_max;
    float xc, yc;

    //file charcter array declarations
    char filename_circlein[64];
    char number_of_in[64];
    char input_indices[64];

    //index string declaration
    char index[2];

    strcpy(number_of_in, input_path);
    strncat(number_of_in, "num",4);
    strncat(number_of_in, ".bin", 5);

    FILE* num = fopen(number_of_in, "wb");
    if(num==NULL)

	printf("ERROR: FAILED TO OPEN NUMBER FILE");

    else fwrite(&NUM_CIRC, sizeof(int), 1 ,num);

    fclose(num);

    strcpy(input_indices, input_path);
    strncat(input_indices, "input",6);
    strncat(input_indices, ".bin", 5);
    FILE* input_file = fopen(input_indices, "wb");
    if (input_file == NULL)
      printf("ERROR: FAILED TO OPEN INPUT FILE");
    for(i=0;i<NUM_CIRC;i++)
      {
	u[i]=0;
	r_circle = NX*ncols * REL_RADIUS[i];

	r2_min = r_circle * r_circle * (1 - r_tolerance) * (1 - r_tolerance);
	r2_max= r_circle * r_circle * (1 + r_tolerance) * (1 + r_tolerance);

	X0 = ncols*NX*REL_POS_X0[i];

	Y0 = nrows*NY*REL_POS_Y0[i];

	FILE* fo;

	sprintf(index,"%d",i);
	strcpy(filename_circlein, input_path);
	strncat(filename_circlein, "figure_",8);
	strncat(filename_circlein, index, 2);
	strncat(filename_circlein, ".bin", 5);

	fo = fopen(filename_circlein, "wb");
	if (fo == NULL)
	  {
		printf ("ERROR: FAILED TO OPEN FIGURE FILE NO. %d",k);
		continue;
	  }

	for (k = 0; k <N ; k++)
	  {

	    /* turn on random edges */
	    r = rand() * INV_RAND_MAX;
	    f[k] = (r < clutter_prob) ? I_MAX : 0.0;


	    xc = x[k] + x0;
	    yc = y[k] + y0;

	    /* turn on circle pixels */
	    dx = (xc - X0);
	    dy = (yc - Y0);
	    r2 = dx*dx + dy*dy;
	    if (r2> r2_min && r2 < r2_max)
	      {

		int t, kk;
		float a = 90.0 + (180./pi)*atanf(dy/dx);
		a = fmod(a, 180.);
		kk = 0.5 + (a / DTH);
		kk = kk % NO;
		// kk = rand() * NO; //randomize orientations
		t = k % NO; /* t is the orientation index */
		if (t == kk )//&& f[k] == 0.0)
		  {
		    f[k] = I_MAX;
		    fwrite(&k , sizeof(int), 1 , fo );
		    u[i]++;
		  /*   printf("we got %d now\n",u[i]); */
/* 		    printf("index %d\n", k) */;
		    //printf("t=%d kk=%d k=%d o = %f r = %f (%d, %d) (%f, %f)\n", t, kk, k, a, sqrt(r2), i, j, dx, dy);
		  }

	      }

	  }
	fclose(fo);
	if(input_file != NULL)
	  fwrite(f , sizeof(float), N , input_file);

      }
    printf("u1 = %d\n u2= %d\n",u[0],u[1]);
    fclose(input_file);
    FILE* fnum = fopen(number_of_in, "ab");
    if (fnum!= NULL)
      {
	for(i=0;i<NUM_CIRC;i++)
	  {
	    fwrite(&u[i], sizeof(int), 1, fnum);
	  }
	fclose(num);
      }
    printf("Please see input path for file results.");
    return 0;
  }

