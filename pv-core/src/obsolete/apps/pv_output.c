#include "pv.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/**
 * Dumps output to path, extension must be added
 */
void pv_dump(char*, float, float, float[], float[], float[], float[]);

void ps_image(char*, float, float, float, float[], float[], float[], float[]);

void pv_output(char* path, float threshold, float x0, float y0,
	      float x[], float y[], float o[], float I[])
{
  char* filename;
  int path_len = strlen(OUTPUT_PATH) + 1 + strlen(path);
  int path_size = (path_len+8)*sizeof(char);	// contains a little extra space

  filename = malloc(path_size);
  if ( filename == NULL ) {
    fprintf(stderr, "ERROR:pv_output: malloc(%d bytes) failed\n", path_size);
    exit(1);
  }

#if defined(OUTPUT_PS)
  strcpy(filename, OUTPUT_PATH);
  strncat(filename, "/", 2);
  strcat(filename, path);
  strncat(filename, ".ps", 4);
  ps_image(filename, threshold, x0, y0, x, y, o, I);
#endif

#if defined(OUTPUT_BIN)
  strcpy(filename, OUTPUT_PATH);
  strncat(filename, "/", 2);
  strcat(filename, path);
  strncat(filename, ".bin", 5);
  pv_dump(filename, x0, y0, x, y, o, I);
#endif

  free(filename);
}


void pv_dump(char* filename, float x0, float y0,
	     float x[], float y[], float o[], float I[])
{
  FILE* fd = fopen(filename, "ab");
  if (fd != NULL) {
	fwrite( I, sizeof(float), N, fd );
	fclose(fd);
  }
}


void ps_image(char* filename, float threshold, float x0, float y0,
	      float x[], float y[], float o[], float I[])
{
  int i;
  int xx, yy, oo;
  FILE* fd = fopen(filename, "w");
  
  if (fd == NULL) return;

  fprintf(fd, "/pix {5 mul} def\n\n");
  fprintf(fd, "newpath\n\n");
  fprintf(fd, "10 pix 20 pix translate\n\n");

  for (i = 0; i < N; i++) {
    if (I[i] >= threshold) {
      //      if (i == 27885) {
      //	printf("ps_image: i = %d\n", i);
      //      }
      xx = (int) (0.5 + x0 + x[i]);
      yy = (int) (0.5 + y0 + y[i]);
      oo = (int) (0.5 + o[i]);
      fprintf(fd, "%d pix %d pix translate %d rotate\n", xx, yy, oo);
      fprintf(fd, "-1.35 pix 0 pix moveto\n");
      fprintf(fd, "1.35 pix 0 pix rlineto\n");
      fprintf(fd, "-%d rotate -%d pix -%d pix translate\n\n", oo, xx, yy);
    }
  }

  fprintf(fd, "closepath\n");
  fprintf(fd, "stroke\n");
  fprintf(fd, "showpage\n");

  fclose(fd);
}


void post(float threshold, float x0, float y0, float x[], float y[], float o[], float F[])
{
  FILE *fp;
  int j;

  fp = fopen("./pvout.dat", "w");
  for (j = 0; j < N; j += 4) {
    fprintf(fp, "%6d:  %11.3f %11.3f %11.3f %11.3f\n", j,
	    F[j], F[j+1], F[j+2], F[j+3]);
  }
  fclose(fp);

  pv_output("output", threshold, x0, y0, x, y, o, F);
}


/**
 * Compress a floating pt mask array (0.0 and 1.0 elements) to a bit pattern
 */
void compress_float_mask(int size, float buf[], unsigned char bits[])
{
  int i, j, k = 0;
  unsigned char shift, c;

  int jend = size%8;
  jend = (jend == 0) ? 8 : jend;
  
  assert(jend == 8);  // TODO - should only be applied at end?

  for (i = 0; i < size; i += 8) {
    c = 0x0;
    shift = 0x1;
    for (j = 0; j < jend; j++) {
      if (buf[i+j] > 0.0) c += shift;
      shift *= 2;
    }

    bits[k++] = c;
  }
}


void pv_output_events_on_circle(int step, float f[], float h[])
{
  int k;
  const int size = 35;
  int ka[] = {2727,2728,2736,2744,2753,2999,3057,3270,3362,3550,
		      3658,4117,4243,4685,4827,4972,5116,5260,5404,5548,
		      5692,5835,5981,6419,6549,7002,7118,7298,7398,7601,
		      7671,7905,7912,7920,7928};
  
  printf("%d: f  =", step);
  for (k = 0; k < size; k++) {
	  if (f[ka[k]] > 0.0) printf("1");
	  else printf("0");
  }

  printf(" h=");
  for (k = 0; k < size; k++) {
	  if (h[ka[k]] > 0.0) printf("1");
	  else printf("0");
  }

  printf("\n");
}


void pv_output_on_circle(int step, const char* name, float max, float buf[])
{
  int k;
  const int size = 35;
  int ka[] = {2727,2728,2736,2744,2753,2999,3057,3270,3362,3550,
		      3658,4117,4243,4685,4827,4972,5116,5260,5404,5548,
		      5692,5835,5981,6419,6549,7002,7118,7298,7398,7601,
		      7671,7905,7912,7920,7928};
  
  printf("%d: %s=", step, name);
  for (k = 0; k < size; k++) {
	  int val = (int) (10.0/max)*buf[ka[k]];
	  if (val < 0) printf("-");
	  else printf("%1d", val);
  }
  printf("\n");
}
