/*
 * test_read_binary.c
 *
 *  Created on: Mar 31, 2009
 *      Author: rasmussn
 */

#include "io.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char * argv[])
{
   int nx, ny, nf, nItems;
   size_t sx, sy, sf;
   int x, y, f;
   int err = 0;

   PVLayerLoc loc;

   pvdata_t   value;
   pvdata_t * buf;

   if (argc != 2) {
      fprintf(stderr, "usage: test_read_binary path\n");
      return 1;
   }

   FILE * fd = pv_open_binary(argv[1], &nx, &ny, &nf);
   if (fd == NULL) {
      fprintf(stderr, "pv_open_binary: FAILED to open %s\n", argv[1]);
      return -1;
   }

   loc.nx = nx;  loc.ny = ny;  loc.nf = nf;  loc.nb = 1;
   loc.halo.lt = loc.halo.rt = loc.halo.dn = loc.halo.up;

   sx = strideX(&loc);
   sy = strideY(&loc);
   sf = strideF(&loc);

   printf("nx=%d, ny=%d, nf=%d\n", nx, ny, nf);
   printf("sx=%d, sy=%d, sf=%d\n", sx, sy, sf);

   nItems = nx * ny * nf;

   buf = (pvdata_t *) malloc(nItems * sizeof(pvdata_t));
   assert(buf != NULL);

   err = pv_read_binary_record(fd, buf, nItems);
   assert(err == 0);

   // pick a point and print value
   x = nx/2;
   y = ny/2;
   f = nf/2;

   value = buf[x*sx + y*sy + f*sf];
   printf("value at (x,y,f)=(%d,%d,%d) = %f\n", x, y, f, value);

   pv_close_binary(fd);

   free(buf);

   return err;
}
