#include "src/io/tiff.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[])
{
   FILE  * fd;
   float * data;

   long nextLoc;
   int convert, width, height;

   if (argc != 2) {
      printf("USAGE: test_read_tiff filename\n");
      exit(1);
   }

   fd = fopen(argv[1], "rb");
   if (fd == NULL) {
      printf("ERROR opening %s\n", argv[1]);
      exit(1);
   }

   tiff_read_header(fd, &nextLoc, &convert);

   while (nextLoc != 0) {
      // get the size of the next image
      long loc = nextLoc;  /* save this location */
      tiff_image_size(fd, &nextLoc, &width, &height, convert);
      nextLoc = loc;       /* use saved location */

      // allocate space for the image
      data = (float *) malloc(width * height * sizeof(float));
      assert(data != NULL);

      // copy the image into the data buffer
      tiff_copy_image_float(fd, &nextLoc, data, width*height, convert);

      free(data);
   }

   return 0;
}
