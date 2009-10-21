#ifndef TIFF_H_
#define TIFF_H_

#include <stdio.h>
#include <stdint.h>

/*
 * Tiff file format obtained from http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
 */

/*
 * Fields for grayscale images
 */
typedef enum {
   NewSubfileTypeTag = 254,               /* LONG */
   ImageWidthTag = 256,                   /* SHORT or LONG */
   ImageLengthTag = 257,                  /* SHORT or LONG */
   BitsPerSampleTag = 258,                /* SHORT: 4 or 8 */
   CompressionTag = 259,                  /* SHORT: 1 or 32773 */
   PhotometricInterpretationTag = 262,    /* SHORT: 0 or 1 */
   DocumentNameTag = 269,                 /* ASCII */
   ImageDescriptionTag = 270,             /* ASCII */
   StripOffsetsTag = 273,                 /* SHORT or LONG */
   OrientationTag = 274,                  /* SHORT */
   SamplesPerPixelTag = 277,              /* SHORT */
   RowsPerStripTag = 278,                 /* SHORT or LONG */
   StripByteCountsTag = 279,              /* SHORT or LONG */
   XResolutionTag = 282,                  /* RATIONAL */
   YResolutionTag = 283,                  /* RATIONAL */
   PlanarConfigurationTag = 284,          /* SHORT */
   ResolutionUnitTag = 296,               /* SHORT: 1, 2 or 3 */
   UnknownTag = 700
} BaselineTags;

typedef enum {
   TiffShort = 3,                      /* unsigned 16 bits */
   TiffLong = 4,                       /* unsigned 32 bits */
   TiffRational = 5                    /* numerator / denominator */
} TiffType;

typedef struct {
   uint16_t tag;
   uint16_t type;
   uint32_t count;
   uint32_t offset;
} IFDEntry;

typedef struct IFD_ {
   uint32_t imageWidth;                   /* SHORT or LONG */
   uint32_t imageLength;                  /* SHORT or LONG */
   uint16_t bitsPerSample;                /* SHORT: 4 or 8 */
   uint16_t compression;                  /* SHORT: 1 or 32773 */
   uint16_t photometricInterpretation;    /* SHORT: 0 or 1 */
   uint32_t stripOffsets;                 /* SHORT or LONG */
   uint32_t rowsPerStrip;                 /* SHORT or LONG */
   uint32_t stripByteCounts;              /* SHORT or LONG */
   float    xResolution;                  /* RATIONAL */
   float    yResolution;                  /* RATIONAL */
   uint16_t resolutionUnit;               /* SHORT: 1, 2 or 3 */
   int      numStripOffsets;
} IFD;


#ifdef __cplusplus
extern "C"
{
#endif

int tiff_read_header(FILE * fd, long * nextLoc, int * convert);
int tiff_read_file(const char * filename, float * buf, int * width, int * height);
int tiff_image_size(FILE * fd, long * nextLoc, int * width, int * height, int convert);
int tiff_copy_image_float(FILE * fd, long * nextLoc, float * data, size_t numItems, int convert);

int tiff_write_header(FILE * fd, long * nextLoc);
int tiff_write_ifd(FILE * fd, long * nextLoc, int width, int height);
int tiff_write_image(FILE * fd, float * buf, int width, int height);
int tiff_write_image_drawBuffer(FILE * fd, unsigned char * buf, int width, int height);
int tiff_write_file(const char * filename, float * buf, int width, int height);
int tiff_write_file_drawBuffer(const char * filename, unsigned char * buf, int width, int height);
int tiff_write_finish(FILE * fd, long nextLoc);

#ifdef __cplusplus
}
#endif

#endif /* TIFF_H_ */
