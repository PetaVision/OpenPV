#include "tiff.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* number of IFD entries that this library writes */
#define NUM_OUT_ENTRIES 11

static uint16_t tiff_convert_short(unsigned char * buf, int convert);
static uint32_t tiff_convert_long(unsigned char * buf, int convert);
static uint16_t tiff_read_short(FILE * fd, int convert);
static uint32_t tiff_read_long(FILE * fd, int convert);

/**
 * simple test of tiff functions
 */
int test_main(int argc, char * argv[])
{
   int i;
   long nextLoc;
   const int width  = 64;
   const int height = 64;
   float V[width*height];
   unsigned char value = 0;

   FILE * fd = fopen("junk.tif", "wb");
   assert(fd != NULL);

   for (i = 0; i < width*height; i++) {
      V[i] = value++/255.0;
   }

   tiff_write_header(fd, &nextLoc);
   tiff_write_ifd(fd, &nextLoc, width, height);
   tiff_write_image(fd, V, width, height);
   tiff_write_finish(fd, nextLoc);

   fclose(fd);

   return 0;
}

uint16_t tiff_convert_short(unsigned char * buf, int convert)
{
   unsigned char s[2];
   if (convert) {
      s[0] = buf[1];  s[1] = buf[0];
   } else {
      s[0] = buf[0];  s[1] = buf[1];
   }
   return * (uint16_t *) s;
}

uint32_t tiff_convert_long(unsigned char * buf, int convert)
{
   unsigned char s[4];
   if (convert) {
      s[0] = buf[3];  s[1] = buf[2];  s[2] = buf[1];  s[3] = buf[0];
   } else {
      s[0] = buf[0];  s[1] = buf[1];  s[2] = buf[2];  s[3] = buf[3];
   }
   return * (uint32_t *) s;
}

uint16_t tiff_read_short(FILE * fd, int convert)
{
   unsigned char buf[2];
   assert( fread(buf, 1, 2, fd) == 2 );
   return tiff_convert_short(buf, convert);
}

uint32_t tiff_read_long(FILE * fd, int convert)
{
   unsigned char buf[4];
   assert( fread(buf, 1, 4, fd) == 4 );
   return tiff_convert_long(buf, convert);
}

/**
 * @param nx
 * @param ny
 */
int tiff_read_file(const char * filename, float * buf, int * width, int * height)
{
   long loc, nextLoc;
   int convert, nx, ny;
   FILE * fd;

   assert(filename != NULL);

   fd = fopen(filename, "rb");
   if (fd == NULL) {
      fprintf(stderr, "[ ]: tiff_read_file: ERROR, Input file %s not found.\n", filename);
      return 1;
   }

   tiff_read_header(fd, &nextLoc, &convert);

   // get the size of the image
   loc = nextLoc;  /* save this location */
   tiff_image_size(fd, &nextLoc, &nx, &ny, convert);
   nextLoc = loc;  /* use saved location */

   assert(*width  >= nx);
   assert(*height >= ny);

   *width  = nx;
   *height = ny;

   // copy the image into the data buffer
   tiff_copy_image_float(fd, &nextLoc, buf, nx*ny, convert);

   return fclose(fd);
}

int tiff_write_file(const char * filename, float * buf, int width, int height)
{
   long nextLoc;

   FILE * fd = fopen(filename, "wb");
   if (fd == NULL) {
      fprintf(stderr, "tiff_write_file: ERROR opening file %s\n", filename);
      return 1;
   }

   tiff_write_header(fd, &nextLoc);
   tiff_write_ifd(fd, &nextLoc, width, height);
   tiff_write_image(fd, buf, width, height);
   tiff_write_finish(fd, nextLoc);

   fclose(fd);

   return 0;
}

int tiff_write_file_drawBuffer(const char * filename, unsigned char * buf, int width, int height)
{
   long nextLoc;

      FILE * fd = fopen(filename, "wb");
      if (fd == NULL) {
         fprintf(stderr, "tiff_write_file: ERROR opening file %s\n", filename);
         return 1;
      }

      tiff_write_header(fd, &nextLoc);
      tiff_write_ifd(fd, &nextLoc, width, height);
      tiff_write_image_drawBuffer(fd, buf, width, height);
      tiff_write_finish(fd, nextLoc);

      fclose(fd);

      return 0;
}


static int tiff_write_next_offset(FILE * fd, long nextLoc, uint32_t val)
{
   uint32_t value = val;

   assert( fseek(fd, nextLoc, SEEK_SET) == 0 );
   assert( fwrite(&value, 1, 4, fd) == 4 );

   return 0;
}

static int tiff_read_ifd_entry(FILE * fd, IFDEntry * entry, int convert)
{
   entry->tag    = tiff_read_short(fd, convert);
   entry->type   = tiff_read_short(fd, convert);
   entry->count  = tiff_read_long(fd, convert);
   entry->offset = tiff_read_long(fd, convert);
   return 0;
}

static uint16_t tiff_convert_offset_to_short(IFDEntry entry, int convert)
{
   uint16_t * val = (uint16_t *) &entry.offset;
   if (convert) return val[1];
   else         return val[0];
}

static uint32_t tiff_convert_offset_to_long(IFDEntry entry, int convert)
{
   return entry.offset;
}

static int tiff_set_ifd_value(IFDEntry entry, IFD * ifd, int convert)
{
   switch (entry.tag) {
   case ImageWidthTag:
      ifd->imageWidth = tiff_convert_offset_to_long(entry, convert);
      break;
   case ImageLengthTag:
      ifd->imageLength = tiff_convert_offset_to_long(entry, convert);
      break;
   case BitsPerSampleTag:
      ifd->bitsPerSample = tiff_convert_offset_to_short(entry, convert);
      break;
   case CompressionTag:
      ifd->compression = tiff_convert_offset_to_short(entry, convert);
      break;
   case PhotometricInterpretationTag:
      ifd->photometricInterpretation = tiff_convert_offset_to_short(entry, convert);
      break;
   case StripOffsetsTag:
      if (ifd->numStripOffsets != 0) {
         assert(ifd->numStripOffsets == entry.count);
      }
      ifd->numStripOffsets = entry.count;
      ifd->stripOffsets = entry.offset;
      break;
   case RowsPerStripTag:
      ifd->rowsPerStrip = entry.offset;
      break;
   case StripByteCountsTag:
      if (ifd->numStripOffsets != 0) {
         assert(ifd->numStripOffsets == entry.count);
      }
      ifd->numStripOffsets = entry.count;
      ifd->stripByteCounts = entry.offset;
      break;
   case XResolutionTag:
      ifd->xResolution = entry.offset;
      break;
   case YResolutionTag:
      ifd->yResolution = entry.offset;
      break;
   case ResolutionUnitTag:
      ifd->resolutionUnit = entry.offset;
      break;
   /* unused here (note fall through) */
   case DocumentNameTag:
   case NewSubfileTypeTag:
   case ImageDescriptionTag:
   case OrientationTag:
   case SamplesPerPixelTag:
   case PlanarConfigurationTag:
   case UnknownTag:
      break;
   default:
      if (entry.tag < 32768) {
         printf("tiff_set_ifd_value: unknown value: tag=%d, type=%d, count=%d, offset=%d\n",
                entry.tag, entry.type, entry.count, entry.offset);
         return 1;
      }
   }

   return 0;
}

static IFDEntry tiff_get_entry(IFDEntry * entries, uint16_t numEntries, int tag)
{
   int i;
   IFDEntry error;

   for (i = 0; i < numEntries; i++) {
      if (entries[i].tag == tag) return entries[i];
   }
   assert(0);  // shouldn't arrive here
   return error;
}

int tiff_get_values_long(FILE * fd, IFDEntry entry, uint32_t * offsets, int convert)
{
   int i;

   if (entry.count == 1) {
      offsets[0] = entry.offset;
      return 0;
   }

   fseek(fd, entry.offset, SEEK_SET);
   for (i = 0; i < entry.count; i++) {
      offsets[i] = tiff_read_long(fd, convert);
   }

   return 0;
}

/**
 * write the tiff header
 * @param nextLoc contains the offset of the next IFD entry on return
 */
int tiff_write_header(FILE * fd, long * nextLoc)
{
   unsigned char buf[4];

   // byte order
   buf[0] = 'I';
   buf[1] = 'I';

   // version
   buf[2] = 42;
   buf[3] = 0;

   assert( fwrite(buf, 1, 4, fd) == 4 );

   // start of first IFDEntry
   *nextLoc = ftell(fd);
   tiff_write_next_offset(fd, *nextLoc, 0);

   assert( ftell(fd) == 8 );

   return 0;
}

/**
 * read the tiff header
 * @param nextLoc contains the offset of the next IFD entry on return
 */
int tiff_read_header(FILE * fd, long * nextLoc, int * pconvert)
{
   unsigned char buf[4];
   uint16_t version;
   int convert = 0;

   // endian (ignore, look at version to discover need to convert)
   assert( fread(buf, 1, 2, fd) == 2 );


   version = tiff_read_short(fd, convert);
   if (version != 42) {
      convert = 1;
      version = tiff_convert_short((unsigned char *) &version, convert);
   }
   assert(version == 42);
   *pconvert = convert;

   // start of first IFDEntry
   *nextLoc = tiff_read_long(fd, convert);

   assert( ftell(fd) == 8 );

   return 0;
}

/**
 * read the IFD and return the image size for future processing
  * @param nextLoc contains the offset to the IFD entry
 */
int tiff_image_size(FILE * fd, long * nextLoc, int * width, int * height, int convert)
{
   int i;
   uint16_t numEntries;
   IFDEntry entry;

   IFD ifd;
   ifd.numStripOffsets = 0;  /* MUST be initialized */
   ifd.imageWidth = 0;
   ifd.imageLength = 0;

   assert( fseek(fd, *nextLoc, SEEK_SET) == 0 );

   numEntries = tiff_read_short(fd, convert);
   for (i = 0; i < numEntries; i++) {
      tiff_read_ifd_entry(fd, &entry, convert);
      tiff_set_ifd_value(entry, &ifd, convert);
   }
   *nextLoc = tiff_read_long(fd, convert);

   *width  = ifd.imageWidth;
   *height = ifd.imageLength;

   return 0;
}

int tiff_copy_image(FILE * fd, IFD * ifd, IFDEntry * entries, uint16_t numEntries,
                    unsigned char * data, int convert)
{
   int i;
   IFDEntry stripOffsetsEntry, stripByteCountsEntry;
   uint32_t * offsets, * counts;

   unsigned char * buf = data;

   offsets = (uint32_t *) malloc(ifd->numStripOffsets * sizeof(uint32_t));
   assert(offsets != NULL);

   counts = (uint32_t *) malloc(ifd->numStripOffsets * sizeof(uint32_t));
   assert( counts != NULL);

   stripOffsetsEntry    = tiff_get_entry(entries, numEntries, StripOffsetsTag);
   stripByteCountsEntry = tiff_get_entry(entries, numEntries, StripByteCountsTag);

   tiff_get_values_long(fd, stripOffsetsEntry, offsets, convert);
   tiff_get_values_long(fd, stripByteCountsEntry, counts, convert);

   // read data
   for (i = 0; i < ifd->numStripOffsets; i++) {
      fseek(fd, offsets[i], SEEK_SET);
//      assert( fread(buf, 1, counts[i], fd) == counts[i] );
      buf += counts[i];
   }

   if (ifd->photometricInterpretation == 0) {
      // make black be zero
      size_t size = ifd->imageLength * ifd->imageWidth;
      for (i = 0; i < size; i++) {
         data[i] = 255 - data[i];
      }
   }

   free(offsets);
   free(counts);

   return 0;
}

/**
 * read the IFD
 *   - must write current position into the previous directories next location
 * @param nextLoc contains the offset of the previous IFD entry (and next on return)
 */
int tiff_copy_image_float(FILE * fd, long * nextLoc, float * data, size_t numItems, int convert)
{
   int i;
   uint16_t numEntries;
   unsigned char * buf;
   IFDEntry * entries;

   float min = 0.0;
   float max = 1.0;

   IFD ifd;
   ifd.numStripOffsets = 0;  /* MUST initialize */

   assert( fseek(fd, *nextLoc, SEEK_SET) == 0 );

   numEntries = tiff_read_short(fd, convert);
   entries = (IFDEntry *) malloc(numEntries * sizeof(IFDEntry));
   assert(entries != NULL);

   for (i = 0; i < numEntries; i++) {
      tiff_read_ifd_entry(fd, &entries[i], convert);
      tiff_set_ifd_value(entries[i], &ifd, convert);
#ifdef DEBUG_TIFF_LIBRARY
      printf("  tag=%d, type=%d, count=%d, offset=%d\n",
             entries[i].tag, entries[i].type, entries[i].count, entries[i].offset);
#endif
   }
   *nextLoc = tiff_read_long(fd, convert);

   assert(numItems == ifd.imageWidth * ifd.imageLength);

   buf = (unsigned char *) malloc(ifd.imageWidth * ifd.imageLength * sizeof(unsigned char));
   assert(buf != NULL);

   // copy image
   tiff_copy_image(fd, &ifd, entries, numEntries, buf, convert);

   // copy and convert data
   for (i = 0; i < numItems; i++) {
      data[i] = min + buf[i] * (max - min) / 255;
   }

   free(buf);
   free(entries);

   return 0;
}

/**
 * write the IFD
 *   - must write current position into the previous directories next location
 * @param nextLoc contains the offset of the previous IFD entry (and next on return)
 */
int tiff_write_ifd(FILE * fd, long * nextLoc, int width, int height)
{
   int i;
   long pos;
   char buf[64];
   uint32_t offset, valuesOffset, imageOffset;
   int imageSize, stripSize, lastStripSize, numStripOffsets, rowsPerStrip;

   IFDEntry entry;

   uint16_t numEntries = NUM_OUT_ENTRIES;

   assert(width  > 0);
   assert(height > 0);

   // write out location of this IFD in previous one
   pos = ftell(fd);
   tiff_write_next_offset(fd, *nextLoc, pos);
   assert( fseek(fd, pos, SEEK_SET) == 0 );

   // calculate some initial parameters

   imageSize = width*height;
   numStripOffsets = 1 + imageSize/8000;  // rough estimate
   stripSize = imageSize / numStripOffsets;
   rowsPerStrip = stripSize / width;
   stripSize = width * rowsPerStrip;
   lastStripSize = imageSize - (numStripOffsets - 1) * stripSize;

   assert(lastStripSize > 0);
   assert(lastStripSize % width == 0);

   // number of entries
   pos = ftell(fd);
   valuesOffset = pos + 2 + numEntries * sizeof(entry) + 4; // last 4 for next IFD offset
   assert( fwrite(&numEntries, 2, 1, fd) == 1 );

   offset = valuesOffset;
   imageOffset = valuesOffset;

   // correct offset for image
   if (numStripOffsets > 1) {
      imageOffset += 2 * numStripOffsets * sizeof(uint32_t);
   }
   imageOffset += 4 * sizeof(uint32_t);  // for X and Y Resolution

   // can be reused for some initial fields
   entry.type  = 3;  // SHORT
   entry.count = 1;

   // ImageWidth
   entry.tag    = ImageWidthTag;
   entry.offset = width;
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // ImageLength
   entry.tag    = ImageLengthTag;
   entry.offset = height;
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // BitsPerSample
   entry.tag    = BitsPerSampleTag;
   entry.offset = 8;  // 8 bits per pixel (shades of gray)
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // Compression
   entry.tag    = CompressionTag;
   entry.offset = 1;  // no compression (but packed)
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // PhotometricInterpretation
   entry.tag    = PhotometricInterpretationTag;
   entry.offset = 1;  // white is zero, black is one
   assert( fwrite(&entry, 12, 1, fd) == 1 );

// TODO - make sure offsets are on word boundaries

   // StripOffsets
   entry.tag    = StripOffsetsTag;
   entry.type   = 4;  // LONG
   entry.count  = numStripOffsets;
   if (numStripOffsets == 1) {
      entry.offset = imageOffset;
   }
   else {
      entry.offset = offset;
      offset += entry.count * sizeof(uint32_t);
   }
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // RowsPerStrip
   entry.tag    = RowsPerStripTag;
   entry.type   = 3;  // SHORT
   entry.count  = 1;
   entry.offset = rowsPerStrip;
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // StripByteCounts
   entry.tag    = StripByteCountsTag;
   entry.type   = 4;  // LONG
   entry.count  = numStripOffsets;
   if (numStripOffsets == 1) {
      entry.offset = imageSize;
   }
   else {
      entry.offset = offset;
      offset += entry.count * sizeof(uint32_t);
   }
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // XResolution
   entry.tag    = XResolutionTag;
   entry.type   = 5;  // RATIONAL
   entry.count  = 1;
   entry.offset = offset;
   offset += 2 * entry.count * sizeof(uint32_t);
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // YResolution
   entry.tag    = YResolutionTag;
   entry.type   = 5;  // RATIONAL
   entry.count  = 1;
   entry.offset = offset;
   offset += 2 * entry.count * sizeof(uint32_t);
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // ResolutionUnit
   entry.tag    = ResolutionUnitTag;
   entry.type   = 3;  // SHORT
   entry.count  = 1;
   entry.offset = 2;  // inch
   assert( fwrite(&entry, 12, 1, fd) == 1 );

   // offset of next IFD (initially zero)
   *nextLoc = ftell(fd);
   tiff_write_next_offset(fd, *nextLoc, 0);

   assert( ftell(fd) == valuesOffset );

   // strip offsets and sizes
   if (numStripOffsets > 1) {
      // write strip offsets
      for (i = 0; i < numStripOffsets; i++) {
         entry.offset = imageOffset + i * stripSize;
         assert( fwrite(&entry.offset, 4, 1, fd) == 1 );
      }
      // write strip sizes
      entry.offset = stripSize;
      for (i = 0; i < numStripOffsets; i++) {
         if (i == numStripOffsets - 1) {
            entry.offset = lastStripSize;
         }
         assert( fwrite(&entry.offset, 4, 1, fd) == 1 );
      }
   }

   // XResolution
   buf[0] = 0; buf[1] = 0; buf[2] = 0; buf[3] = 72;
   assert( fwrite(buf, 1, 4, fd) == 4 );
   buf[3] = 1;
   assert( fwrite(buf, 1, 4, fd) == 4 );

   // YResolution
   buf[3] = 72;
   assert( fwrite(buf, 1, 4, fd) == 4 );
   buf[3] = 1;
   assert( fwrite(buf, 1, 4, fd) == 4 );

   assert( ftell(fd) == imageOffset );

   return imageOffset;
}

int tiff_write_image(FILE * fd, float * buf, int width, int height)
{
   int i;
   unsigned char value;
   int imageSize = width * height;

   for (i = 0; i < imageSize; i++) {
     // TODO - check that range is 0:1
      value = 255 * buf[i];
      assert( fwrite(&value, 1, 1, fd) == 1 );
   }
   return 0;
}
/**
 * Writes from the restricted frame.
 *
 */
int tiff_write_image_drawBuffer(FILE * fd, unsigned char * buf, int width, int height)
{
   int i;
   unsigned char value;
   int imageSize = width * height;

   for (i = 0; i < imageSize; i++) {
     // TODO - check that range is 0:1
      value = 255 * (float) buf[i];
      assert( fwrite(&value, 1, 1, fd) == 1 );
   }
   return 0;
}

int tiff_write_finish(FILE * fd, long nextLoc)
{
   return tiff_write_next_offset(fd, nextLoc, 0);
}
