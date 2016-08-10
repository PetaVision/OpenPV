/*
 * ImageLayer.hpp
 *
 *    Layer that represents an image or list of images
 *    loaded from disk. Supports .jpg, .png, and .bmp
 *    formats, or a .txt list of files in those formats.
 */


#ifndef IMAGELAYER_HPP_
#define IMAGELAYER_HPP_

#include "utils/Image.hpp"
#include "InputLayer.hpp"

#include <cMakeHeader.h>

namespace PV {

   class ImageLayer : public InputLayer {

   protected:
      ImageLayer();
      int initialize(const char * name, HyPerCol * hc);
      virtual Buffer retrieveData(std::string filename);
      virtual void readImage(std::string filename);
      virtual int postProcess(double timef, double dt);
      virtual bool readyForNextFile();

      /**
       * Converts a grayscale buffer to a multiband buffer, by replicating the buffer in each band.
       * On entry, *buffer points to an nx-by-ny-by-1 buffer that must have been created with the new[] operator.
       * On exit, *buffer points to an nx-by-ny-by-numBands buffer that was created with the new[] operator.
       */
//      static int convertGrayScaleToMultiBand(float ** buffer, int nx, int ny, int numBands);

      /**
       * Converts a multiband buffer to a grayscale buffer, using the colorType to weight the bands.
       * On entry, *buffer points to an nx-by-ny-by-numBands buffer that must have been created with the new[] operator.
       * On exit, *buffer points to an nx-by-ny-by-1 buffer that was created with the new[] operator.
       */
//      static int convertToGrayScale(float ** buffer, int nx, int ny, int numBands, InputColorType colorType);

      /**
       * Based on the value of colorType, fills the bandweights buffer with weights to assign to each band
       * of a multiband buffer when converting to grayscale.
       */
//      static int calcBandWeights(int numBands, float * bandweights, InputColorType colorType);

      /**
       * Called by calcBandWeights when the color type is unrecognized; it fills each bandweights entry
       * with 1/numBands.
       */
/*      static inline void equalBandWeights(int numBands, float * bandweights) {
         float w = 1.0/(float) numBands;
         for( int b=0; b<numBands; b++ ) bandweights[b] = w;
      }
*/

   public:
      ImageLayer(const char * name, HyPerCol * hc);
      virtual ~ImageLayer();

   private:
      int initialize_base();

   protected:
      std::unique_ptr<Image> mImage;

   };
}

#endif 
