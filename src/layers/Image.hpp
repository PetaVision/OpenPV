/*
 * Image.hpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */


#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "utils/PVImg.hpp"
#include "BaseInput.hpp"

#include <cMakeHeader.h>

namespace PV {

   class Image : public BaseInput {

   protected:
      /** 
       * List of parameters needed from the Image class
       * @name Image Parameters
       * @{
       */
      /**
       * @brief writeStep: The Image class changes the default of writeStep to -1 (i.e. never write to the output pvp file).
       */
      virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
      /** @} */

   protected:
      Image();
      int initialize(const char * name, HyPerCol * hc);
      virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
      virtual Buffer retrieveData(std::string filename);
      virtual void readImage(std::string filename);
      virtual int postProcess(double timef, double dt);

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
      Image(const char * name, HyPerCol * hc);
      virtual ~Image();
      virtual int communicateInitInfo();
      virtual double getDeltaUpdateTime();
      virtual int updateState(double time, double dt);

   private:
      int initialize_base();

   protected:
      std::unique_ptr<PVImg> mImage;

   }; // class Image
}  // namespace PV

#endif /* IMAGE_HPP_ */
