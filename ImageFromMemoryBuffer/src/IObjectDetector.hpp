/*
 * IObjectDetector
 *
 * Created on Nov 3, 2014
 *     Author: Pete Schultz
 *
 * A placeholder class to implement the core_analytics.h specification
 */
 
#ifndef IOBJECTDETECTOR_HPP_
#define IOBJECTDETECTOR_HPP_

#include <stdint.h>
#include <columns/HyPerCol.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>

#define VIDINT_SUCCESS 0
#define VIDINT_FAILURE 1

namespace vidint {

typedef enum 
{
   GRAY8,
   RGB24,
   YV12
} ImageFormat;                                     //! Other formats will be added as needed, for example BSQ_RGB is probably needed for VAST

typedef struct
{
        int32_t x;                                     //!< pixels from the left edge to the start the ROI
        int32_t y;                                     //!< pixels from the top edge of the start the ROI
        int32_t width;                                 //!< width in pixels of the ROI (increase width to the right)
        int32_t height;                                //!< height in pixels of the ROI  (increasing height down the image)
} Roi;

typedef struct
{
    int32_t     width;                             //!< width of the image in pixels (zeroth column is at the left of the image)
    int32_t     height;                            //!< height of the image in pixels (zeroth row is at the top of the image)
    ImageFormat type;                              //!< identifier specifying pixel format
    uint8_t*    pPixel;                            //!< pointer to the start of pixel memory
} Image;

typedef struct                                     //! basic pitch/roll/yaw based on camera orientation
{ 
  float yaw;                                    
  float pitch;
  float roll;
} Rotation;

typedef struct                                                                          //! landmark definition for use in the object structure
{
        int32_t x;
        int32_t y;
        char    type[32];
} Landmark;

typedef struct
{
   Roi roi;                                                                                     //!< Objects may include roi and/or landmarks
   Landmark             landmarks[128];                                         //!< Landmarks are defined to facilitate passing the landmark information
   uint32_t             numberOfLandmarks;                  //!<  from the object detector to the encode() and process() operations
   Rotation rotation;
   float confidence;
   char type[32];   
} Object;

class IObjectDetector {
public:
   virtual int32_t detect(Image* image, Roi* roi, Object* results, uint32_t max, uint32_t* resultCount);

   IObjectDetector(PV::HyPerCol * hc, char const * imagelayername, double simTimeInterval);
   virtual ~IObjectDetector();
   
   PV::HyPerCol * getHyPerColumn() { return hypercolumn; }
   PV::ImageFromMemoryBuffer * getImageLayer() { return imagelayer; }
   double getTimeInterval() { return timeInterval; }
   
protected:
   int initialize(PV::HyPerCol * hc, char const * imagelayername, double simTimeInterval);
   IObjectDetector();

private:
   int initialize_base();

// Member variables
private:
   PV::HyPerCol * hypercolumn;
   PV::ImageFromMemoryBuffer * imagelayer;
   double timeInterval;

}; // class IObjectDetector
 
}  // namespace vidint

#endif // IOBJECTDETECTOR_HPP_
