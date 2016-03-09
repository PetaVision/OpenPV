/**
 * Base Input class for reading images, movies, pvp files, as well as creating patterns and loading from memory buffer.
 */
#ifndef BASEINPUT_HPP_
#define BASEINPUT_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"
#include "../columns/Random.hpp"
#include "../io/imageio.hpp"


namespace PV {

class BaseInput: public HyPerLayer{
protected:
   
   /** 
    * List of parameters needed from the BaseInput class
    * @name BaseInput Parameters
    * @{
    */

   /**
    * @brief inputPath: The file the input is reading. The type of file depends on which subclass is being used.
    */
   virtual void ioParam_inputPath(enum ParamsIOFlag ioFlag);

   /**
    * @brief offsetX: offset in X direction <br />
    * offsetY: offset in Y direction
    * @details Defines an offset in image space where the column is viewing the image
    */
   virtual int ioParam_offsets(enum ParamsIOFlag ioFlag); // reads offsetX, offsetY from params.  Override with empty function if a derived class doesn't use these parameters (e.g. Patterns)

   /**
    * @brief offsetAnchor: Defines where the anchor point is for the offsets.
    * @details Specified as a 2 character string, "xy" <br />
    * x can be 'l', 'c', or 'r' for left, center, right respectively <br />
    * y can be 't', 'c', or 'b' for top, center, bottom respectively <br />
    * Defaults to "tl"
    */
   virtual void ioParam_offsetAnchor(enum ParamsIOFlag ioFlag);

   /** 
    * @brief writeImages: A boolean flag that specifies if the Image class should output the images.
    */
   virtual void ioParam_writeImages(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeImageExtension: If writeImages is set, specifies the extention of the image output.
    * @details Defaults to .tif
    */
   virtual void ioParam_writeImagesExtension(enum ParamsIOFlag ioFlag);

   /**
    * @brief inverseFlag: If set to true, inverts the image
    */
   virtual void ioParam_inverseFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief normalizeLuminanceFlag: If set to true, will normalize the image.
    * The normalization method is determined by the normalizeStdDev parameter.
    */
   virtual void ioParam_normalizeLuminanceFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief normalizeStdDev: This flag is used if normalizeLuminanceFlag is true.
    * If normalizeStdDev is set to true, the image will normalize with a mean of 0 and std of 1
    * If normalizeStdDev is set to false, the image will normalize with a min of 0 and a max of 1
    */
   virtual void ioParam_normalizeStdDev(enum ParamsIOFlag ioFlag);

   /**
    * @brief jitterFlag: If set to true, will move the image around by specified pixels
    */
   virtual void ioParam_jitterFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief jitterType: If jitter flag is set, specifies the type of jitter. 0 for random walk, 1 for random jump
    * @details - Random Walk: Jitters the specified step in any direction
    * - Random Jump: Jitters any value between -step and step in any direction
    */
   virtual void ioParam_jitterType(enum ParamsIOFlag ioFlag);

   /**
    * @brief jitterRefactoryPeriod: If jitter flag is set, specifies the minimum amount of time until next jitter
    */
   virtual void ioParam_jitterRefractoryPeriod(enum ParamsIOFlag ioFlag);

   /**
    * @brief stepSize: If jitter flag is set, sets the step size
    */
   virtual void ioParam_stepSize(enum ParamsIOFlag ioFlag);

   /**
    * @brief persistenceProb: If jitter flag is set, sets the probability that offset stays the same
    */
   virtual void ioParam_persistenceProb(enum ParamsIOFlag ioFlag);

   /**
    * @brief recurrenceProb: If jitter flag is set, sets the probability that the offset returns to bias position
    */
   virtual void ioParam_recurrenceProb(enum ParamsIOFlag ioFlag);

   /**
    * @brief padValue: If the image is being padded (image smaller than layer), the value to use for padding
    */
   virtual void ioParam_padValue(enum ParamsIOFlag ioFlag);

   /**
    * @brief biasChangeTime: If jitter flag is set, sets the time period for recalculating bias position
    */
   virtual void ioParam_biasChangeTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief biasConstraintMethod: If jitter flag is set, defines the method to coerce into bounding box
    * @details Can be 0 (ignore), 1 (mirror BC), 2 (threshold), or 3 (circular BC)
    */
   virtual void ioParam_biasConstraintMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief offsetConstraintMethod: If jitter flag is set, defines the method to coerce into bounding box
    * @details Can be 0 (ignore), 1 (mirror BC), 2 (threshold), or 3 (circular BC)
    */
   virtual void ioParam_offsetConstraintMethod(enum ParamsIOFlag ioFlag);

   /**
    * @brief writePosition: If jitter flag is set, writes position to input/image-pos.txt
    */
   virtual void ioParam_writePosition(enum ParamsIOFlag ioFlag);

   /**
    * @brief initVType: Image does not have a V, do not set
    */
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: BaseInput and derived classes do not use triggering, and always set triggerLayerName to NULL.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   //TODO this functionality should be in both pvp and image. Set here for now, as pvp does not support imageBC
   /**
    * @brief useImageBCFlag: Specifies if the Image layer should use the image to fill margins 
    */
   virtual void ioParam_useImageBCflag(enum ParamsIOFlag ioFlag);

   /**
    * @}
    */

protected:
   BaseInput();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   int initRandState();

   virtual int allocateV();
   virtual int initializeV();
   virtual int initializeActivity();

   virtual bool jitter();
   virtual int calcBias(int current_bias, int step, int sizeLength);
   virtual int calcNewBiases(int stepSize);
   virtual int calcBiasedOffset(int bias, int current_offset, int step, int sizeLength);
   virtual bool calcNewOffsets(int stepSize);
   static bool constrainPoint(int * point, int min_x, int max_x, int min_y, int max_y, int method);
   virtual bool constrainBiases();
   virtual bool constrainOffsets();

   /**
    * This virtual function gets called from getFrame. Each subclass needs to overwrite this
    * function to define how to fill the activity buffer of the input class.
    */
   virtual int retrieveData(double timef, double dt) = 0;
   /**
    * This is the interface for loading a new "frame" (which can be either pvp, image, etc)
    * into the activity buffer. This function calls retrieveData.
    */
   virtual int getFrame(double timef, double dt);
   /**
    * This function achieves post processing of the activity buffer after a frame is loaded.
    */
   virtual int postProcess(double timef, double dt);

   /**
    * Returns PV_SUCCESS if offsetAnchor is a valid anchor string
    * (two characters long; first characters one of 't', 'c', or 'b'; second characters one of 'l', 'c', or 'r')
    * Returns PV_FAILURE otherwise
    */
   int checkValidAnchorString();

   int toGrayScale();

   int copyFromInteriorBuffer(float * buf, int batchIdx, float fac);
   int copyToInteriorBuffer(unsigned char * buf, int batchIdx, float fac);


public:
   BaseInput(const char * name, HyPerCol * hc);
   virtual ~BaseInput();
   //virtual int communicateInitInfo();
   virtual int requireChannel(int channelNeeded, int * numChannelsResult);
   virtual int allocateDataStructures();
   virtual int updateState(double time, double dt);

   //virtual bool needUpdate(double time, double dt);
   //virtual int updateState(double time, double dt);
   //virtual int outputState(double time, bool last=false);
   
   virtual int checkpointRead(const char * cpDir, double * timeptr);

   // partially override implementation of LayerDataInterface interface
   //
   const pvdata_t * getLayerData(int delay=0)   { return data; }

   virtual bool activityIsSpiking() {return false;}

   virtual PVLayerLoc getImageLoc() {return imageLoc; }

   int writeImage(const char * filename, int batchIdx);

   int exchange();

   int getOffsetX(const char* offsetAnchor, int offsetX);
   int getOffsetY(const char* offsetAnchor, int offsetY);

   /**
    * getImageStartX() returns the x-coordinate in the original input corresponding to x=0 in layer coordinates.
    */
   int getImageStartX() { return getOffsetX(offsetAnchor, offsets[0]); }

   /**
    * getImageStartY() returns the y-coordinate in the original input corresponding to y=0 in the layer coordinates.
    */
   int getImageStartY() { return getOffsetX(offsetAnchor, offsets[0]); }
   
   int getBiasX() { return biases[0]; }
   int getBiasY() { return biases[1]; }
   const int * getBiases() { return biases; }
   char * getInputPath(){return inputPath;}

   //virtual int scatterImageFile(const char * filename, int xOffset, int yOffset, PV::Communicator * comm, const PVLayerLoc * loc, float * buf, int frameNumber, bool autoResizeFlag, int batchIdx);

private:
   int initialize_base();

protected:
   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   pvdata_t * data;       // buffer containing reduced image

   PVLayerLoc imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image

   int writeImages;      // controls writing of image file during outputState
   char * writeImagesExtension; // ".pvp", ".tif", ".png", etc.; the extension to use when writing images
   // bool useGrayScale;    // whether to convert image to grayscale
   
   bool inverseFlag;
   bool normalizeLuminanceFlag; // if true, normalize the input image as specified by normalizeStdDev
   bool normalizeStdDev;        // if true and normalizeLuminanceFlag == true, normalize the standard deviation to 1 and mean = 0
                                // if false and normalizeLuminanceFlag == true, nomalize max = 1, min = 0
                                //
   // Jitter parameters
   int jitterFlag;        // If true, use jitter
   int stepSize;
   int biases[2];         // When jittering, the distribution for the offset is centered on offsetX=biasX, offsetY=biasY
                          // Jittering can change the bias point on a slower timescale than the offset point changes.
   int biasConstraintMethod;  // If biases escape the bounding box, the method to coerce them into the bounding box.
   int offsetConstraintMethod; // If offsets escape the bounding box, the method to coerce them into the bounding box.
                          // The constraint method codes are 0=ignore, 1=mirror boundary conditions, 2=thresholding, 3=circular boundary conditions
   float recurrenceProb;  // If using jitter, probability that offset returns to bias position
   float persistenceProb; // If using jitter, probability that offset stays the same
   int writePosition;     // If using jitter, write positions to input/image-pos.txt
   PV_Stream * fp_pos;    // If writePosition is true, write the positions to this file
   double biasChangeTime;    // If using jitter, time period for recalculating bias position
   double nextBiasChange;    // The next time biasChange will be called
   int jitterRefractoryPeriod; // After jitter, minimum amount of time until next jitter
   int timeSinceLastJitter; // Keeps track of timesteps since last jitter
   int jitterType;       // If using jitter, specify type of jitter (random walk or random jump)
   const static int RANDOM_WALK = 0;  // const denoting jitter is a random walk
   const static int RANDOM_JUMP = 1;  // const denoting jitter is a random jump

   int offsets[2];        // offsets array points to [offsetX, offsetY]
   char* offsetAnchor;

   float padValue;
   bool useImageBCflag;

   Random * randState;

   char* inputPath;

};
}

#endif

