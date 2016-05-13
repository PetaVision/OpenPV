#ifndef _BBFIND_HPP_
#define _BBFIND_HPP_

#include <vector>
#include <list>

using ::std::vector;
using ::std::list;
   
class BBFind
{
   public:
   
      class Rectangle
      {
         public:
            int x,  y, width, height;
            int left()   const { return x - width / 2; }
            int right()  const { return x + width / 2; }
            int top()    const { return y - height/ 2; }
            int bottom() const { return y + height/ 2; }
            static Rectangle  join        (const Rectangle &rectA, const Rectangle &rectB);
            static float      intersecting(const Rectangle &rectA, const Rectangle &rectB);
            static bool       touching    (const Rectangle &rectA, const Rectangle &rectB);
            static bool       equal       (const Rectangle &rectA, const Rectangle &rectB);
            
      };
      
      typedef vector< vector< vector<float> > > Map3;       // [feature][y][x]
      typedef vector< vector<float> >           Map2;       // [y][x]
      typedef vector< list<Rectangle> >         Rectangles; // [feature][rectangles]
   
      // Helper functions to pass in data from PetaVision
      static Map3 bufferToMap3(const float *buffer, int nx, int ny, int nf);
      static Map3 extendedBufferToMap3(const float *buffer, int nx, int ny, int nf, int lt, int rt, int dn, int up);
      
   private:

      // The most recent and next most recent confidence maps handed to us.
      // If framesPerMap > 1, we interpolate between these and store the
      // result in currentConfMap. Otherwise, currentConfMap = interpNextMap.
      Map3 mInterpPreviousMap;
      Map3 mInterpNextMap;
      
      Map3 mAccumulatedConfMap;
      Map3 mCurrentConfMap;
      Map3 mDistMap;
      Rectangles mDetections;
      
      // Previous rectangles for each category
      // [category][rectIndex][width=0 height=1]
      Map3 mRectSizesPerCategory;
      
      float mThreshold = 0.95f;
      float mContrast = 1.5f;
      float mContrastStrength = 1.0f;
      float mPrevInfluence = 1.5f;
      float mAccumulateAmount = 0.05f;
      float mPrevLeakTao = 16.0f;

      int mOriginalConfidenceWidth = -1;
      int mOriginalConfidenceHeight = -1;
      int mInternalConfidenceWidth = 64;
      int mInternalConfidenceHeight = 64;
      int mImageWidth = 256;
      int mImageHeight = 256;
      int mMaxRectangleMemory = 64;
      int mMinBlobSize = 16;
      int mBBGuessSize = 24;
      int mSlidingAverageSize = 4;
      int mFramesPerMap = 1;
      int mFramesSinceNewMap = 0;
      int mDetectionWait = 0;
      int mDetectionWaitTimer = 0;
      
      // Helper function to index PetaVision buffers
      static int bufferIndexFromCoords(int x, int y, int f, int nx, int ny, int nf);
      
   protected:
   
      // Map2 Functions
      Map2 scale(const Map2 &source, int newWidth, int newHeight, bool linear);
      Map2 applyThreshold(const Map2 confMap, float threshold);
      float sigmoidedRMS(const Map2 confMap, const Rectangle &bounds);
      Map2 makeEdgeDistanceMap(const Map2 confMap);
      void squash(Map2 &map, float scaleMin, float scaleMax);

      // Map3 Functions
      Map3 getInterpolatedConfs(int framesSinceNewMap);
      Map3 scale(const Map3 &source, int newWidth, int newHeight, bool linear);
      Map3 increaseContrast(const Map3 fullMap, float contrast, float strength);
      Map3 contrastAndAverage(const Map3 fullMap, float contrast, float strength);
      Map3 blendMaps(const Map3 &mapA, const Map3 &mapB, float interp);
      Map3 sumMaps(const Map3 &mapA, const Map3 &mapB, float scale);
      void clip(Map3 &confMap, float minVal, float maxVal);
      void squash(Map3 &map, float scaleMin, float scaleMax);
      void accumulateIntoPrev(Map3 &prevMap, const Map3 &currentMap, float accumulateAmt, float frameMemory, float scaleMin, float scaleMax);
      void clipSquash(Map3 &map, int numPasses, float initialMax);
      
      // Bounding box functions
      Rectangles placePotentialBoxes(const Map3 fullMap);
      void joinBoundingBoxes(Rectangles &boundingBoxes);
      void smoothBoundingBoxes(Rectangles &boundingBoxes);
      void competeBoundingBoxes(Rectangles &boundingBoxes, float maxIntersectAllowed);
   
   public:
   
      // Every time we have a new confidence map, hand it over with giveMap()
      void giveMap(Map3 newMap);
      
      // Performs one timestep of detection.
      // The main functionality of this class is performed here.
      // If mFramesPerMap > 1, this should be called mFramesPerMap times
      // inbetween giveMap() calls.
      void detect();
      
      // Clears previous values and buffers to initial state.
      // Use when switching videos or images.
      void reset();
      
      // After calling detect(), these functions can retreive the results.
      const Rectangles getDetections() { return mDetections; }
      const Map3 getConfMap() { return mCurrentConfMap; }
      const Map3 getDistMap() { return mDistMap; }
      
      //These return the maps scaled to the image size for easy overlapping
      const Map3 getScaledConfMap() { return scale(mCurrentConfMap, mImageWidth, mImageHeight, true); }
      const Map3 getScaledDistMap() { return scale(mDistMap, mImageWidth, mImageHeight, true); }
      
      // Setters (are getters necessary?)
   
      // How many frames of video do we interpolate between before receiving
      // a new confidence map? If we advance past framesPerMap frames without
      // new data, we continue to output the final result until new data
      // arrives.
      // Default is 1 for single-frame mode (no video), 16 for video.
      void setFramesPerMap(int framesPerMap) { mFramesPerMap = framesPerMap; }
      
      // Value from 0-1 indicating cut-off threshold for post-processed
      // confidence map. Values can saturate based on contrast, so a
      // threshold very close to 1.0 is recommended. Default is 0.95.
      void setThreshold(float threshold) { mThreshold = threshold; }
      
      // Exponent for the last step in the increaseContrast. Larger values
      // cause the output to saturate more easily. Default is 1.5.
      void setContrast(float contrast) { mContrast = contrast; }
      
      // Blend factor for contrast values. At 0, increaseContrast does not
      // modify values. At 1.0, values are completely replaced with the
      // result of the contrast operation. Default is 1.0
      void setContrastStrength(float contrastStrength) { mContrastStrength = contrastStrength; }
      
      // How much the accumulated previous buffer should affect detections.
      // Setting this to 0 uses only the current frame to find objects.
      // Default is 1.5
      void setPrevInfluence(float prevInfluence) { mPrevInfluence = prevInfluence; }
      
      // How much the previous buffer should be affected by each new 
      // frame. Small values work best, as a significantly large input
      // can effectively erase the previous buffer's old contents.
      // Default is 0.05
      void setAccumulateAmount(float accumulateAmt) { mAccumulateAmount = accumulateAmt; }
      
      // How quickly the previous buffer leaks its contents.
      // Larger values mean slower leaking.
      // Default is 16.0
      void setPrevLeakTao(float prevLeakTao) { mPrevLeakTao = prevLeakTao; }
      
      // A blob of confidences above threshold must be at least
      // this size to be a potential detection.
      // Default is 16
      void setMinBlobSize(int minBlobSize) { mMinBlobSize = minBlobSize; }
      
      // When we have a potential detection, what's the smallest size
      // bounding box we guess it could have. These are then merged, so
      // it's rare to actually have a single box this small.
      // Default is 24
      void setBBGuessSize(int bbGuessSize) { mBBGuessSize = bbGuessSize; }
      
      // Determines the area we take each step of the sigmoidedRMS over.
      // Default is 4
      void setSlidingAverageSize(int slidingAverageSize) { mSlidingAverageSize = slidingAverageSize; }
      
      // How many previous rectangles to keep in memory for each category.
      // Used to push new rectangles towards the average size.
      // Default is 64
      void setMaxRectangleMemory(int maxRectangleMemory) { mMaxRectangleMemory = maxRectangleMemory; }
      
      // How many frames to wait before starting object detection.
      // This gives the accumulated previous buffer time to fill up
      // before we start labeling objects.
      // Default is 16 for video, 0 for signle frame.
      void setDetectionWait(int detectionWait) { mDetectionWait = detectionWait; }
      
      // What size the internal confidence buffer should be scaled to.
      // A slightly higher size than the original helps detect bounding
      // boxes more accurately.
      void setInternalMapSize(int width, int height)
      {
         mInternalConfidenceWidth = width;
         mInternalConfidenceHeight = height;
      }
      
      // What size is the image we're detecting objects in?
      // Bounding box results will be given in this space.
      void setImageSize(int width, int height)
      {
         mImageWidth = width;
         mImageHeight = height;
      }
};

#endif
