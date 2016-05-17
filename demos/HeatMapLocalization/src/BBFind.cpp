//****************************************************************
// BBFind.cpp
//
// Class used to find object bounding boxes in a given confidence map.
// Can work on a single image or sequential frames (video).
// Results are better with video, as previous detections are recorded.
//
// Austin Thresher, 5-11-16
//****************************************************************

#include <cstdio>
#include "BBFind.hpp"

#include <cmath>

//********************************
// Detection Algorithm
//********************************


void BBFind::giveMap(Map3 newMap)
{
   if(mOriginalConfidenceWidth == -1 || mOriginalConfidenceHeight == -1 || mFramesPerMap == 1)
   {
      mOriginalConfidenceWidth = newMap[0][0].size();
      mOriginalConfidenceHeight = newMap[0].size();

      mInterpPreviousMap = newMap;
      mInterpNextMap = newMap;
   }
   else
   {
      mInterpPreviousMap = mInterpNextMap;
      mInterpNextMap = newMap;   
   }
   mFramesSinceNewMap = 0;
}

void BBFind::detect()
{
   Map3 interpMap = getInterpolatedConfs(mFramesSinceNewMap++);

   int numCategories = interpMap.size();

   mCurrentConfMap.resize(numCategories);
   mDistMap.resize(numCategories);

   // Scale maps to our internal buffer size
   mCurrentConfMap = contrastAndAverage(
                        scale(interpMap, mInternalConfidenceWidth, mInternalConfidenceHeight, true),
                        mContrast,
                        mContrastStrength);

   // Apply light thresholding to remove noise
   for(int c = 0; c < numCategories; c++)
   {
      mCurrentConfMap[c] = applyThreshold(mCurrentConfMap[c], mThreshold / 3.0f);
   } 

   if(mPrevInfluence > 0)
   {
      mCurrentConfMap = sumMaps(mCurrentConfMap, mAccumulatedConfMap, mPrevInfluence);
      accumulateIntoPrev(mAccumulatedConfMap, mCurrentConfMap, mAccumulateAmount, mPrevLeakTau, -0.5f, 1.0f);
   }

   // Our values are all over the place right now,
   // so bring them back to 0 - 1 by alternating
   // a clip and squash, starting at a max of 3.5
   // and gradually reaching 1.0 over 8 iterations
   clipSquash(mCurrentConfMap, 8, 3.5f);

   // Apply our threshold and generate our distance map,
   // thresholded to the minimum blob size
   for(int c = 0; c < numCategories; c++)
   {
      mCurrentConfMap[c] = applyThreshold(mCurrentConfMap[c], mThreshold);
      mDistMap[c] = applyThreshold(
                        makeEdgeDistanceMap(
                           scale(mCurrentConfMap[c], mImageWidth, mImageHeight, true)
                           ),
                        mMinBlobSize);
   }

   // Clip our distance map so that regions
   // 2*minBlobSize pixels into an object are
   // saturated
   clip(mDistMap, 0.0f, mMinBlobSize * 2);

   // If mDetectionWait > 0, we wait a few frames
   // before generating bounding boxes. This allows us
   // to use previous frames for initial detections.
   if(mDetectionWaitTimer >= mDetectionWait)
   {
      // Detect potential boxes
      mDetections = placePotentialBoxes(mDistMap);
      // Join boxes that touch
      joinBoundingBoxes(mDetections);
      // Smooth box sizes with historical averages and join again
      smoothBoundingBoxes(mDetections);
      // Remove boxes that overlap by more than 75%
      competeBoundingBoxes(mDetections, 0.75f);
   }
   else mDetectionWaitTimer++;

   // Detection is finished. Use getDetections, getConfMap,
   // and getDistMap to retrieve the results.
}

void BBFind::reset()
{
   mCurrentConfMap.clear();
   mAccumulatedConfMap.clear();
   mRectSizesPerCategory.clear();
   mDistMap.clear();
   mDetections.clear();

   mDetectionWaitTimer = 0;
   mFramesSinceNewMap = 0;
   mOriginalConfidenceWidth = -1;
   mOriginalConfidenceHeight = -1;

}

//********************************
// Helper Functions
//********************************

int BBFind::bufferIndexFromCoords(int x, int y, int f, int nx, int ny, int nf)
{
   // Based on kIndex from conversions.h
   return f + (x + y * nx) * nf;
}

BBFind::Map3 BBFind::bufferToMap3(const float *bufferStart, int nx, int ny, int nf, int const * displayedCategories, int numDisplayedCategories)
{
   // This takes a pointer to a raw float buffer and converts
   // it to our handy dandy 3 dimensional float vector, Map3.
   // Intented to bridge the gap to PetaVision.

   Map3 result(numDisplayedCategories);
   for(int idx = 0; idx < numDisplayedCategories; idx++)
   {
      int const f = displayedCategories[idx]-1;
      result[idx].resize(ny);
      for(int y = 0; y < ny; y++)
      {
         result[idx][y].resize(nx);
         for(int x = 0; x < nx; x++)
         {
            result[idx][y][x] = bufferStart[bufferIndexFromCoords(x, y, f, nx, ny, nf)];
         }
      }
   }
   return result;
}

BBFind::Map3 BBFind::extendedBufferToMap3(const float *bufferStart, int nx, int ny, int nf, int lt, int rt, int up, int dn, int const * displayedCategories, int numDisplayedCategories)
{
   // This takes a pointer to an extended float buffer and converts
   // it to our handy dandy 3 dimensional float vector, Map3.
   // Intented to bridge the gap to PetaVision.
   // Based on kIndexExtended from conversions.h

   Map3 result(numDisplayedCategories);
   for(int idx = 0; idx < numDisplayedCategories; idx++)
   {
      int const f = displayedCategories[idx]-1;
      result[idx].resize(ny);
      for(int y = 0; y < ny; y++)
      {
         result[idx][y].resize(nx);
         for(int x = 0; x < nx; x++)
         {
            int k = bufferIndexFromCoords(x, y, f, nx, ny, nf);
            int kx_ex = (k/nf) % nx;
            int ky_ex = k / (nx*nf) % ny;
            int kf = k % nf;
            result[idx][y][x] = bufferStart[bufferIndexFromCoords(kx_ex, ky_ex, kf, nx + lt + rt, ny + dn + up, nf)];
         }
      }
   }
   return result;
}

BBFind::Map3 BBFind::bufferToMap3(const float *bufferStart, int nx, int ny, int nf)
{
   int displayedCategories[nf];
   for (int k=0; k<nf; k++) { displayedCategories[k]=k+1; }
   return bufferToMap3(bufferStart, nx, ny, nf, displayedCategories, nf);
}

BBFind::Map3 BBFind::extendedBufferToMap3(const float *bufferStart, int nx, int ny, int nf, int lt, int rt, int up, int dn)
{
   int displayedCategories[nf];
   for (int k=0; k<nf; k++) { displayedCategories[k]=k+1; }
   return extendedBufferToMap3(bufferStart, nx, ny, nf, lt, rt, up, dn, displayedCategories, nf);
}

void BBFind::clipSquash(Map3 &map, int numPasses, float initialMax)
{
   // This gently reigns in the higher values while minimizing clipping.
   // Brings values into the range 0 - 1

   float inc = (initialMax - 1.0f) / (numPasses * 2);
   for(int p = 0; p < numPasses; p++)
   {
      clip(map, 0.0f, initialMax);
      initialMax -= inc;
      squash(map, 0.0f, initialMax);
      initialMax -= inc;
   }
}

//********************************
// Rectangle Functions
//********************************

BBFind::Rectangle BBFind::Rectangle::join(const Rectangle &rectA, const Rectangle &rectB)
{
   // Returns the smallest rectangle that contains rectA and rectB

   int left   = std::min(rectA.left(),   rectB.left());
	int top    = std::min(rectA.top(),    rectB.top());
	int right  = std::max(rectA.right(),  rectB.right());
	int bottom = std::max(rectA.bottom(), rectB.bottom());
	return {(right + left) / 2, (bottom + top) / 2, right - left, bottom - top};
}

float BBFind::Rectangle::intersecting(const Rectangle &rectA, const Rectangle &rectB)
{
   // Returns a ratio representing how much of the smaller
   // rectangle is inside the larger rectangle.

   if(!touching(rectA, rectB)) return 0.0f;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	// Check if one completely contains another
   if(areaA > areaB)
	{
		if(rectB.left()   >= rectA.left() &&
			rectB.right()  <= rectA.right() &&
			rectB.top()    >= rectA.top() &&
			rectB.bottom() <= rectA.bottom())
         return 1.0f;
	}
	else
	{
		if(rectA.left()   >= rectB.left() &&
			rectA.right()  <= rectB.right() &&
			rectA.top()    >= rectB.top() &&
			rectA.bottom() <= rectB.bottom())
         return 1.0f;
	}
	// Find intersection rectangle
	int left   = std::max(rectA.left(),   rectB.left());
	int top    = std::max(rectA.top(),    rectB.top());
	int right  = std::min(rectA.right(),  rectB.right());
	int bottom = std::min(rectA.bottom(), rectB.bottom());
	float intersectArea = (right - left) * (bottom - top);
	if(areaA > areaB) return intersectArea / areaB;
	return intersectArea / areaA;
}

bool BBFind::Rectangle::touching(const Rectangle &rectA, const Rectangle &rectB)
{
   	return !(
		rectB.left()   > rectA.right()  ||
		rectB.right()  < rectA.left()   ||
		rectB.top()    > rectA.bottom() ||
		rectB.bottom() < rectA.top());
}

bool BBFind::Rectangle::equal(const Rectangle &rectA, const Rectangle &rectB)
{
  	return rectA.x == rectB.x && rectA.y == rectB.y && rectA.width == rectB.width && rectA.height == rectB.height;
}


//********************************
// Map2 Functions
//********************************

BBFind::Map2 BBFind::scale(const Map2 &source, int newWidth, int newHeight, bool bilinear)
{
   // Rescales the map's dimensions using either
   // bilinear interpolation or nearest neighbor.

   Map2 result(newHeight);
   int sourceWidth = source[0].size();
   int sourceHeight = source.size();

   if(bilinear) // Bilinear scaling
   {
      for(int j = 0; j < newHeight; j++)
      {
         result[j].resize(newWidth);
         for(int i = 0; i < newWidth; i++)
         {
            float xSource = i / (float)(newWidth-1) * (sourceWidth-1);
            float ySource = j / (float)(newHeight-1) * (sourceHeight-1);

            int leftIndex   = (int)xSource;
            int rightIndex  = (int)ceil(xSource);
            int topIndex    = (int)ySource;
            int bottomIndex = (int)ceil(ySource);

            if(topIndex < 0)  topIndex = 0;
            if(leftIndex < 0) leftIndex = 0;
            if(rightIndex >= sourceWidth)   rightIndex = sourceWidth-1;
            if(bottomIndex >= sourceHeight) bottomIndex = sourceHeight-1;

            float xAlign = xSource - leftIndex;
            float yAlign = ySource - topIndex;

            float tl = source[topIndex][leftIndex]     * (1.0f - xAlign) * (1.0f - yAlign);
            float tr = source[topIndex][rightIndex]    * xAlign * (1.0f - yAlign);
            float bl = source[bottomIndex][leftIndex]  * (1.0f - xAlign) * yAlign;
            float br = source[bottomIndex][rightIndex] * xAlign * yAlign;
            result[j][i] = tl+tr+bl+br;
         }
      }
   }
   else // Nearest neighbor scaling, no interpolation
   {
      float xRatio = sourceWidth / (float)(newWidth);
      float yRatio = sourceHeight / (float)(newHeight);

      #ifdef PV_USE_OPENMP_THREADS
			#pragma omp parallel for
		#endif
      for(int j = 0; j < newHeight; j++)
      {
         result[j].resize(newWidth);
         for(int i = 0; i < newWidth; i++)
         {
            result[j][i] = source[(int)(j*yRatio)][(int)(i*xRatio)];
         }
      }
   }
   return result;
}

BBFind::Map2 BBFind::applyThreshold(const Map2 confMap, float threshold)
{
   // Clips any values below threshold
   // TODO: Can this just modify the map in-place?
	int mapWidth = confMap[0].size();
	int mapHeight = confMap.size();
	Map2 resultMap = confMap;

   #ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
   #endif
   for(int x = 0; x < mapWidth; x++)
	{
		for(int y = 0; y < mapHeight; y++)
		{
         resultMap[y][x] = (confMap[y][x] >= threshold ? confMap[y][x] : 0.0f);
		}
	}
	return resultMap;
}

float BBFind::sigmoidedRMS(const Map2 confMap, const Rectangle &bounds)
{
   // Takes the RMS of the given sub area and applies a sigmoid to
   // smoothly cap values at 1.0

	int mapWidth = confMap[0].size();
	int mapHeight = confMap.size();
	float sum = 0.0f;
	for(int x = 0; x < bounds.width; x++)
	{
		for(int y = 0; y < bounds.height; y++)
		{
			int _x = bounds.left() + x;
			int _y = bounds.top()  + y;
			if(_x < 0 || _x >= mapWidth || _y < 0 || _y >= mapHeight) continue;
			sum += pow(confMap[_y][_x], 2);
		}
	}
	float avg = sqrt(sum / (mapWidth*mapHeight));
	float e = exp(1);
	// This is a very easy curve that still soft caps at 1.0.
	// Combined with the increaseContrast function, it allows
   // values > 0.85 to bring the average up based on the contrast argument.
	return  (1.0f / (1.0f + pow(e,-e*avg)) - 0.5f) * 2.0f;
}

BBFind::Map2 BBFind::makeEdgeDistanceMap(const Map2 confMap)
{
   // Modified Dijkstra map algorithm based on:
   // http://www.roguebasin.com/index.php?title=The_Incredible_Power_of_Dijkstra_Maps
   // The original algorithm finds nearest distance to a goal.
   // This modified version finds the larger distance, horizontal or vertical,
   // to a 0 confidence value (used to find object edges after threshold clipping)

	int mapWidth = confMap[0].size();
	int mapHeight = confMap.size();
	int maxVal = std::max(mapWidth, mapHeight);

   Map2 horizMap(mapHeight);
   Map2 vertMap(mapHeight);

   for(int y = 0; y < mapHeight; y++)
	{
      horizMap[y].resize(mapWidth);
      vertMap[y].resize(mapWidth);
		for(int x = 0; x < mapWidth; x++)
		{
         if(confMap[y][x] > 0)
         {
            horizMap[y][x] = maxVal;
            vertMap[y][x] = maxVal;
         }
      }
   }

   // Finds the shortest distance to 0 conf value in horizontal and vertical direction,
   // and stores the biggest one into result. This allows long, thin confidence chunks
   // to avoid clipping

   bool changed = true;
   while(changed)
   {
      changed = false;
      for(int y = 1; y < mapHeight-1; y++)
      {
         for(int x = 1; x < mapWidth-1; x++)
         {
            float lowest = horizMap[y][x];
            lowest = std::min(lowest, horizMap[y][x-1]);
            lowest = std::min(lowest, horizMap[y][x+1]);

            if(horizMap[y][x] > lowest+1)
            {
               horizMap[y][x] = lowest + 1;
               changed = true;
            }
         }
      }
   }

   changed = true;
   while(changed)
   {
      changed = false;
      for(int y = 1; y < mapHeight-1; y++)
      {
         for(int x = 1; x < mapWidth-1; x++)
         {
            float lowest = vertMap[y][x];
            lowest = std::min(lowest, vertMap[y-1][x]);
            lowest = std::min(lowest, vertMap[y+1][x]);

            if(vertMap[y][x] > lowest+1)
            {
               vertMap[y][x] = lowest + 1;
               changed = true;
            }
         }
      }
   }

   Map2 resultMap = horizMap;

   #ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
   #endif
   for(int y = 0; y < mapHeight; y++)
	{
		for(int x = 0; x < mapWidth; x++)
		{
         if(vertMap[y][x] > resultMap[y][x])
         {
            resultMap[y][x] = vertMap[y][x];
         }
      }
   }

	return resultMap;
}

void BBFind::squash(Map2 &map, float scaleMin, float scaleMax)
{
   // Overloaded to work on Map2 or Map3. Normalizes, but sets
   // initial min / max values to given arguments. If the given
   // range is larger than the data's range, the data is scaled
   // down proportionally. Otherwise, it just normalizes to the
   // given range.

   int mapWidth = (int)map[0].size();
   int mapHeight = (int)map.size();

   float maxVal = scaleMax;
   float minVal = scaleMin;

   for(int x = 0; x < mapWidth; x++)
   {
      for(int y = 0; y < mapHeight; y++)
      {
         float val = map[y][x];
         maxVal = val > maxVal ? val : maxVal;
         minVal = val < minVal ? val : minVal;
      }
   }
   float range = maxVal - minVal;
   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int x = 0; x < mapWidth; x++)
   {
      for(int y = 0; y < mapHeight; y++)
      {
         map[y][x] = scaleMin + ((map[y][x] - minVal) / range) * scaleMax;
      }
   }
}


//********************************
// Map3 Functions
//********************************

BBFind::Map3 BBFind::getInterpolatedConfs(int framesSinceNewMap)
{
   float interp = (float)framesSinceNewMap / mFramesPerMap;
   if(interp > 1.0f) interp = 1.0f;
   return blendMaps(mInterpPreviousMap, mInterpNextMap, interp);
}

BBFind::Map3 BBFind::scale(const Map3 &source, int newWidth, int newHeight, bool bilinear)
{
   // Rescales the map's dimensions using either
   // bilinear interpolation or nearest neighbor.
   // Overloaded for 3 dimensions

   Map3 result(source.size());
   for(int c = 0; c < result.size(); c++)
   {
      result[c] = scale(source[c], newWidth, newHeight, bilinear);
   }
   return result;
}

BBFind::Map3 BBFind::increaseContrast(const Map3 fullMap, float contrast, float strength)
{
   // Increases the contrast in the map and blends the result with the
   // original map, using strength as the blend factor.

   int numCategories = fullMap.size();
   int mapWidth = fullMap[0][0].size();
   int mapHeight = fullMap[0].size();
   Map3 resultMap = fullMap;

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   FILE * devnull = fopen("/dev/null", "w");
   for(int c = 0; c < numCategories; c++)
   {
      for(int y = 0; y < mapHeight; y++)
      {
         for(int x = 0; x < mapWidth; x++)
         {
            // This makes values "break even" at val = 0.85 and
            // pass 1.0 at val = 0.9. How far it passes 1 depends
            // on contrast.
            float val = fullMap[c][y][x];
            fprintf(devnull, "%f\n", val);
            fprintf(devnull, "%f\n", contrast);
            resultMap[c][y][x] = val * (1.0f - strength) + pow(pow(50.0f, val) / 33.3f, contrast) * strength;
         }
      }
   }
   fclose(devnull);

   return resultMap;
}

BBFind::Map3 BBFind::contrastAndAverage(const Map3 fullMap, float contrast, float strength)
{
   // Applies increaseContrast and sigmoidedRMS to the given map

	int numCategories = fullMap.size();
	int mapWidth = fullMap[0][0].size();
	int mapHeight = fullMap[0].size();
	Rectangle bounds = {0, 0, mSlidingAverageSize, mSlidingAverageSize};
   Map3 curvedMap = increaseContrast(fullMap, contrast, strength);
   Map3 resultMap(numCategories);

   float mapScaleFactor = sqrt(
      ((float)mInternalConfidenceWidth * mInternalConfidenceHeight)
           / (mOriginalConfidenceWidth * mOriginalConfidenceHeight));

   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      resultMap[c].resize(mapHeight);
      for(int y = 0; y < mapHeight; y++)
      {
         resultMap[c][y].resize(mapWidth);
         for(int x = 0; x < mapWidth; x++)
         {
            bounds.x = x; bounds.y = y;
				resultMap[c][y][x] = sigmoidedRMS(curvedMap[c], bounds) * 10.0f * mapScaleFactor;
			}
		}
	}

	return resultMap;
}

BBFind::Map3 BBFind::blendMaps(const Map3 &mapA, const Map3 &mapB, float interp)
{
   // Returns a blend of mapA and mapB.
   // interp = 0 returns mapA, interp = 1 returns mapB
   if(mapB.empty()) return mapA;

   int numCategories = mapA.size();
   int mapWidth = mapA[0][0].size();
   int mapHeight = mapA[0].size();

   Map3 interpolatedMap(numCategories);

   #ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      interpolatedMap[c].resize(mapHeight);
      for(int y = 0; y < mapHeight; y++)
      {
         interpolatedMap[c][y].resize(mapWidth);
         for(int x = 0; x < mapWidth; x++)
         {
            float val = mapA[c][y][x];
            float next = mapB[c][y][x];
            interpolatedMap[c][y][x] = val * (1.0f - interp) + next * interp;
         }
      }
   }

   return interpolatedMap;
}


BBFind::Map3 BBFind::sumMaps(const Map3 &mapA, const Map3 &mapB, float scale)
{
   // Returns mapA + (mapB * scale)

   int numCategories = mapA.size();
   int mapWidth = mapA[0][0].size();
   int mapHeight = mapA[0].size();

   if(mapB.empty()) return mapA;

   Map3 summedMap(numCategories);

   #ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      summedMap[c].resize(mapHeight);
      for(int y = 0; y < mapHeight; y++)
      {
         summedMap[c][y].resize(mapWidth);
         for(int x = 0; x < mapWidth; x++)
         {
            float val = mapA[c][y][x];
            float add = mapB[c][y][x];
            summedMap[c][y][x] = val + add * scale;
         }
      }
   }

   return summedMap;
}

void BBFind::clip(Map3 &confMap, float minVal, float maxVal)
{
   // Clips the values in the map to the given min and max

   int numCategories = confMap.size();
	int mapHeight = confMap[0].size();
   int mapWidth = confMap[0][0].size();
   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      for(int y = 0; y < mapHeight; y++)
      {	
         for(int x = 0; x < mapWidth; x++)
         {
            confMap[c][y][x] = std::min(maxVal, std::max(minVal, confMap[c][y][x]));
         }
      }
   }
}

void BBFind::squash(Map3 &map, float scaleMin, float scaleMax)
{
   // Overloaded to work on Map2 or Map3. Normalizes, but sets
   // initial min / max values to given arguments. If the given
   // range is larger than the data's range, the data is scaled
   // down proportionally. Otherwise, it just normalizes to the
   // given range.

   int numCategories = (int)map.size();
   int mapWidth = (int)map[0][0].size();
   int mapHeight = (int)map[0].size();

   float maxVal = scaleMax;
   float minVal = scaleMin;

   for(int c = 0; c < numCategories; c++)
   {
      for(int y = 0; y < mapHeight; y++)
      {
         for(int x = 0; x < mapWidth; x++)
         {
            float val = map[c][y][x];
            maxVal = val > maxVal ? val : maxVal;
            minVal = val < minVal ? val : minVal;
         }
      }
   }
   float range = maxVal - minVal;
   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      for(int y = 0; y < mapHeight; y++)
      {
         for(int x = 0; x < mapWidth; x++)
         {
            map[c][y][x] = scaleMin + ((map[c][y][x] - minVal) / range) * scaleMax;
         }
      }
   }
}

void BBFind::accumulateIntoPrev(Map3 &prevMap, const Map3 &currentMap, float accumulateAmt, float frameMemory, float scaleMin, float scaleMax)
{
   // This is the entire update step for the previous confidence values
   // buffer. We add currentMap * accumulateAmt to the prevMap buffer,
   // then decay it by e ^ (-1 / frameMemory). Lastly, we clip any values
   // that have grown too large and squash the values back to an acceptable
   // range. This ends up functioning as a form of competition, as large
   // new values cause older values to be squashed.

   int numCategories = currentMap.size();
	int mapWidth = currentMap[0][0].size();
	int mapHeight = currentMap[0].size();
   float fadeFactor = exp(-1.0f / frameMemory);
   if(prevMap.empty())
   {
      prevMap = currentMap;
      for(int c = 0; c < numCategories; c++)
      {
         for(int y = 0; y < mapHeight; y++)
         {
            for(int x = 0; x < mapWidth; x++)
            {
               prevMap[c][y][x] = scaleMin;
            }
         }
      }
   }
   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      for(int y = 0; y < mapHeight; y++)
      {
         for(int x = 0; x < mapWidth; x++)
         {
            float val = (prevMap[c][y][x] + currentMap[c][y][x] * accumulateAmt) * fadeFactor;
            if(val > scaleMax) val = scaleMax + (val - scaleMax) * accumulateAmt; //hard knee
            prevMap[c][y][x] = val;
         }
      }
   }
   clip(prevMap, scaleMin, scaleMax * 1.5f);
   squash(prevMap, scaleMin, scaleMax);
}


//********************************
// Bounding Box Functions
//********************************

BBFind::Rectangles BBFind::placePotentialBoxes(const Map3 fullMap)
{
   // Takes in a map of distances to edges and greedily places
   // a potential box at every local maximum. Because the distance
   // map is clipped, this should place numerous small boxes
   // inside the center of detected objects above the minimum size.
   // These boxes are later merged to form a single larger
   // bounding box for each object.

   int numCategories = fullMap.size();
	int mapWidth = fullMap[0][0].size();
	int mapHeight = fullMap[0].size();
	float val = 0.0f;
	float maxVal = 0.0f;
	Rectangles boundingBoxes(numCategories);

	const int stride = mBBGuessSize / 8;

   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int c = 0; c < numCategories; c++)
   {
      Map2 distanceMap = applyThreshold(fullMap[c], mMinBlobSize);
      for(int y = 1; y < mapHeight-1; y+=stride)
		{
         for(int x = 1; x < mapWidth-1; x+=stride)
         {
				//Look at neighboring values to deduce if this is a local maximum
				maxVal = distanceMap[y][x];
				for(int i = -1; i <= 1; i++)
				{
					for(int j = -1; j <= 1; j++)
					{
						val = distanceMap[y+j][x+i];
						maxVal = val > maxVal ? val : maxVal;
					}
				}
				//If this was a local maximum, place a potential bounding box
				if(maxVal > 0 && maxVal == distanceMap[y][x])
				{
					boundingBoxes[c].push_back({x, y, mBBGuessSize, mBBGuessSize});
				}
			}
		}
	}

	return boundingBoxes;
}

void BBFind::joinBoundingBoxes(Rectangles &boundingBoxes)
{
   // Within each category, find any touching bounding boxes and merge them.
   // Repeat until no merges were made.

	int numCategories = boundingBoxes.size();

   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
	for(int c = 0; c < numCategories; c++)
	{
		bool joinedBox = true;
		while(joinedBox)
		{
			joinedBox = false;
			for(auto a = boundingBoxes[c].begin(); a != boundingBoxes[c].end(); a++)
			{
				for(auto b = boundingBoxes[c].begin(); b != boundingBoxes[c].end(); b++)
				{
               if(a == b) continue;
					if(Rectangle::touching(*a, *b))
					{
                  Rectangle result = Rectangle::join(*a, *b);
                  a->x = result.x;
                  a->y = result.y;
                  a->width = result.width;
                  a->height = result.height;
                  boundingBoxes[c].erase(b);
						joinedBox = true;
						break;
					}
				}
				if(joinedBox) break;
			}
		}
	}
}

void BBFind::smoothBoundingBoxes(Rectangles &boundingBoxes)
{
   // Averages given bounding boxes with previous boxes of the same
   // category, then attempt to merge any boxes that may now overlap.
   // Store the results in a running average to affect future boxes.

	int numCategories = boundingBoxes.size();
   if(mRectSizesPerCategory.size() < numCategories)
      mRectSizesPerCategory.resize(numCategories);

   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
	for(int c = 0; c < numCategories; c++)
	{
      if(mRectSizesPerCategory[c].size() > 0)
      {
         float avgW = 0.0f;
         float avgH = 0.0f;
         for(int r = 0; r < mRectSizesPerCategory[c].size(); r++)
         {
            avgW += mRectSizesPerCategory[c][r][0];
            avgH += mRectSizesPerCategory[c][r][1];
         }
         avgW /= mRectSizesPerCategory[c].size();
         avgH /= mRectSizesPerCategory[c].size();
         for(auto a : boundingBoxes[c])
         {
            a.width = (int)((a.width + avgW) / 2);
            a.height = (int)((a.height + avgH) / 2);
         }
      }
   }

   joinBoundingBoxes(boundingBoxes);

   for(int c = 0; c < numCategories; c++)
	{
      while(mRectSizesPerCategory[c].size() > mMaxRectangleMemory)
      {
         mRectSizesPerCategory[c].erase(mRectSizesPerCategory[c].begin());
      }
      for(auto a : boundingBoxes[c])
      {
         mRectSizesPerCategory[c].push_back({(float)a.width, (float)a.height});
      }
   }
}

void BBFind::competeBoundingBoxes(Rectangles &boundingBoxes, float maxIntersectAllowed)
{
	int numCategories = boundingBoxes.size();
	bool boxFight = true;
	while(boxFight)
	{
		boxFight = false;
		for(int c1 = 0; c1 < numCategories; c1++)
		{
			for(auto a = boundingBoxes[c1].begin(); a != boundingBoxes[c1].end(); a++)
			{
				for(int c2 = 0; c2 < numCategories; c2++)
				{
               for(auto b = boundingBoxes[c2].begin(); b != boundingBoxes[c2].end(); b++)
               {
                  if(c1 == c2 || a == b) continue;
						if(Rectangle::intersecting(*a, *b) >= maxIntersectAllowed)
						{
							if(a->width * a->height > b->width * b->height)
							{
								boundingBoxes[c2].erase(b);
							}
							else
							{
								boundingBoxes[c2].erase(a);
							}
							boxFight = true;
							break;
						}
					}
					if(boxFight) break;
				}
				if(boxFight) break;
			}
			if(boxFight) break;
		}
	}
}







