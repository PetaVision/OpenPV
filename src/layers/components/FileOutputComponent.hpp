#pragma once

#include "OutputComponent.hpp"
#include <utils/PVLog.hpp>

#include <string>
#include <vector>
#include <memory>
#include <climits>
#include <cfloat>

namespace PV
{
	class OutputComponent;
	
	class FileOutputComponent : public OutputComponent
	{
		public:
			virtual void initialize();
			virtual void transform();
			virtual void readFile(std::string fileName) = 0;
			virtual void scatterFileBuffer();
			//virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
		
		protected:
			//These could be public static with very few changes if they'd be useful elsewhere
			vector<pvdata_t> scale(vector<pvdata_t> bufferToScale, int sourceWidth, int sourceHeight, int newWidth, int newHeight);
			vector<pvdata_t> shift(vector<pvdata_t> bufferToShift, int sourceWidth, int sourceHeight, int xShift, int yShift);
			vector<pvdata_t> flip(vector<pvdata_t> bufferToFlip, int sourceWidth, int sourceHeight, bool xFlip, bool yFlip);
			vector<pvdata_t> crop(vector<pvdata_t> bufferToCrop, int sourceWidth, int sourceHeight, int leftCrop, int rightCrop, int topCrop, int bottomCrop);
			vector<pvdata_t> expand(vector<pvdata_t> bufferToExpand, int sourceWidth, int sourceHeight, int newWidth, int newHeight);
			//Helper function for indexing mFileBuffer
			int getBufferIndex(int x, int y, int width, int features);
			
			pvdata_t getFromFile(int x, int y, int f);
			
			//virtual void ioParam_fileName(enum ParamsIOFlag ioFlag);
			//virtual void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

			bool mFilelist = false;
			bool mShuffle = false;
			bool mLinearInterpolation = true;
			bool mAutoScale = true;
			bool mFillExtended = true;
			bool mXFlip = false;
			bool mYFlip = false;
			int mPlaylistIndex = 0;
			int mDisplayPeriod = 100;
			int mDisplayTimer = INT_MAX;
			int mFileWidth = 1;
			int mFileHeight = 1;
			int mXShift = 0;
			int mYShift = 0;
			std::vector<std::string> mFilenames;
			std::vector<pvdata_t> mFileBuffer;
	};
}


