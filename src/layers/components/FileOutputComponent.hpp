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
         ~FileOutputComponent();
         virtual void initialize();
			virtual void transform();
			virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
         virtual double getDeltaUpdateTime();
         virtual void updateFileBuffer(std::string fileName, std::vector<pvdata_t> &fileBuffer) = 0; 
      
		protected:
			//These could be public static with very few changes if they'd be useful elsewhere
			vector<pvdata_t> scale(vector<pvdata_t> bufferToScale, int sourceWidth, int sourceHeight, int newWidth, int newHeight);
			vector<pvdata_t> shift(vector<pvdata_t> bufferToShift, int sourceWidth, int sourceHeight, int xShift, int yShift);
			vector<pvdata_t> flip(vector<pvdata_t> bufferToFlip, int sourceWidth, int sourceHeight, bool xFlip, bool yFlip);
			vector<pvdata_t> crop(vector<pvdata_t> bufferToCrop, int sourceWidth, int sourceHeight, int left, int top, int width, int height);
			vector<pvdata_t> expand(vector<pvdata_t> bufferToExpand, int sourceWidth, int sourceHeight, int newWidth, int newHeight);
			
         //Helper function for indexing mFileBuffer
			int getBufferIndex(int x, int y, int width, int features);
         void scatterFileBuffer(int batchIndex);
         void readFileList(std::string fileName);

			virtual void ioParam_fileName(enum ParamsIOFlag ioFlag);
         virtual void ioParam_isFileList(enum ParamsIOFlag ioFlag);
			virtual void ioParam_updatePeriod(enum ParamsIOFlag ioFlag);
         virtual void ioParam_updatesPerFileAdvance(enum ParamsIOFlag ioFlag);
         virtual void ioParam_autoScale(enum ParamsIOFlag ioFlag);
         virtual void ioParam_linearInterpolation(enum ParamsIOFlag ioFlag);
         virtual void ioParam_fillExtended(enum ParamsIOFlag ioFlag);
         
			bool mIsFileList = false;
			bool mLinearInterpolation = true;
			bool mAutoScale = true;
			bool mFillExtended = true;
			bool mXFlip = false;
			bool mYFlip = false;
			int mFileListIndex = 0;
         int mUpdatesPerFileAdvance = 1;
         int mFileAdvanceTimer = 0;
			int mUpdatePeriod = 100;
         int mFileWidth = 1;
			int mFileHeight = 1;
			int mXShift = 0;
			int mYShift = 0;
         char *mCFileName = nullptr;
         std::string mFileName;
			std::vector<std::string> mFileList;
			std::vector< std::vector<pvdata_t> > mFileBuffers; //TODO: Batching should probably have a vector of these, one for each batch
	};
}


