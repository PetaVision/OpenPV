#include "FileOutputComponent.hpp"

namespace PV
{
   //Very WIP, probably does not work yet. Should build, but completely untested.

   void FileOutputComponent::initialize() 
	{
      bool iAmRoot = (mParentLayer->getParent()->icCommunicator()->commRank() == 0);
      if(iAmRoot)
      {
         if(mIsFileList)
         {
            std::ifstream fList(mFileName);
            if(!fList.is_open()) { pvError() << mParentLayer->getName() << ": Could not open file list: " << mFileName << std::endl; throw; }
            std::string path;
            mFileList.clear();
            mFileListIndex = 0;
            while(std::getline(fList, path)) { mFileList.push_back(path); }
            if(mFileList.size() <= 0) { pvError() << mParentLayer->getName() << ": File list is empty: " << mFileName << std::endl; throw; }
            pvInfo() << mParentLayer->getName() << ": Read " << mFileList.size() << " file list entries." << std::endl;
         }
         
         mFileBuffers.resize(mParentLayer->getParent()->getNBatch());
      }
   }
   
   FileOutputComponent::~FileOutputComponent()
   {
      if(mCFileName != nullptr) free(mCFileName);
   }
   
   double FileOutputComponent::getDeltaUpdateTime()
   {
      return mUpdatePeriod;
   }

   void FileOutputComponent::scatterFileBuffer(int batchIndex) 
	{
      //These used to be inside the commRank==0 if below, make sure it was ok to move up here
      vector<pvdata_t> fileSlice;
      int nX = mParentLayer->getLayerLoc()->nx;
      int nY = mParentLayer->getLayerLoc()->ny;
      int nF = mParentLayer->getLayerLoc()->nf;
      
      if(mFillExtended) //This is what Image.cpp was doing, but does this actually work with mpi?
      {
         nX += mParentLayer->getLayerLoc()->halo.lt+mParentLayer->getLayerLoc()->halo.rt;
         nY += mParentLayer->getLayerLoc()->halo.up+mParentLayer->getLayerLoc()->halo.dn;
      }

      MPI_Comm mpiComm = mParentLayer->getParent()->icCommunicator()->communicator();
      int nXProcs = mParentLayer->getParent()->icCommunicator()->numCommColumns();
      int nYProcs = mParentLayer->getParent()->icCommunicator()->numCommRows();
      int commRank = mParentLayer->getParent()->icCommunicator()->commRank();
      
      if(commRank == 0) //Root process, send data to other processes
      {
         int columnWidth = nX * nXProcs;
         int columnHeight = nY * nYProcs;

         vector<pvdata_t> preparedFile;

         if(mAutoScale) { preparedFile = scale(mFileBuffers[batchIndex], mFileWidth, mFileHeight, columnWidth, columnHeight); }
         else { preparedFile = expand(mFileBuffers[batchIndex], mFileWidth, mFileHeight, columnWidth, columnHeight); }
         
         if(mXFlip || mYFlip) { preparedFile = flip(preparedFile, mFileWidth, mFileHeight, mXFlip, mYFlip); }
         if(mXShift != 0 || mYShift != 0) { preparedFile = shift(preparedFile, mFileWidth, mFileHeight, mXShift, mYShift); }
       
         //Slice up the data and send it to each process
         for(int dest = nYProcs*nXProcs-1; dest >= 0; dest--)
         {
            int col = columnFromRank(dest, nYProcs, nXProcs);
            int row = rowFromRank(dest, nYProcs, nXProcs);
            int layerXStart = nX * col;
            int layerYStart = nY * row;
            fileSlice = crop(preparedFile, columnWidth, columnHeight, layerXStart, layerYStart, nX, nY);
            int dataSize = fileSlice.size();
            if(dest > 0) //Don't send to ourselves, just keep fileSlice
            {
               MPI_Send(&dataSize,  1,            MPI_INT,   dest, 0, mpiComm);
               MPI_Send(&fileSlice[0],  dataSize, MPI_FLOAT, dest, 1, mpiComm);
               pvDebug() << "Root sending " << dataSize << " floats to process " << dest << " from corner (" << layerXStart << ", " << layerYStart << ")" << std::endl;
            }
         }
      }
      else //We aren't root, receive data from root
      {
         int dataSize;
         MPI_Recv(&dataSize,     1,        MPI_INT,   0,    0, mpiComm, MPI_STATUS_IGNORE);
         pvDebug() << "Process " << commRank << " preparing to receive " << dataSize << " floats." << std::endl;
         fileSlice.resize(dataSize);
         MPI_Recv(&fileSlice[0], dataSize, MPI_FLOAT, 0,    1, mpiComm, MPI_STATUS_IGNORE);
         pvDebug() << "Process " << commRank << " received " << fileSlice.size() << " floats." << std::endl;
      }
      
      // fileSlice should now have the correct data regardless of which process we're in      
      pvdata_t * dataBatch = mTransformBuffer->data + batchIndex * mParentLayer->calcBatchOffset();
      if(mFillExtended)
      {
         for(int n = 0; n < mParentLayer->getNumExtended(); n++)
         {
            dataBatch[n] = fileSlice[n];
         }
      }
      else
      {
         const PVHalo * halo = &mParentLayer->getLayerLoc()->halo;
         for(int n = 0; n < mParentLayer->getNumNeurons(); n++)
         {
            int extendedIndex = kIndexExtended(n, nX, nY, nF, halo->lt, halo->rt, halo->dn, halo->up);
            dataBatch[extendedIndex] = fileSlice[n];
         }
      }
   }
   
   //This doesn't give you features- just add the feature index to the returned index
   int FileOutputComponent::getBufferIndex(int x, int y, int width, int features)
   {
      return (x + y * width) * features;
   }
   
   //TODO: Specify how the extended buffer should be filled in (currently 0)
   vector<pvdata_t> FileOutputComponent::expand(vector<pvdata_t> bufferToExpand, int sourceWidth, int sourceHeight, int newWidth, int newHeight)
   {
      int nF = mParentLayer->getLayerLoc()->nf;
      int newOriginX = newWidth / 2 - sourceWidth / 2;
      int newOriginY = newHeight / 2 - sourceHeight / 2;
      vector<pvdata_t> result(newWidth*newHeight*nF);
      
      for(int i = 0; i < sourceWidth; i++)
      {
         for(int j = 0; j < sourceHeight; j++)
         {
            for(int f = 0; f < nF; f++)
            {
               int resultIndex = getBufferIndex(i+newOriginX, j+newOriginY, newWidth, nF) + f;
               result[resultIndex] = bufferToExpand[getBufferIndex(i, j, sourceWidth, nF)+f];
            }
         }
      }
      return result;
   }
   
   vector<pvdata_t> FileOutputComponent::crop(vector<pvdata_t> bufferToCrop, int sourceWidth, int sourceHeight, int left, int top, int width, int height)
   {
      int nF = mParentLayer->getLayerLoc()->nf;
      vector<pvdata_t> result(width*height*nF);
      for(int i = 0; i < width; i++)
      {
         for(int j = 0; j < height; j++)
         {
            for(int f = 0; f < nF; f++)
            {
               int resultIndex = getBufferIndex(i, j, width, nF) + f;
               result[resultIndex] = bufferToCrop[getBufferIndex(left+i, top+j, sourceWidth, nF)+f];
            }
         }
      }
      return result;
   }
   
   vector<pvdata_t> FileOutputComponent::flip(vector<pvdata_t> bufferToFlip, int sourceWidth, int sourceHeight, bool xFlip, bool yFlip)
   {
      int nF = mParentLayer->getLayerLoc()->nf;
      vector<pvdata_t> result(sourceWidth*sourceHeight*nF);
      for(int f = 0; f < nF; f++)
      {
         for(int i = 0; i < sourceWidth; i++)
         {
            for(int j = 0; j < sourceHeight; j++)
            {
               int readX = xFlip ? sourceWidth - i - 1 : i;
               int readY = yFlip ? sourceHeight - j - 1 : j;
               result[getBufferIndex(i, j, sourceWidth, nF)+f] = bufferToFlip[getBufferIndex(readX, readY, sourceWidth, nF)+f];
            }
         }
      }
      return result;
   }
   
   vector<pvdata_t> FileOutputComponent::shift(vector<pvdata_t> bufferToShift, int sourceWidth, int sourceHeight, int xShift, int yShift)
   {
      int nF = mParentLayer->getLayerLoc()->nf;
      vector<pvdata_t> result(sourceWidth*sourceHeight*nF);
      for(int f = 0; f < nF; f++)
      {
         for(int i = 0; i < sourceWidth; i++)
         {
            for(int j = 0; j < sourceHeight; j++)
            {
               int readX = (i + sourceWidth - xShift) % sourceWidth;
               int readY = (j + sourceHeight - yShift) % sourceHeight;
               result[getBufferIndex(i, j, sourceWidth, nF)+f] = bufferToShift[getBufferIndex(readX, readY, sourceWidth, nF)+f];
            }
         }
      }
      return result;
   }

   vector<pvdata_t> FileOutputComponent::scale(vector<pvdata_t> bufferToScale, int sourceWidth, int sourceHeight, int newWidth, int newHeight)
   {
      int nF = mParentLayer->getLayerLoc()->nf;
      vector<pvdata_t> result(newWidth*newHeight*nF);

      float xRatio = sourceWidth / static_cast<float>(newWidth);
      float yRatio = sourceHeight / static_cast<float>(newHeight);
      
      if(mLinearInterpolation)
      {
         for(int j = 0; j < newHeight; j++)
         {
            for(int i = 0; i < newWidth; i++)
            {
               float xSource = i / (float)(newWidth-1) * (sourceWidth-1);
               float ySource = j / (float)(newHeight-1) * (sourceHeight-1);
               
               for(int f = 0; f < nF; f++)
               {
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
                  
                  float tl = bufferToScale[getBufferIndex(leftIndex, topIndex, sourceWidth, nF)+f] * (1.0f - xAlign) * (1.0f - yAlign);
                  float tr = bufferToScale[getBufferIndex(rightIndex, topIndex, sourceWidth, nF)+f] * xAlign * (1.0f - yAlign);
                  float bl = bufferToScale[getBufferIndex(leftIndex, bottomIndex, sourceWidth, nF)+f] * (1.0f - xAlign) * yAlign;
                  float br = bufferToScale[getBufferIndex(rightIndex, bottomIndex, sourceWidth, nF)+f] * xAlign * yAlign;
                  result[getBufferIndex(leftIndex, topIndex, sourceWidth, nF)+f] = tl+tr+bl+br;
               }
            }
         }
      }
      else //Nearest neighbor, no interpolation
      {
         for(int i = 0; i < newWidth; i++)
         {
            for(int j = 0; j < newHeight; j++)
            {
               int sourceIndex = getBufferIndex(static_cast<int>(i*xRatio), static_cast<int>(j*yRatio), sourceWidth, nF);
               int newIndex = getBufferIndex(i, j, newWidth, nF);
               for(int f = 0; f < nF; f++)
               {
                  result[newIndex+f] = bufferToScale[sourceIndex+f];
               }
            }
         }
      }
      return result;
   }
	
   void FileOutputComponent::transform()
	{
      bool iAmRoot = (mParentLayer->getParent()->icCommunicator()->commRank() == 0);
      int numBatches = mParentLayer->getParent()->getNBatch();
      
      if(iAmRoot)
      {
         bool advanceFile = false;
         
         if(mIsFileList)
         {
            mFileAdvanceTimer++;
            if(mFileAdvanceTimer >= mUpdatesPerFileAdvance)
            {
               mFileAdvanceTimer = 0;
               advanceFile = true;
            }
         }
         
         for(int b = 0; b < numBatches; b++)
         {
            if(!mIsFileList || advanceFile)
            {
                  mFileName = mFileList[ (mFileListIndex + b + 1) % mFileList.size() ];
                  pvInfo() << "Reading file " << mFileName << " into batch " << b << std::endl;
                  updateFileBuffer(mFileName, mFileBuffers[b]);
            }
            // Passing an empty string indicates to update the file buffer
            // without reading new information from disk. This is useful for
            // things like movies or audio that may have more data cached.
            // Components that do not use this kind of data should simply
            // return when passed an empty file name.
            else updateFileBuffer("", mFileBuffers[b]);
            if(advanceFile) { mFileListIndex = (mFileListIndex + numBatches) % mFileList.size(); }
         }
      }

      for(int b = 0; b < numBatches; b++) { scatterFileBuffer(b); }
	}
   
   void FileOutputComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
	{
		ioParam_fileName(ioFlag);
      ioParam_isFileList(ioFlag);
      ioParam_updatePeriod(ioFlag);
      ioParam_updatesPerFileAdvance(ioFlag);
      ioParam_autoScale(ioFlag);
      ioParam_linearInterpolation(ioFlag);
      ioParam_fillExtended(ioFlag);
	}
   
   void FileOutputComponent::ioParam_fileName(enum ParamsIOFlag ioFlag)
   {
      parent->ioParamStringRequired(ioFlag, mParentLayer->getName(), "fileName", &mCFileName);
      mFileName = mCFileName;
   }
      
   void FileOutputComponent::ioParam_isFileList(enum ParamsIOFlag ioFlag)
      { parent->ioParamValue(ioFlag, mParentLayer->getName(), "isFileList", &mIsFileList, mIsFileList); }
      
   void FileOutputComponent::ioParam_updatePeriod(enum ParamsIOFlag ioFlag)
      { parent->ioParamValue(ioFlag, mParentLayer->getName(), "updatePeriod", &mUpdatePeriod, mUpdatePeriod); }
      
   void FileOutputComponent::ioParam_updatesPerFileAdvance(enum ParamsIOFlag ioFlag)
      { parent->ioParamValue(ioFlag, mParentLayer->getName(), "updatesPerFileAdvance", &mUpdatesPerFileAdvance, mUpdatesPerFileAdvance); }
      
   void FileOutputComponent::ioParam_autoScale(enum ParamsIOFlag ioFlag)
      { parent->ioParamValue(ioFlag, mParentLayer->getName(), "autoScale", &mAutoScale, mAutoScale); }
      
   void FileOutputComponent::ioParam_linearInterpolation(enum ParamsIOFlag ioFlag)
      { parent->ioParamValue(ioFlag, mParentLayer->getName(), "linearInterpolation", &mLinearInterpolation, mLinearInterpolation); }
      
   void FileOutputComponent::ioParam_fillExtended(enum ParamsIOFlag ioFlag)
      { parent->ioParamValue(ioFlag, mParentLayer->getName(), "fillExtended", &mFillExtended, mFillExtended); }
}


