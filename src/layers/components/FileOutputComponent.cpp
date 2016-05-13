#include "FileOutputComponent.hpp"

namespace PV
{
   //Very WIP, does not work yet. Should build, but completely untested.
   
   void FileOutputComponent::initialize()
   {

   }

   //readFile() should fill mFileBuffer and will only be called on the root process.
   //this will handle scattering mFileBuffer to other processes. If our data is in
   //the form of a list, we'll give each batch a different file
   void FileOutputComponent::scatterFileBuffer() 
	{
   }
   
   //This doesn't give you features- just add the feature index to the returned index
   int FileOutputComponent::getBufferIndex(int x, int y, int width, int features)
   {
      return x + y * width * features;
   }
   
   pvdata_t FileOutputComponent::getFromFile(int x, int y, int f)
   {
      //The extra addition here means it will wrap correctly all the way until x < -sourceWidth, which should never happen
      mFileBuffer[getBufferIndex((x + mFileWidth) % mFileWidth, (y + mFileHeight) % mFileHeight, mFileWidth, mParentLayer->getLayerLoc()->nf)+f];
   }
   
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
   
   vector<pvdata_t> FileOutputComponent::crop(vector<pvdata_t> bufferToCrop, int sourceWidth, int sourceHeight, int leftCrop, int rightCrop, int topCrop, int bottomCrop)
   {
      int nF = mParentLayer->getLayerLoc()->nf;
      int newWidth = sourceWidth - leftCrop - rightCrop;
      int newHeight = sourceHeight - topCrop - bottomCrop;
      vector<pvdata_t> result(newWidth*newHeight*nF);
      for(int i = leftCrop; i < sourceWidth-rightCrop; i++)
      {
         for(int j = topCrop; j < sourceHeight-bottomCrop; j++)
         {
            for(int f = 0; f < nF; f++)
            {
               int resultIndex = getBufferIndex(i-leftCrop, j-topCrop, newWidth, nF) + f;
               result[resultIndex] = bufferToCrop[getBufferIndex(i, j, sourceWidth, nF)+f];
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
         for(int i = 0; i < newWidth; i++)
         {
            for(int j = 0; j < newHeight; j++)
            {
               float xSource = i / static_cast<float>(newWidth-1) * (sourceWidth-1);
               float ySource = j / static_cast<float>(newHeight-1) * (sourceHeight-1);
               
               for(int f = 0; f < nF; f++)
               {
                  float xAvg = 0.0f;
                  float yAvg = 0.0f;
                  
                  int leftIndex = static_cast<int>(xSource);
                  int rightIndex = static_cast<int>(ceil(xSource));
                  int topIndex = static_cast<int>(ySource);
                  int bottomIndex = static_cast<int>(ceil(ySource));
                  
                  if(xRatio < 1.0f) //New buffer is bigger than the source, we'll interpolate between two neighboring values
                  {
                     for(int yIndex = topIndex; yIndex <= bottomIndex; yIndex++)
                     {
                        float xAlign = xSource - leftIndex;
                        xAvg += bufferToScale[getBufferIndex(leftIndex, yIndex, sourceWidth, nF)+f] * (1.0f - xAlign);
                        xAvg += bufferToScale[getBufferIndex(rightIndex, yIndex, sourceWidth, nF)+f] * xAlign;
                     }
                     if(bottomIndex - topIndex > 0) xAvg /= bottomIndex - topIndex + 1;
                  }
                  else if(xRatio == 1.0f)
                  {
                     xAvg = bufferToScale[getBufferIndex(leftIndex, topIndex, sourceWidth, nF)+f];
                  }
                  else //New buffer is smaller than the source, we'll average the values between the source locations
                  {
                     for(int yIndex = topIndex; yIndex <= bottomIndex; yIndex++)
                     {
                        for(int xIndex = leftIndex; xIndex <= rightIndex; xIndex++)
                        {
                           xAvg += bufferToScale[getBufferIndex(xIndex, yIndex, sourceWidth, nF)+f];
                        }
                     }
                     if(rightIndex - leftIndex > 0) xAvg /= rightIndex - leftIndex + 1;
                     if(bottomIndex - topIndex > 0) xAvg /= bottomIndex - topIndex + 1;
                  }

                  if(yRatio < 1.0f)
                  {
                     for(int xIndex = leftIndex; xIndex <= rightIndex; xIndex++)
                     {
                        float yAlign = ySource - topIndex;
                        yAvg += bufferToScale[getBufferIndex(xIndex, topIndex, sourceWidth, nF)+f] * (1.0f - yAlign);
                        yAvg += bufferToScale[getBufferIndex(xIndex, bottomIndex, sourceWidth, nF)+f] * yAlign;
                     }
                     if(rightIndex - leftIndex > 0) yAvg /= rightIndex - leftIndex + 1;
                  }
                  else if(yRatio == 1.0f)
                  {
                     yAvg = bufferToScale[getBufferIndex(leftIndex, topIndex, sourceWidth, nF)+f];
                  }
                  else //New buffer is smaller than the source, we'll average the values between the source locations
                  {
                     for(int yIndex = topIndex; yIndex <= bottomIndex; yIndex++)
                     {
                        for(int xIndex = leftIndex; xIndex <= rightIndex; xIndex++)
                        {
                           yAvg += bufferToScale[getBufferIndex(xIndex, yIndex, sourceWidth, nF)+f];
                        }
                     }
                     if(rightIndex-leftIndex > 0) yAvg /= rightIndex-leftIndex + 1;
                     if(topIndex-bottomIndex > 0) yAvg /= topIndex-bottomIndex + 1;
                  }
                  result[getBufferIndex(i, j, newWidth, nF)+f] = (xAvg + yAvg) * 0.5f;
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
      if(mDisplayTimer++ < mDisplayPeriod) return;
      mDisplayTimer = 0;

      //A call to readfile should go somewhere around here, plus playlist advancing

      MPI_Comm mpiComm = mParentLayer->getParent()->icCommunicator()->communicator();
      int nXProcs = mParentLayer->getParent()->icCommunicator()->numCommColumns();
      int nYProcs = mParentLayer->getParent()->icCommunicator()->numCommRows();
      int commRank = mParentLayer->getParent()->icCommunicator()->commRank();
      
      //These used to be inside the commRank==0 if below, make sure it was ok to move up here
      int nX = mParentLayer->getLayerLoc()->nx;
      int nY = mParentLayer->getLayerLoc()->ny;
      int nF = mParentLayer->getLayerLoc()->nf;
      if(mFillExtended)
      {
         nX += mParentLayer->getLayerLoc()->halo.lt+mParentLayer->getLayerLoc()->halo.rt;
         nY += mParentLayer->getLayerLoc()->halo.up+mParentLayer->getLayerLoc()->halo.dn;
      }
      
      vector<pvdata_t> fileSlice;
      if(commRank == 0) //Root process, send data to other processes
      {
         
         
         int columnWidth = nX * nXProcs;
         int columnHeight = nY * nYProcs;

         vector<pvdata_t> preparedFile;
        
         if(mAutoScale)
         {
            preparedFile = scale(mFileBuffer, mFileWidth, mFileHeight, columnWidth, columnHeight);
         }
         else
         {
            preparedFile = expand(mFileBuffer, mFileWidth, mFileHeight, columnWidth, columnHeight);
         }
         
         //shift / flip goes here
         
         //Slice up the data and send it to each process
         for(int dest = nYProcs*nXProcs-1; dest >= 0; dest--)
         {
            int col = columnFromRank(dest, nYProcs, nXProcs);
            int row = rowFromRank(dest, nYProcs, nXProcs);
            int layerXStart = nX * col;
            int layerYStart = nY * row;
            fileSlice = crop(preparedFile, columnWidth, columnHeight, layerXStart, layerYStart, columnWidth-(layerXStart+nX), columnHeight-(layerYStart+nY));
            int dataSize = fileSlice.size();
            if(dest > 0) //Don't send to ourselves, just keep fileSlice
            {
               MPI_Send(&dataSize,  1,        MPI_INT,   dest, 0, mpiComm);
               MPI_Send(&fileSlice[0],  dataSize, MPI_FLOAT, dest, 1, mpiComm);
               pvInfo() << "Root sending " << dataSize << " floats to process " << dest << std::endl;
            }
         }
      }
      else //We aren't root, receive data from root
      {
         int dataSize;
         MPI_Recv(&dataSize,     1,        MPI_INT,   0,    0, mpiComm, MPI_STATUS_IGNORE);
         pvInfo() << "Process " << commRank << " receiving " << dataSize << " floats." << std::endl;
         fileSlice.resize(dataSize);
         MPI_Recv(&fileSlice[0], dataSize, MPI_FLOAT, 0,    1, mpiComm, MPI_STATUS_IGNORE);
         pvInfo() << "Process " << commRank << " received." << std::endl;
      }
      //fileSlice should now have the correct data regardless of which process we're in
       
       //TODO: Batching. Probably enclose this whole function in a for(batches) loop,
       //call readFile() each time to make sure we switch files for each batch
      int batchIndex = 0; 
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
}


