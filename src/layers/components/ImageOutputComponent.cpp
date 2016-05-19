#include "ImageOutputComponent.hpp"

#include <io/imageio.hpp>

#ifdef PV_USE_GDAL
#  include <gdal.h>
#  include <gdal_priv.h>
#  include <ogr_spatialref.h>
#else
#  define GDAL_CONFIG_ERR_STR "PetaVision must be compiled with GDAL to use this file type\n"
#endif

namespace PV
{
	BaseObject * createImageOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new ImageOutputComponent() : NULL;
	}
	
	void ImageOutputComponent::updateFileBuffer(std::string fileName, std::vector<pvdata_t> &fileBuffer)
   {
      if(fileName == "") return;
      
      GDALAllRegister();
      GDALDataset* image = PV_GDALOpen(fileName.c_str());
      
      if(image == nullptr) { pvError() << mParentLayer->getName() << ": GDAL Failed to open file " << fileName << std::endl; throw; }
      
      GDALDataType dataType = image->GetRasterBand(1)->GetRasterDataType();
      int xImageSize = image->GetRasterXSize();
      int yImageSize = image->GetRasterYSize();
      int numChannels = image->GetRasterCount();
      float valueFactor = 1.0f;
      
      if(dataType == GDT_Byte) valueFactor = 255.0f;
      if(dataType == GDT_UInt16) valueFactor = 65535.0f;
      
      fileBuffer.resize(xImageSize * yImageSize * numChannels);
      mFileWidth = xImageSize;
      mFileHeight = yImageSize;
      
      image->RasterIO(GF_Read,   0,                //x offset
                                 0,                //y offset
                                 xImageSize,       //original width
                                 yImageSize,       //original height
                                 &fileBuffer[0],  //buffer to read into
                                 xImageSize,       //buffer width
                                 yImageSize,       //buffer height
                                 GDT_Float32,      //buffer type
                                 numChannels,      //channel count
                                 NULL,             //which bands to read (NULL = all)
                                 numChannels * sizeof(float), // Byte offset between pixels
                                 numChannels * xImageSize * sizeof(float), //Byte offset between lines
                                 sizeof(float)); //Byte offset between channels
      
      GDALClose(image);
                                 
      for(int i = 0; i < fileBuffer.size(); i++)
      {
         fileBuffer[i] /= valueFactor;
      }
   }
}

