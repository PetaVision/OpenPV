/*
 * PlasticCopyConn.cpp
 *
 *  Created on: June 19, 2019
 *      Author: Xinhua Zhang
 */

#include "AffineCopyConn.hpp"
#include "components/AffineCopyWeightsPair.hpp"
#include "weightupdaters/AffineCopyUpdater.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>

#define HAVE_OPENCV_IMGPROC
#define HAVE_OPENCV_IMGCODECS

using namespace std;
using namespace cv;

namespace PV {
   AffineCopyConn::AffineCopyConn(const char *name, PVParams *params, Communicator const *comm)
   {
      initialize(name, params, comm);
   }

   AffineCopyConn::AffineCopyConn() {}

   AffineCopyConn::~AffineCopyConn() {}

   WeightsPairInterface *AffineCopyConn::createWeightsPair() {
      return new AffineCopyWeightsPair(name, parameters(), mCommunicator);
   }

   BaseWeightUpdater *AffineCopyConn::createWeightUpdater() {
      return new AffineCopyUpdater(name, parameters(), mCommunicator);
   }

   Response::Status AffineCopyConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
      auto *affineCopyWeightsPair = getComponentByType<AffineCopyWeightsPair>();
      pvAssert(affineCopyWeightsPair);
      if (!affineCopyWeightsPair->getOriginalWeightsPair()->getInitialValuesSetFlag()) {
         return Response::POSTPONE;
      }
      affineCopyWeightsPair->copy();
      return Response::SUCCESS;
   }
   
   // void AffineCopyConn::rotation_test(const char *imageFileName, const char *outputImageFileName)
   // {
   //    Mat input, rot_mat( 2, 3, CV_32F);      
   //    input = imread(imageFileName,IMREAD_GRAYSCALE);
      
   //    int rows = input.rows, cols = input.cols, size2D = rows * cols;
   //    cout << "rows: " << rows << " cols: " << cols << " total: " << input.total() << " type: " << input.type() << endl;
      
   //    vector<float> inputVec(size2D),outputVec(size2D);
      
   //    for (int i = 0; i < size2D; i++) {
   //       inputVec[i] = (float)input.at<uchar>(i);
   //    }

   //    Mat inputBuffer,outputBuffer(rows,cols,CV_32F);
   //    inputBuffer = Mat(inputVec).reshape(1,rows);
   //    cout << "inputBuffer:\nrows: " << inputBuffer.rows << " cols: " << inputBuffer.cols << " " << inputBuffer.size() << endl;
      
   //    Point center = Point(inputBuffer.size() / 2); 
   //    double angle = 90;
   //    double scale = 1;
   //    rot_mat = getRotationMatrix2D(Point(inputBuffer.size() / 2) , angle, scale );
   //    cout << "rot_mat:\nrows: " << rot_mat.rows << " cols: " << rot_mat.cols << " type: " << rot_mat.type() << endl;
   //    warpAffine( inputBuffer, outputBuffer, rot_mat, inputBuffer.size());      
   //    imwrite(outputImageFileName,outputBuffer);      
   // }
   
   
}

