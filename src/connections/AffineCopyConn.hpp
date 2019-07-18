/*
 * AffineCopyConn.hpp
 *
 *  Created on: June 19, 2019
 *      Author: Xinhua Zhang
 */

#ifndef AFFINECOPYCONN_HPP_
#define AFFINECOPYCONN_HPP_

#include "CopyConn.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include <stdlib.h>

using namespace cv;
using namespace std;

namespace PV {
   class AffineCopyConn  : public CopyConn
   {
   public:
      AffineCopyConn(const char *name, PVParams *params, Communicator const *comm);
      static void rotation_test(const char *imageFileName,const char *outputImageFileName, const char *angleStr) 
         {
            Mat input, rot_mat( 2, 3, CV_32F);      
            input = imread(imageFileName,IMREAD_GRAYSCALE);
      
            int rows = input.rows, cols = input.cols, size2D = rows * cols;
            cout << "Image:\nrows: " << rows << " cols: " << cols << " total: " << input.total() << " type: " << input.type() << endl;
      
            vector<float> inputVec(size2D),outputVec(size2D);
      
            for (int i = 0; i < size2D; i++) {
               inputVec[i] = (float)input.at<uchar>(i);
            }

            Mat inputBuffer,outputBuffer(rows,cols,CV_32F);
            inputBuffer = Mat(inputVec).reshape(1,rows);
            cout << "inputBuffer:\nrows: " << inputBuffer.rows << " cols: " << inputBuffer.cols << " " << inputBuffer.size() << endl;
      
            Point center = Point(inputBuffer.size() / 2); 
            double angle = (int)atoi(angleStr);
            double scale = 1;
            rot_mat = getRotationMatrix2D(Point(inputBuffer.size() / 2) , angle, scale );
            cout << "rot_mat:\nrows: " << rot_mat.rows << " cols: " << rot_mat.cols << " type: " << rot_mat.type() << endl;
            cout << "angle of rotation: " << angle << endl;
             
            warpAffine( inputBuffer, outputBuffer, rot_mat, inputBuffer.size());      
            imwrite(outputImageFileName,outputBuffer);      
         }
      
      virtual ~AffineCopyConn();      
   protected:
      AffineCopyConn();
      virtual WeightsPairInterface *createWeightsPair() override;
      virtual BaseWeightUpdater *createWeightUpdater() override;

      virtual Response::Status
      initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   };
} // namespace PV

#endif /* AFFINECOPYCONN_HPP_ */
