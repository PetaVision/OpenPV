/* 
 *  grabcut.cxx
 *  Vadas Gintautas 2011
 *  vadasg@gmail.com
 *
 *  This requires OpenCV to be installed.
 */


#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    bool init_with_rect;

    //initialize matrices for grabcut
    Mat bgdModel, fgdModel;
    Mat mask;

    //see if executable was called properly
    //number of arguments determines initialization method
    
    if(argc==5){

        init_with_rect = false;

    }
    else if(argc==8){

        init_with_rect = true;

    }
    else{

        printf("\nUsage:\n\n\
To initialize with bounding box rectangle ( rect_x1 rect_y1 rect_x2 rect_y2 ):\n\n\
\tgrabcut number_iterations <input-image-file-name> <output-file-name> rect_x1 rect_y1 rect_x2 rect_y2\n\n\
To initialize with input mask:\n\n\
\tgrabcut number_iterations <input-image-file-name> <output-file-name> <input-mask-file-name>\n\n\7");

        exit(0);

        }


    // load an image  
    Mat img=cvLoadImage(argv[2]);

    if(init_with_rect){

        //initialize output matrix
        mask = Scalar(0);

        //bounding box rectangle from command line arguments
        Rect rect(Point(atoi(argv[4]), atoi(argv[5])), Point(atoi(argv[6]), atoi(argv[7])));
        
        //segment image
        grabCut( img, mask, rect, bgdModel, fgdModel, atoi(argv[1]), GC_INIT_WITH_RECT );   

    }
    else{  //intialize with input mask

        //load input mask from image
        mask=cvLoadImage(argv[4],0);

        //generic bounding box rectangle
        Rect rect(Point(1, 1), Point(2, 2));

        //segment image
        grabCut( img, mask, rect, bgdModel, fgdModel, atoi(argv[1]), GC_INIT_WITH_MASK );    

    }

    //save output as black-white mask
    imwrite(argv[3],(mask & 1)*255);

    return 0;

}
