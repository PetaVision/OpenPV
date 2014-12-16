#include <iostream>
#include <stdio.h>
#include <math.h>

#include "mail.h"
#include "io_disp.h"
#include "utils.h"

using namespace std;

vector<float> disparityErrorsOutlier (DisparityImage &D_gt,DisparityImage &D_orig,DisparityImage &D_ipol) {

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  for (int32_t i=0; i<2*5; i++)
    errors.push_back(0);
  int32_t num_pixels = 0;
  int32_t num_pixels_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_err  = fabs(D_gt.getDisp(u,v)-D_ipol.getDisp(u,v));
        for (int32_t i=0; i<5; i++)
          if (d_err>(float)(i+1))
            errors[i*2+0]++;
        num_pixels++;
        if (D_orig.isValid(u,v)) {
          for (int32_t i=0; i<5; i++)
            if (d_err>(float)(i+1))
              errors[i*2+1]++;
          num_pixels_result++;
        }
      }
    }
  }

  // check number of pixels
  if (num_pixels==0) {
    cout << "ERROR: Ground truth defect => Please write me an email!" << endl;
    throw 1;
  }

  // normalize errors
  for (int32_t i=0; i<errors.size(); i+=2)
    errors[i] /= max((float)num_pixels,1.0f);
  if (num_pixels_result>0)
    for (int32_t i=1; i<errors.size(); i+=2)
      errors[i] /= max((float)num_pixels_result,1.0f);

  // push back density
  errors.push_back((float)num_pixels_result/max((float)num_pixels,1.0f));

  // return errors
  return errors;
}

vector<float> disparityErrorsAverage (DisparityImage &D_gt,DisparityImage &D_orig,DisparityImage &D_ipol) {

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  for (int32_t i=0; i<2; i++)
    errors.push_back(0);
  int32_t num_pixels = 0;
  int32_t num_pixels_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_err = fabs(D_gt.getDisp(u,v)-D_ipol.getDisp(u,v));
        errors[0] += d_err;
        num_pixels++;
        if (D_orig.isValid(u,v)) {
          errors[1] += d_err;
          num_pixels_result++;
        }
      }
    }
  }

  // normalize errors
  errors[0] /= max((float)num_pixels,1.0f);
  errors[1] /= max((float)num_pixels_result,1.0f);

  // return errors
  return errors;
}

bool eval (string result_sha,Mail* mail) {

  // ground truth and result directories
  string gt_noc_dir = "data/stereo_flow/disp_noc";
  string gt_occ_dir = "data/stereo_flow/disp_occ";
  string gt_img_dir = "data/stereo_flow/image_0";
  string result_dir = "results/" + result_sha;

  // create output directories
  system(("mkdir " + result_dir + "/errors_noc_out/").c_str());
  system(("mkdir " + result_dir + "/errors_occ_out/").c_str());
  system(("mkdir " + result_dir + "/errors_noc_avg/").c_str());
  system(("mkdir " + result_dir + "/errors_occ_avg/").c_str());
  system(("mkdir " + result_dir + "/errors_img/").c_str());
  system(("mkdir " + result_dir + "/disp_orig/").c_str());
  system(("mkdir " + result_dir + "/disp_ipol/").c_str());
  system(("mkdir " + result_dir + "/image_0/").c_str());

  // vector for storing the errors
  vector< vector<float> > errors_noc_out;
  vector< vector<float> > errors_occ_out;
  vector< vector<float> > errors_noc_avg;
  vector< vector<float> > errors_occ_avg;

  // for all test files do
  for (int32_t i=0; i<195; i++) {

    // file name
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    
    // output
    mail->msg("Processing: %s.png",prefix);

    // catch errors, when loading fails
    try {

      // load ground truth disparity maps
      DisparityImage D_gt_noc(gt_noc_dir + "/" + prefix + ".png");
      DisparityImage D_gt_occ(gt_occ_dir + "/" + prefix + ".png");
      
      // check submitted result
      string image_file = result_dir + "/data/" + prefix + ".png";
      if (!imageFormat(image_file,png::color_type_gray,16,D_gt_noc.width(),D_gt_noc.height())) {
        mail->msg("ERROR: Input must be png, 1 channel, 16 bit, %d x %d px",
                  D_gt_noc.width(),D_gt_noc.height());
        return false;        
      }

      // load submitted result and interpolate missing values
      DisparityImage D_orig(image_file);
      DisparityImage D_ipol(D_orig);
      D_ipol.interpolateBackground();

      // add disparity errors
      vector<float> errors_noc_out_curr = disparityErrorsOutlier(D_gt_noc,D_orig,D_ipol);
      vector<float> errors_occ_out_curr = disparityErrorsOutlier(D_gt_occ,D_orig,D_ipol);
      vector<float> errors_noc_avg_curr = disparityErrorsAverage(D_gt_noc,D_orig,D_ipol);
      vector<float> errors_occ_avg_curr = disparityErrorsAverage(D_gt_occ,D_orig,D_ipol);
      errors_noc_out.push_back(errors_noc_out_curr);
      errors_occ_out.push_back(errors_occ_out_curr);
      errors_noc_avg.push_back(errors_noc_avg_curr);
      errors_occ_avg.push_back(errors_occ_avg_curr);

      // save detailed infos for first 20 images
      if (i<20) {
      
        // save errors of error images to text file
        FILE *errors_noc_out_file = fopen((result_dir + "/errors_noc_out/" + prefix + ".txt").c_str(),"w");
        FILE *errors_occ_out_file = fopen((result_dir + "/errors_occ_out/" + prefix + ".txt").c_str(),"w");
        FILE *errors_noc_avg_file = fopen((result_dir + "/errors_noc_avg/" + prefix + ".txt").c_str(),"w");
        FILE *errors_occ_avg_file = fopen((result_dir + "/errors_occ_avg/" + prefix + ".txt").c_str(),"w");
        if (errors_noc_out_file==NULL || errors_occ_out_file==NULL ||
            errors_noc_avg_file==NULL || errors_occ_avg_file==NULL) {
          mail->msg("ERROR: Couldn't generate/store output statistics!");
          return false;
        }
        for (int32_t j=0; j<errors_noc_out_curr.size(); j++) {
          fprintf(errors_noc_out_file,"%f ",errors_noc_out_curr[j]);
          fprintf(errors_occ_out_file,"%f ",errors_occ_out_curr[j]);
        }
        for (int32_t j=0; j<errors_noc_avg_curr.size(); j++) {
          fprintf(errors_noc_avg_file,"%f ",errors_noc_avg_curr[j]);
          fprintf(errors_occ_avg_file,"%f ",errors_occ_avg_curr[j]);
        }
        fclose(errors_noc_out_file);
        fclose(errors_occ_out_file);
        fclose(errors_noc_avg_file);
        fclose(errors_occ_avg_file);

        // save error image
        png::image<png::rgb_pixel> D_err = D_ipol.errorImage(D_gt_noc,D_gt_occ);
        D_err.write(result_dir + "/errors_img/" + prefix + ".png");
        
        // compute maximum disparity
        float max_disp = D_gt_occ.maxDisp();
               
        // save original flow image
        D_orig.writeColor(result_dir + "/disp_orig/" + prefix + ".png",max_disp);
        
        // save interpolated flow image
        D_ipol.writeColor(result_dir + "/disp_ipol/" + prefix + ".png",max_disp);

        // copy left camera image        
        string img_src = gt_img_dir + "/" + prefix + ".png";
        string img_dst = result_dir + "/image_0/" + prefix + ".png";
        system(("cp " + img_src + " " + img_dst).c_str());
      }

    // on error, exit
    } catch (...) {
      mail->msg("ERROR: Couldn't read: %s.png", prefix);
      return false;
    }
  }

  // open stats file for writing
  string stats_noc_out_file_name = result_dir + "/stats_noc_out.txt";
  string stats_occ_out_file_name = result_dir + "/stats_occ_out.txt";
  string stats_noc_avg_file_name = result_dir + "/stats_noc_avg.txt";
  string stats_occ_avg_file_name = result_dir + "/stats_occ_avg.txt";
  FILE *stats_noc_out_file = fopen(stats_noc_out_file_name.c_str(),"w");
  FILE *stats_occ_out_file = fopen(stats_occ_out_file_name.c_str(),"w");
  FILE *stats_noc_avg_file = fopen(stats_noc_avg_file_name.c_str(),"w");
  FILE *stats_occ_avg_file = fopen(stats_occ_avg_file_name.c_str(),"w");
  if (stats_noc_out_file==NULL || stats_occ_out_file==NULL || errors_noc_out.size()==0 || errors_occ_out.size()==0 ||
      stats_noc_avg_file==NULL || stats_occ_avg_file==NULL || errors_noc_avg.size()==0 || errors_occ_avg.size()==0) {
    mail->msg("ERROR: Couldn't generate/store output statistics!");
    return false;
  }
  
  // write mean
  for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
    fprintf(stats_noc_out_file,"%f ",statMean(errors_noc_out,i));
    fprintf(stats_occ_out_file,"%f ",statMean(errors_occ_out,i));
  }
  for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
    fprintf(stats_noc_avg_file,"%f ",statMean(errors_noc_avg,i));
    fprintf(stats_occ_avg_file,"%f ",statMean(errors_occ_avg,i));
  }
  fprintf(stats_noc_out_file,"\n");
  fprintf(stats_occ_out_file,"\n");
  fprintf(stats_noc_avg_file,"\n");
  fprintf(stats_occ_avg_file,"\n");
  
  // write min
  for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
    fprintf(stats_noc_out_file,"%f ",statMin(errors_noc_out,i));
    fprintf(stats_occ_out_file,"%f ",statMin(errors_occ_out,i));
  }
  for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
    fprintf(stats_noc_avg_file,"%f ",statMin(errors_noc_avg,i));
    fprintf(stats_occ_avg_file,"%f ",statMin(errors_occ_avg,i));
  }
  fprintf(stats_noc_out_file,"\n");
  fprintf(stats_occ_out_file,"\n");
  fprintf(stats_noc_avg_file,"\n");
  fprintf(stats_occ_avg_file,"\n");
  
  // write max
  for (int32_t i=0; i<errors_noc_out[0].size(); i++) {
    fprintf(stats_noc_out_file,"%f ",statMax(errors_noc_out,i));
    fprintf(stats_occ_out_file,"%f ",statMax(errors_occ_out,i));
  }
  for (int32_t i=0; i<errors_noc_avg[0].size(); i++) {
    fprintf(stats_noc_avg_file,"%f ",statMax(errors_noc_avg,i));
    fprintf(stats_occ_avg_file,"%f ",statMax(errors_occ_avg,i));
  }
  fprintf(stats_noc_out_file,"\n");
  fprintf(stats_occ_out_file,"\n");
  fprintf(stats_noc_avg_file,"\n");
  fprintf(stats_occ_avg_file,"\n");
  
  // close files
  fclose(stats_noc_out_file);
  fclose(stats_occ_out_file);
  fclose(stats_noc_avg_file);
  fclose(stats_occ_avg_file);

  // success
	return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=2 && argc!=4) {
    cout << "Usage: ./eval_stereo result_sha [user_sha email]" << endl;
    return 1;
  }

  // read arguments
  string result_sha = argv[1];

  // init notification mail
  Mail *mail;
  if (argc==4) mail = new Mail(argv[3]);
  else         mail = new Mail();
  mail->msg("Thank you for participating in our evaluation!");

  // run evaluation
  if (eval(result_sha,mail)) {
    mail->msg("Your evaluation results are available at:");
    mail->msg("http://www.cvlibs.net/datasets/kitti/user_submit_check_login.php?benchmark=stereo&user=%s&result=%s",argv[2], result_sha.c_str());
  } else {
    system(("rm -r results/" + result_sha).c_str());
    mail->msg("An error occured while processing your results.");
    mail->msg("Please make sure that the data in your zip archive has the right format!");
  }

  // send mail and exit
  delete mail;
  return 0;
}

