#include "../pv.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//#define PARTIAL_OUTPUT 1


int main(int argc, char* argv[])
  {
    const int NUM_CIRC = 1;
    const float REL_POS_X0[1]={1.0/2.0};
    const float REL_POS_Y0[1]={1.0/2.0};
    const float REL_RADIUS[1]={1.0/4.0};
    const char input_path[64] = "./input/circle1_";
    const int n= NX*NY*NO*NK;
    int comm_size=1;    

    int k,i, kk, t, b;
    int uu= 0;
    int du1=0;
    int du2=0;
    int e=0;
    int u[NUM_CIRC];
    int j, jj, jjj,jjjj,p, qi; //indices for x and y initialization 
    //    int ki[NK];// index array for curvature
    eventtype_t f[n];
    float dx, dy, r, r2;
    int nrows = (int) 1;
    int ncols = (int) 1/nrows;
    float xa, ya, oa;
    float x[n], y[n], o[n]; 
    
    float X0, Y0;
    float x0 = 0.0;
    float y0 = 0.0;
    

    p=0;
    for (j = 0; j < NY; j++)
      {
        ya = j*DY;
        for (jj = 0; jj < NX; jj++)
          {
            xa = jj*DX;
	
            for (jjj= 0; jjj < NO; jjj++)
              {
		oa= jjj*DTH;
		for (jjjj= 0; jjjj<NK; jjjj++)
		  {
		    ka = jjj*DK;
		    x[p] = xa;
		    o[p] = oa;
		    y[p] = ya;
		    kappa[p] = ka;
		    p++;
		  }//jjjj<nk
              } // jjj < no
          } // jj < nx
      } // j < ny
   /*  for(i=0;i<NUM_CIRC;i++) */
   
    
    float pi = 2.0*acos(0.0);

    const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;
    const float clutter_prob = CLUTTER_PROB; // prob of pixel being "on"
   
    float r_circle;
    float r_tolerance = 0.04; // percent deviation from radius, 0.05 = max for non-sloppy circle 
    float r2_min, r2_max;
    float xc, yc;
    
    //file charcter array declarations
    char filename_circlein[64];
    char number_of_in[64];
    char clutter_indices[64];
    char input_indices[64];

    //index string declaration
    char index[2];
    

    //file names
    strcpy(number_of_in, input_path);
    strncat(number_of_in, "num",4);
    strncat(number_of_in, ".bin", 5);

    strcpy(input_indices, input_path);
    strncat(input_indices, "input",6);
    strncat(input_indices, ".bin", 5);
    
    strcpy(clutter_indices, input_path);
    strncat(clutter_indices, "clutter",8);
    strncat(clutter_indices, ".bin", 5);

    //open num file and write out number of circles
    FILE* num = fopen(number_of_in, "wb");
    if(num==NULL)
      
	printf("ERROR: FAILED TO OPEN NUMBER FILE");
        
    else fwrite(&NUM_CIRC, sizeof(int), 1 ,num);
    
    fclose(num);
    
    // open clutter indices file
    FILE* fclutter = fopen(clutter_indices, "wb");
    if(fclutter ==NULL)
      printf("ERROR: FAILED TO OPEN CLUTTER INDICES FILE");

    //start for loop (am I on the circle or not)
    for(i=0;i<NUM_CIRC;i++)
      {
	u[i]=0;
	r_circle = NX*ncols * REL_RADIUS[i];
	//kappaF = (1/r_circle);
	r2_min = r_circle * r_circle * (1 - r_tolerance) * (1 - r_tolerance);
	r2_max= r_circle * r_circle * (1 + r_tolerance) * (1 + r_tolerance);
	
	X0 = ncols*NX*REL_POS_X0[i];
	
	Y0 = nrows*NY*REL_POS_Y0[i];
	
	FILE* fo;
	
	sprintf(index,"%d",i);
	strcpy(filename_circlein, input_path);
	strncat(filename_circlein, "figure_",8);
	strncat(filename_circlein, index, 2);
	strncat(filename_circlein, ".bin", 5);
	
	fo = fopen(filename_circlein, "wb");
	if (fo == NULL)
	  {
		printf ("ERROR: FAILED TO OPEN FIGURE FILE NO. %d",k);
		continue;
	  }
		
	for (k = 0; k <n ; k+= NO*NK)
	  {
	    for ( t=0; t<NO*NK; t+=NK)
	      {
	 	
		for (qi=0; qi<NK; qi++)
		  {
		    b = k+t+qi;
		    /* turn on random edges */
		    r = rand() * INV_RAND_MAX;
		    f[b] = (r < clutter_prob) ? 1.0 : 0.0;
		    xc = x[b] + x0;
		    yc = y[b] + y0;
		    
		    /* turn on circle pixels */
		    dx = (xc - X0);
		    dy = (yc - Y0);
		    r2 = dx*dx + dy*dy;
		    if (r2> r2_min && r2 < r2_max)
		      {	
			float a = 90.0 + (180./pi)*atanf(dy/dx);
			a = fmod(a, 180.);
			kk = 0.5 + (a / DTH);
			kk = kk % NO; 	    
			// kk = rand() * NO; //randomize orientations
			if ((t/NK) == kk )//&& f[k] == 0.0)
			  {		
			    f[b] = 1.0;
			 
			    fwrite(&b , sizeof(int), 1 , fo ); 
			    u[i]++;	     
			    e=1;
			    //printf("f[%d]= %f\n", b, f[b]);
			    /*   printf("we got %d now\n",u[i]); */
			    /* 		    printf("index %d\n", k) */;
			    //printf("t=%d kk=%d k=%d o = %f r = %f (%d, %d) (%f, %f)\n", t, kk, k, a, sqrt(r2), i, j, dx, dy);
			  }
		      }
		    if(r< clutter_prob && e == 0 && fclutter!= NULL)
		      {
			fwrite(&k, sizeof(int),1, fclutter);
			uu++;
		      }
		     e=0;
		  }
	      }
	   
	   
	  }
	
	
	//close circle indices file
	fclose(fo);
	
	
	  }
	
    //close clutter file
    fclose(fclutter);

    //open input file and write firing array
    FILE* input_file = fopen(input_indices, "wb");
    if (input_file == NULL)
      {
	printf("ERROR: FAILED TO OPEN INPUT FILE");
      }
    else
      {
	fwrite(f , sizeof(float), n , input_file); 
      }
    fclose(input_file);

    printf("u1 = %d\n",u[0]);

    //open the num file and write the number of indices for each thing (clutter and circles)
    FILE* fnum = fopen(number_of_in, "ab");
    if (fnum!= NULL)
      {
	fwrite(&uu, sizeof(int),1, fnum);
	printf("uu= %d\n", uu);
	for(i=0;i<NUM_CIRC;i++)
	  {
	    fwrite(&u[i], sizeof(int), 1, fnum);
	  
	  }
	fclose(num);
      }   

    //Tell the user to go look at results
    printf("Please see input path for file results.");
    return 0;
  } 

