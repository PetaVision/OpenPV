#include  <stdio.h>
#include  <stdlib.h>
#include  <math.h>
#include  <string.h>
#include  <time.h>  
#include  "/nh/nest/u/manghel/NUMREC/include/nrutil.h"

int    n_modes;
long   N;
float  time_delay;

char   dir[100];
char   version[10];
char   field[10];
double *av_amp;

char    infile[100];


void   InitParams(void);
void   OpenFiles(void);
void   CompAvAmp(void);
void   CompCorr(void);


main(int argc,char *argv[])
{
  char s[100];

  if( argc != 4 ){
    printf("\n Usage: %s root_dir version field",*argv);
    exit(1);
  }
  
  strcpy(dir,*(++argv));          /* read input data dir */
  strcpy(version,*(++argv));	  /* read data version   */
  strcpy(field,*(++argv));	  /* read field to process   */

  InitParams();

  CompAvAmp();

  CompCorr();


}




void InitParams(void)
{

 
  printf("\nprint n_modes recorded (space amp modes): ");
  scanf("%d",&n_modes);


  printf("\nprint time_delay (between space amp projections) : ");
  scanf("%f",&time_delay);

  sprintf(infile,"%s/data/KL_av_surf_%s_norm_projection_%lf_%s.dat",
	  dir,field,time_delay,version);


  av_amp = dvector(1,n_modes);

}


void CompAvAmp(void)
{

  int i,j;
  double amp;
  FILE *fp;

             
  /* open input data file */


  if((fp = fopen(infile,"r")) == NULL){
    printf("Data file %s could not be opened\n",infile);
  }
  printf("reading from %s\n",infile);

  N=0;
  do{
				/* read next amp data point */
    for(j=1;j<=n_modes;j++){
      fscanf(fp,"%lf",&amp);
      av_amp[j]+=amp;
    }
    N++;

  } while (!feof(fp)); /* end training data loop */
  fclose(fp);

  printf("\n\tN= %ld\n",N);

  for(j=1;j<=n_modes;j++)
    av_amp[j] /= N;

}




void CompCorr(void)
{

  int i,j;
  long n;
  double *amp,**corr;
  FILE *fp;
  char s[100];

  amp=dvector(1,n_modes);
  corr = dmatrix(1,n_modes,1,n_modes);

  for(i=1;i<n_modes;i++)
    for(j=1;j<=n_modes;j++)
      corr[i][j] = 0.0;


  if((fp = fopen(infile,"r")) == NULL){
    printf("Data file %s could not be opened\n",s);
  }
  printf("reading from %s\n",infile);


  for(n=1;n<=N;n++){
				/* read amp data  */
    for(j=1;j<=n_modes;j++){
      fscanf(fp,"%lf",&amp[j]);
    }

    for(i=1;i<n_modes;i++)
      for(j=1;j<=n_modes;j++)
	corr[i][j] += (amp[i]-av_amp[i])*(amp[j]-av_amp[j]);

  } /* end training data loop */
  fclose(fp);

  for(i=1;i<n_modes;i++)
    for(j=1;j<=n_modes;j++){
      corr[i][j] /= N;
      if(i==j)
	amp[i] = sqrt(corr[i][i]);
    }

  printf("\n\ncorr before var normalization:\n");
  for(i=1;i<n_modes;i++)
    for(j=1;j<=n_modes;j++){
      if(i == j)
	printf("corr[%d][%d]= %lf\n",i,j,corr[i][j]);
      corr[i][j] = corr[i][j]/(amp[i]*amp[j]);
    }

  printf("\n\ncorr after var normalization:\n");
  for(i=1;i<n_modes;i++)
    for(j=1;j<=n_modes;j++)
      if(i == j)
	printf("corr[%d][%d]= %lf\n",i,j,corr[i][j]);


				/* outfile */
  sprintf(s,"%s/NNdata/surf_%s_norm_projection_corr_%lf.dat",
	  dir,field,time_delay);
  if((fp = fopen(s,"w")) == NULL){
    printf("Data file %s could not be opened\n",s);
  }
  printf("writing corr data to %s\n",s);

  for(i=1;i<=n_modes;i++){
    for(j=1;j<=n_modes;j++)
      fprintf(fp,"%lf ",corr[i][j]);
    fprintf(fp,"\n");
  }
  fclose(fp);

  free_dvector(amp,1,n_modes);
  free_dvector(av_amp,1,n_modes);
  free_dmatrix(corr,1,n_modes,1,n_modes);

}


