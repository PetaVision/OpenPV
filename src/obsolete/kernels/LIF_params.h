#ifndef LIF_PARAMS_H_
#define LIF_PARAMS_H_

#define pvconductance_t float

typedef struct LIF_params_ {
   float Vrest;
   float Vexc;
   float Vinh;
   float VinhB;

   float tau;
   float tauE;
   float tauI;
   float tauIB;

   float VthRest;
   float tauVth;
   float deltaVth;
   float deltaGIB;

   float noiseFreqE;
   float noiseAmpE;
   float noiseFreqI;
   float noiseAmpI;
   float noiseFreqIB;
   float noiseAmpIB;
} LIF_params;

#endif /* LIF_PARAMS_H_ */

