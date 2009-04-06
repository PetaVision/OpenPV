#ifndef CLOCK_H_
#define CLOCK_H_

#ifdef __cplusplus
extern "C" {
#endif

void   start_clock();
void   stop_clock();
double elapsed_time();

#ifdef __cplusplus
}
#endif

#endif // CLOCK_H_
