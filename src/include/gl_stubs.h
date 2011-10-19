/*
 * gl_stubs.h
 *
 *  Created on: Jan 14, 2010
 *      Author: rasmussn
 */

#ifndef GL_STUBS_H_
#define GL_STUBS_H_

#define GLUT_RGB                1
#define GLUT_DOUBLE             1
#define GLUT_DEPTH              1

#ifndef GL_FILL
#define GL_FILL                 1
#endif
#ifndef GL_FRONT_AND_BACK
#define GL_FRONT_AND_BACK       1
#endif
#ifndef GL_LINEAR
#define GL_LINEAR               1
#endif
#ifndef GL_REPEAT
#define GL_REPEAT               1
#endif
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D           1
#endif
#ifndef GL_TEXTURE_MAG_FILTER
#define GL_TEXTURE_MAG_FILTER   1
#endif
#ifndef GL_TEXTURE_MIN_FILTER
#define GL_TEXTURE_MIN_FILTER   1
#endif
#ifndef GL_TEXTURE_WRAP_S
#define GL_TEXTURE_WRAP_S       1
#endif
#ifndef GL_TEXTURE_WRAP_T
#define GL_TEXTURE_WRAP_T       1
#endif
#ifndef GL_UNPACK_ALIGNMENT
#define GL_UNPACK_ALIGNMENT     1
#endif
#ifndef GL_TEXTURE_ENV
#define GL_TEXTURE_ENV          1
#endif
#ifndef GL_TEXTURE_ENV_MODE
#define GL_TEXTURE_ENV_MODE     1
#endif
#ifndef GL_MODULATE
#define GL_MODULATE             1
#endif
#ifndef GL_LUMINANCE
#define GL_LUMINANCE            1
#endif
#ifndef GL_UNSIGNED_BYTE
#define GL_UNSIGNED_BYTE        1
#endif
#ifndef GL_COLOR_BUFFER_BIT
#define GL_COLOR_BUFFER_BIT     1
#endif
#ifndef GL_DEPTH_BUFFER_BIT
#define GL_DEPTH_BUFFER_BIT     1
#endif
#ifndef GL_MODELVIEW
#define GL_MODELVIEW            1
#endif
#ifndef GL_QUADS
#define GL_QUADS                1
#endif
#ifndef GL_PROJECTION
#define GL_PROJECTION           1
#endif
#ifndef GL_DEPTH_TEST
#define GL_DEPTH_TEST           1
#endif

#ifdef __cplusplus
extern "C" {
#endif

void glutInit(int * argc, char * argv[]) { }
void glutMainLoop(void) { }
void glutPostRedisplay() { }
void glutReshapeFunc(void (*func)(int width, int height)) { }
void glutSwapBuffers(void) { }
void glutTimerFunc(float msecs, void (*func)(int value), int value) { }

void glutInitDisplayMode(int a) { }
void glutInitWindowSize(int width, int height) { }
void glutInitWindowPosition(int a, int b) { }
void glutCreateWindow(const char * str) { }

void glutKeyboardFunc(void (*func)(unsigned char a, int x, int y)) { }
void glutDisplayFunc(void (*func)(void)) { }

// #ifndef PV_USE_OPENCL  /* some GL files included from OpenCL */
// It appears that these functions are not in OpenCL anymore.

void glBegin(int a) { }
void glBindTexture(int a, int b) { }
void glClear(int a) { }
void glDisable(int a) { }
void glEnable(int a) { }
void glEnd(void) { }
void glLoadIdentity(void) { }
void glMatrixMode(int a) { }
void glPixelStorei(int a, int b) { }
void glPolygonMode(int a, int b) { }
void glTexCoord2f(float a, float b) { }
void glTexParameteri(int a, int b, int c) { }
void glTexEnvf(int a, int b, int c) { }
void glVertex3f(float a, float b, float c) { }
void glViewport(int a, int b, int width, int height) { }

void glTexImage2D(int a, int b, int c,
                  int width, int height, int d, int e, int f, void * buf) { }

void glRotatef(int a, int b, int c, int d) { }
void glScalef(int a, int b, int c) { }
void glTranslatef(int a, int b, int c) { }

// #endif /* PV_USE_OPENCL */

void gluPerspective(int a, float b, int c, int d) { }


#ifdef __cplusplus
}
#endif

#endif /* GL_STUBS_H_ */
