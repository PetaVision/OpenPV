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

#define GL_FILL                 1
#define GL_FRONT_AND_BACK       1
#define GL_LINEAR               1
#define GL_REPEAT               1
#define GL_TEXTURE_2D           1
#define GL_TEXTURE_MAG_FILTER   1
#define GL_TEXTURE_MIN_FILTER   1
#define GL_TEXTURE_WRAP_S       1
#define GL_TEXTURE_WRAP_T       1
#define GL_UNPACK_ALIGNMENT     1
#define GL_TEXTURE_ENV          1
#define GL_TEXTURE_ENV_MODE     1
#define GL_MODULATE             1
#define GL_LUMINANCE            1
#define GL_UNSIGNED_BYTE        1
#define GL_COLOR_BUFFER_BIT     1
#define GL_DEPTH_BUFFER_BIT     1
#define GL_MODELVIEW            1
#define GL_QUADS                1
#define GL_PROJECTION           1
#define GL_DEPTH_TEST           1

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

void gluPerspective(int a, float b, int c, int d) { }


#ifdef __cplusplus
}
#endif

#endif /* GL_STUBS_H_ */
