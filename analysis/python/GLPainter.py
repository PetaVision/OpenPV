if __name__ == '__build__':
   raise Exception

import string
__version__ = string.split('$Revision: 0.1 $')[1]
__date__ = string.join(string.split('$Date: 12/19/2009$')[1:3], ' ')
__author__ = 'Soren Christian Rasmussen <soren.rasmussen@aggiemail.usu.edu>'


'''Global parameters'''
glWindowWidth = 512
glWindowHeight = 512
nx = 64
ny = 64

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys



def drawRec(x, y, dx, dy):
   glVertex3f(x+dx, y+0.0, 0.0)
   glVertex3f(x+dx, y+dy, 0.0)
   glVertex3f(x+0.0, y+dy, 0.0)
   glVertex3f(x+0.0, y+0.0, 0.0)
#end drawRect
   
def reSizeGLScene(wWidth, wHeight):
      if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
            Height = 1

      glViewport(0, 0, wWidth, wHeight)		# Reset The Current Viewport And Perspective Transformation
      glMatrixMode(GL_PROJECTION)
      glLoadIdentity()
      gluPerspective(45.0, float(wWidth)/float(wHeight), 0.1, 100.0)
      glMatrixMode(GL_MODELVIEW)
#end reSizeGLScene

def drawGLScene():
   dx = 1.0/nx
   dy = 1.0/ny
   
   # Clear The Screen And The Depth Buffer
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   glLoadIdentity()					# Reset The View 

   glTranslatef(-1.0, -1.0, -2.5)

   glColor3f(0.8,0.6,0.0)
      
   # Draw a square (quadrilateral)
   glBegin(GL_QUADS)                   # Start drawing a 4 sided polygon
   drawRec(0,0,dx,dy)
   drawRec(0,1.7,dx,dy)
   drawRec(1.7,0,dx,dy)
   drawRec(1.7,1.7,dx,dy)

   glEnd()                             # We are done with the polygon

   #  since this is double buffered, swap the buffers to display what just got drawn. 
   glutSwapBuffers()
#end drawGLScene
   
def keyPressed(*args):
   print "key pressed"
   # If escape is pressed, kill everything.
   if args[0] == ESCAPE:
         sys.exit()
#end keyPressed    


class GLPainter:
   def __init__ (self, width, height):
      glutInit(sys.argv)
      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
      glutInitWindowSize(width, height)
      glutInitWindowPosition(100, 100)
      window = glutCreateWindow("Soren's GL Box")
      glutDisplayFunc(drawGLScene)

      print "nx==", nx
      print "ny==", ny
      
      self.initGL(width, height)
   

   def initGL (self, wWidth, wHeight):
      glClearColor(0.0, 0.0, 0.25, 0.0)	# This Will Clear The Background Color To Black
      glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
      glDepthFunc(GL_LESS)				# The Type Of Depth Test To Do
      glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
      glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading

      glMatrixMode(GL_PROJECTION)
      glLoadIdentity()					# Reset The Projection Matrix
						    	# Calculate The Aspect Ratio Of The Window
      gluPerspective(45.0, float(wWidth)/float(wHeight), 0.1, 100.0)

      glMatrixMode(GL_MODELVIEW)


# end class GLPainter


def main():
   global window
   p = GLPainter(glWindowWidth, glWindowHeight)
   glutMainLoop()
# end main()

main()
