if __name__ == '__build__':
   raise Exception

import string
__version__ = string.split('$Revision: 0.1 $')[1]
__date__ = string.join(string.split('$Date: 12/19/2009$')[1:3], ' ')
__author__ = 'Soren Christian Rasmussen <soren.rasmussen@aggiemail.usu.edu>'


'''Global parameters'''
glWindowWidth = 512
glWindowHeight = 512
xScale = 2.0
yScale = 2.0
g_rs = None
g_nx = None
g_ny = None
g_msecs = 100
g_z = 0
nf = 1
activity = []
ESCAPE = '\033'

from PVTransforms import *
from PVReadSparse import *
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import sys

def drawRect(x, y, dx, dy):
   gl.glBegin(gl.GL_QUADS) #Draw a square
   gl.glVertex3f(x+dx, -y+0.0, g_z)
   gl.glVertex3f(x+dx, -y+dy, g_z)
   gl.glVertex3f(x+0.0, -y+dy, g_z)
   gl.glVertex3f(x+0.0, -y+0.0, g_z)
   gl.glEnd() #Done with the polygon
#end drawRect
   
def reSizeGLScene(wWidth, wHeight):
      if wHeight == 0:				      # Prevent A Divide By Zero If The Window Is Too Small 
            wHeight = 1

      gl.glViewport(0, 0, wWidth, wHeight)		# Reset The Current Viewport And Perspective Transformation
      gl.glMatrixMode(gl.GL_PROJECTION)
      gl.glLoadIdentity()
      glu.gluPerspective(45.0, float(wWidth)/float(wHeight), 0.1, 100.0)
      gl.glMatrixMode(gl.GL_MODELVIEW)
#end reSizeGLScene

def drawGLScene():
   dx = xScale/g_nx
   dy = yScale/g_ny

   gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

   gl.glColor3f( 0.8,0.6, 0.0)
   
   for k in activity:
      kx = kxPos(k, g_nx, g_ny, nf)
      ky = kyPos(k, g_nx, g_ny, nf)
      x = xScale*kx/g_nx
      y = yScale*ky/g_ny
      drawRect(x,y,dx,dy)

      '''draw borders'''
      #drawRect(0.0, 0.0, dx, dy)
      #drawRect(xScale*1.0, 0.0, dx, dy)
      #drawRect(xScale*1.0, yScale*1.0, dx, dy)
      #drawRect(0.0, yScale*1.0, dx, dy)

   #since this is double buffered, swap the buffers to display what just got drawn. 
   glut.glutSwapBuffers()
   
#end drawGLScene
   
def keyPressed(*args):
   if args[0] == ESCAPE:
         sys.exit()
   else:
      print "key pressed"
      print "press esc to exit"
#end keyPressed    

def getNextRecord(value):
   global activity
   
   try:
      activity = g_rs.next_record()
   except:
      print "end of file detected"
      sys.exit()

   if len(activity) > 0:
      print "time==", g_rs.time, "num_records==", len(activity)
      glut.glutPostRedisplay()
      glut.glutTimerFunc(g_msecs, getNextRecord, 1)
   else:
      print "time==", g_rs.time, "num_records==", len(activity)
      glut.glutTimerFunc(g_msecs, getNextRecord, 1)
#end getNextRecord

class GLPainter:
   def __init__ (self, wWidth, wHeight, filename):
      global g_rs, g_nx, g_ny, g_msecs

      g_rs = PVReadSparse(filename)
      g_nx = g_rs.nx
      g_ny = g_rs.ny

      glut.glutInit(sys.argv)
      glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
      glut.glutInitWindowSize(wWidth, wHeight)
      glut.glutInitWindowPosition(0, 0)
      window = glut.glutCreateWindow("PetaVision Activity")
      gl.glTranslatef( -1.0 , 1.0 , -2.5)
      
      # register callbacks
      glut.glutReshapeFunc(reSizeGLScene)
      glut.glutTimerFunc(g_msecs, getNextRecord, 1)
      glut.glutDisplayFunc(drawGLScene)
      glut.glutKeyboardFunc(keyPressed)
      
      print "nx==", g_nx
      print "ny==", g_ny

      self.initGL(wWidth, wHeight)
      
   def initGL (self, wWidth, wHeight):
      gl.glClearColor(0.0, 0.0, 0.25, 0.0)	# This Will Clear The Background Color To Black
      gl.glClearDepth(1.0)			# Enables Clearing Of The Depth Buffer
      gl.glDepthFunc(gl.GL_LESS)		# The Type Of Depth Test To Do
      gl.glEnable(gl.GL_DEPTH_TEST)		# Enables Depth Testing
      gl.glShadeModel(gl.GL_SMOOTH)		# Enables Smooth Color Shading

      gl.glMatrixMode(gl.GL_PROJECTION)
      gl.glLoadIdentity()			# Reset The Projection Matrix
						# Calculate The Aspect Ratio Of The Window
      glu.gluPerspective(45.0, float(wWidth)/float(wHeight), 0.1, 100.0)

      gl.glMatrixMode(gl.GL_MODELVIEW)
# end class GLPainter


def main():
   import sys
   global window

   if len(sys.argv) < 2:
      print "\nusage: python GLPainter.py filename\n"
      return

   p = GLPainter(glWindowWidth, glWindowHeight, sys.argv[1])

   print type(g_rs)
   glut.glutMainLoop()


# end main()

main()
