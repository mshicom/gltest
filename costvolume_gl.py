#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:36:17 2016

@author: kaihong
"""
import sys
from PyQt4 import QtCore,QtGui,QtOpenGL
from OpenGL import GLU
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays.vbo import VBO
from OpenGLContext.quaternion import  fromEuler
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tools import *
#%%
frames, wGc, K, Z = loaddata1()
imheight,imwidth = frames[0].shape[:2]

fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

#%%
class FBO_Test():
    def __init__():
        self.drawbuffer = []

    def generateColorTexture(width, height):
        pass

    def GenerateFBO(self, width, height):

        # 1. Generate a framebuffer object(FBO) and target texture
        self.FBO = glGenFramebuffers(1)
        self.texture_color = generateColorTexture(width, height)

        # 2. Setup the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.FBO)         # bind it to the pipeline to make it active

        # 2a. bind textures to pipeline
        attachment_index_color_texture = 0                  # bookeeping
        glFramebufferTexture(GL_FRAMEBUFFER,
                             GL_COLOR_ATTACHMENT0 + attachment_index_color_texture,
                             self.texture_color,
                             0)                             # the mipmap level.
        self.drawbuffer.append(GL_COLOR_ATTACHMENT0 + attachment_index_color_texture)    # bookeeping

        # 3. Check for FBO completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError( "Error! FrameBuffer is not complete")
        # 4. Done & de-activate it
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

    def buildShaders(self):
        imheight,imwidth = frames[0].shape[:2]

        A,B,C,D = 0., imwidth-1., 0., imheight-1.
        map0 = np.array([[2/(A-B),   0, 1-2*A/(A-B)],
                         [0,   2/(C-D), 1-2*C/(C-D)]])

        map1 = np.array([[1/(B-A),   0, -A/(B-A)],
                         [0,   1/(D-C), -C/(D-C)]])

        vsrc = """#version 130
            in vec2 p_ref;        // p_ref = [x, y]
            in float pcolor;
            uniform mat3 H;
            const mat3x2 map0 = mat3x2( %(map0)s );
            const mat3x2 map1 = mat3x2( %(map1)s );
            uniform sampler2D im;
            out float c;

            const float width = %(width)d-1, height = %(height)d-1;

            bool isInImage(vec2 p)
            {    return p.x>0 && p.x<width && p.y>0 && p.y<height;  }   // 1 pixel border gap

            vec2 toNDC(vec2 p)
            {   return map0*vec3(p,1); }    // map from [0:h,0:w] to [-1:1,-1:1]

            vec2 toTEX(vec2 p)
            {   return map1*vec3(p,1); }    // map from [0:h,0:w] to [0:1,0:1]

            void main(void)
            {
                vec3 p_cur = H*vec3(p_ref,1);      // := K*R*inv(K)*p_ref+K*T/d
                vec2 tc = p_cur.xy / p_cur.z;
                bool isValid = isInImage(tc);
                if(isValid)
                    c = abs(pcolor - texture2D(im, toTEX(tc)).r );
                else
                    c = pcolor;
                gl_Position.xy = toNDC(p_ref);
                gl_Position.zw = vec2(0,1);

            }""" % {"map0": "".join(str(v)+"," for v in map0.ravel('F'))[:-1],
                    "map1": "".join(str(v)+"," for v in map1.ravel('F'))[:-1],
                    "width": imwidth,
                    "height": imheight }

        vertex = shaders.compileShader(vsrc, GL_VERTEX_SHADER)

        fragment = shaders.compileShader("""#version 130
            in float c;
            out vec4 color;
            void main(void)
            {
                color.rgb = vec3( c);
            }""",GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vertex,fragment)

        self.p_ref_loc = glGetAttribLocation(self.shader, 'p_ref')
        self.color_loc = glGetAttribLocation(self.shader, 'pcolor')

        self.H_loc = glGetUniformLocation(self.shader, "H")
        self.map0_loc = glGetUniformLocation(self.shader, "map0")
        self.map1_loc = glGetUniformLocation(self.shader, "map1")
        self.im_loc = glGetUniformLocation(self.shader, "im")

    def initGeometry(self):
        """ setup image texture """
        tex_data = np.ascontiguousarray(frames[-1]/255.0, 'f')

        h,w = tex_data.shape[:2]
        self.tex = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0,
                        GL_RED,
                        w, h,
                        0,GL_RED,GL_FLOAT,
                        tex_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        """ """
        y, x = np.mgrid[0:h, 0:w]
        P = np.vstack([x.ravel(),y.ravel()])
        P = np.ascontiguousarray(P.T, 'f')
        self.p = VBO(P, GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        self.im = np.ascontiguousarray(frames[0]/255.0, 'f')
        self.I = VBO(self.im, GL_STATIC_DRAW, GL_ARRAY_BUFFER)

        """ (optional) setup VAO configuration Macro"""
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.p.bind()
        glVertexAttribPointer(self.p_ref_loc,   # index
                              2,                # number of components per vertex attribute
                              GL_FLOAT,         # type
                              GL_FALSE,         # normalized
                              0,                # stride, byte offset between consecutive vertex attributes
                              self.p)           # *pointer
        glEnableVertexAttribArray(self.p_ref_loc)

        if self.color_loc!=-1:
            self.I.bind()
            glVertexAttribPointer(self.color_loc,
                                  1,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  0,
                                  self.I)
            glEnableVertexAttribArray(self.color_loc)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        cGr = np.dot(np.linalg.inv(wGc[-1]), wGc[0])
        R,T = cGr[:3,:3], cGr[:3,3]
        self.M1 = K.dot(R).dot(inv(K))
        self.M2 = K.dot(T)
        self.d = 2.0

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0,  0))

        glViewport(0, 0, 640, 480)
        self.buildShaders()
        self.initGeometry()

        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        H = self.M1.copy()
        H[:, 2] += self.M2/self.d
        try:
            self.p.bind()
            self.I.bind()
            glUseProgram(self.shader)

            glUniformMatrix3fv(self.H_loc, 1, GL_TRUE, H.astype('f')) # location,count,transpose,*value

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.tex)

            glBindVertexArray(self.vao)
            glDrawArrays(GL_POINTS, 0, imheight*imwidth)    # mode,first,count

        finally:

            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            self.p.unbind()
            self.I.unbind()
            glUseProgram(0)

    def wheelEvent(self, e):
        # QtGui.QWheelEvent(e)
        """ zoom in """
        inc = 0.2 if e.delta()>0 else -0.2
        self.d = np.clip(self.d+inc, 0.1, 5.0)
        print self.d
        self.updateGL()

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(imwidth, imheight)
        self.setWindowTitle('GL Cube Test')

        self.initActions()

        self.glWidget = GLWidget(self)

        self.setCentralWidget(self.glWidget)

    def initActions(self):
        self.exitAction = QtGui.QAction('Quit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), self.close)

    def close(self):
        QtGui.qApp.quit()

if __name__ == "__main__":
    app_created = False
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
        app_created = True
    app.references = set()
    window = MainWindow()
    app.references.add(window)
    window.show()
    if app_created:
        app.exec_()