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
import numpy as np
import matplotlib.pyplot as plt

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.yRotDeg = 0.0

    def buildShaders(self):
        vertex = shaders.compileShader("""#version 430 core
            layout (location = 0) in vec3 position;
            layout (location = 4) in vec2 tc;
            out VS_OUT {
                vec2 tc;
            } vs_out;
            void main(void)
            {
                vs_out.tc = tc;
                gl_Position = vec4(position,1);
            }""",GL_VERTEX_SHADER)

        fragment = shaders.compileShader("""#version 430 core
            layout (binding = 0) uniform sampler2D tex_object;

            in VS_OUT {
                vec2 tc;
            } fs_in;

            out vec4 color;
            void main(void)
            {
                color = texture(tex_object, fs_in.tc);
            }""",GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vertex,fragment)
        self.attri_tc = 4
        shaders.glUseProgram(self.shader)

        tex_data = plt.imread('/home/nubot/data/workspace/vdtm_rect_linux/Tsukuba/scene1.0.bmp')
        h,w = tex_data.shape[:2]

        self.tex = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0,
                        GL_RGB,
                        w, h,
                        0,GL_RGB,GL_UNSIGNED_BYTE,
                        tex_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0,  150))
        self.initGeometry()
        self.buildShaders()

        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)



    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        with self.quad:
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, self.quad)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 20, self.quad+12)
            glEnableVertexAttribArray(4)

            glDrawArrays(GL_QUADS,0, 4)

            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(4)


    def initGeometry(self):
        self.quad = VBO(np.array([[-1,-1, 0, 0, 1],
                                  [ 1,-1, 0, 1, 1],
                                  [ 1, 1, 0, 1, 0],
                                  [-1, 1, 0, 0, 0]],'f'), GL_STATIC_DRAW, GL_ARRAY_BUFFER)



    def spin(self):
        self.yRotDeg = (self.yRotDeg  + 1) % 360.0
        self.parent.statusBar().showMessage('rotation %f' % self.yRotDeg)
        self.updateGL()

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(300, 300)
        self.setWindowTitle('GL Cube Test')

        self.initActions()
#        self.initMenus()

        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        QtCore.QObject.connect(timer, QtCore.SIGNAL('timeout()'), self.glWidget.spin)
#        timer.start()

    def initActions(self):
        self.exitAction = QtGui.QAction('Quit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), self.close)

    def initMenus(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(self.exitAction)

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
