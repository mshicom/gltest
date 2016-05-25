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

def calcProjection(l, r, b, t, n, f):
    """ This creates a Projection matrix given 6 frustum
        params (l, r, b, t, n, f) for glFrustum()"""
    proj_matrix = np.identity(4)
    proj_matrix[0,0] = 2 * n / (r - l)
    proj_matrix[0,2] = (r + l) / (r - l)
    proj_matrix[1,1] = 2 * n / (t - b)
    proj_matrix[1,2] = (t + b) / (t - b)
    proj_matrix[2,2] = -(f + n) / (f - n)
    proj_matrix[2,3] = -(2 * f * n) / (f - n)
    proj_matrix[3,2] = -1
    proj_matrix[3,3] = 0
    return proj_matrix

def calcFrustum(fovY, aspectRatio, front, back):
    """ This creates a symmetric frustum. Given 4 params (fovy, aspect, near, far)
        it gives 6 params (l, r, b, t, n, f) for glFrustum()"""
    DEG2RAD = np.pi / 180

    tangent = tan(fovY/2.0 * DEG2RAD)   # tangent of half fovY
    height = front * tangent          # half height of near plane
    width = height * aspectRatio      # half width of near plane

    # params: left, right, bottom, top, near, far
#    glMatrixMode(GL_PROJECTION)
#    glLoadIdentity()
#    glFrustum(-width, width, -height, height, front, back)
#    glMatrixMode(GL_MODELVIEW)

    return calcProjection(-width, width, -height, height, front, back)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.yRotDeg = 0.0

    def buildShaders(self):
        vertex = shaders.compileShader("""#version 120
            varying vec4 vertex_color;
            uniform mat4 ProjectionMatrix;  // equal to gl_ProjectionMatrix
            void main() {
                gl_Position = ProjectionMatrix * (gl_ModelViewMatrix * gl_Vertex);
                vertex_color = gl_Color;
            }""",GL_VERTEX_SHADER)

        fragment = shaders.compileShader("""#version 120
            varying vec4 vertex_color;
            void main() {
                gl_FragColor = vertex_color;
            }""",GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vertex,fragment)
        self.uni_projection = glGetUniformLocation(self.shader, 'ProjectionMatrix')

        shaders.glUseProgram(self.shader)


    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0,  150))
        self.initGeometry()
        self.buildShaders()

        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        if height == 0: height = 1

        glViewport(0, 0, width, height)

        aspect = width / float(height)
        self.proj_mat = calcFrustum(45.0, aspect, 1.0, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslate(0.0, 0.0, -50.0)
        glScale(20.0, 20.0, 20.0)
        glRotate(self.yRotDeg, 0.2, 1.0, 0.3)
        glTranslate(-0.5, -0.5, -0.5)

        glUniformMatrix4fv(self.uni_projection, 1, GL_TRUE, self.proj_mat)


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointerf(self.cubeVtxArray)     # -> gl_Vertex
        glColorPointerf(self.cubeClrArray)      # -> gl_Color
        glDrawElementsui(GL_QUADS, self.cubeIdxArray)


    def initGeometry(self):
        self.cubeVtxArray = array(
                [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]])
        self.cubeIdxArray = [
                0, 1, 2, 3,
                3, 2, 6, 7,
                1, 0, 4, 5,
                2, 1, 5, 6,
                0, 3, 7, 4,
                7, 6, 5, 4 ]
        self.cubeClrArray = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0 ]]

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
        self.initMenus()

        glWidget = GLWidget(self)
        self.setCentralWidget(glWidget)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)
        QtCore.QObject.connect(timer, QtCore.SIGNAL('timeout()'), glWidget.spin)
        timer.start()


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
