#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:26:01 2016

@author: kaihong
"""
import sys
from PyQt4 import QtCore,QtGui,QtOpenGL
from OpenGL import GLU
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays.vbo import VBO
import numpy as np
from OpenGLContext.quaternion import Quaternion

def calcProjection(l, r, b, t, n, f):
    proj_matrix = np.identity(4)
    proj_matrix[0,0] = 2 * n / (r - l)
    proj_matrix[0,2] = (r + l) / (r - l)
    proj_matrix[1,1] = 2 * n / (t - b)
    proj_matrix[1,2]  = (t + b) / (t - b)
    proj_matrix[2,2] = -(f + n) / (f - n)
    proj_matrix[2,3] = -(2 * f * n) / (f - n)
    proj_matrix[3,2] = -1
    proj_matrix[3,3] = 0
    return proj_matrix


def calcFrustum(fovY, aspectRatio, front, back):
    """ This creates a symmetric frustum.
        It converts to 6 params (l, r, b, t, n, f) for glFrustum()
        from given 4 params (fovy, aspect, near, far)"""
    DEG2RAD = 3.14159265 / 180

    tangent = tan(fovY/2 * DEG2RAD)   # tangent of half fovY
    height = front * tangent          # half height of near plane
    width = height * aspectRatio      # half width of near plane

    # params: left, right, bottom, top, near, far
    # glFrustum(-width, width, -height, height, front, back)
    return calcProjection(-width, width, -height, height, front, back)


class GLWidget(QtOpenGL.QGLWidget):
    """ http://doc.qt.io/qt-5/qglwidget.html#details """
    def __init__(self, parent=None):

        self.parent = parent
        self.glformat = QtOpenGL.QGLFormat.defaultFormat()
        self.glformat.setDepth(True)
        self.glformat.setDoubleBuffer(True)
        super(GLWidget, self).__init__(self.glformat, parent)
        self.initGeometry()
        self.setMouseTracking(True)

        self.isPressed = False
        self.oldx,self.oldy = 0,0
        self.yRotDeg = 0.0

        self.proj_matrix = np.identity(4)

    def buildShaders(self):
        VERTEX_SHADER = shaders.compileShader("""
            #version 430 core
            uniform mat4 mv_matrix;
            uniform mat4 proj_matrix;
            layout (location = 0) in vec3 position;
            void main(void)
            {
                // Calculate the position of each vertex
                gl_Position = proj_matrix * mv_matrix * vec4(position, 1.0);
            }""", GL_VERTEX_SHADER)

        FRAGMENT_SHADER = shaders.compileShader("""
            #version 430 core
            out vec4 color;            // Output to framebuffer
            void main(void)
            {
                color = vec4(1.0, 0.0, 0.0, 1.0);
            }""", GL_FRAGMENT_SHADER)

        self.shader_pipline = shaders.compileProgram(VERTEX_SHADER,
                                                     FRAGMENT_SHADER)
        self.mv_loc = glGetUniformLocation(self.shader_pipline, "mv_matrix")
        self.proj_loc = glGetUniformLocation(self.shader_pipline, "proj_matrix")


    def initializeGL(self):
        self.makeCurrent()

        self.buildShaders()

        glClearColor(0, 0, 0.5, 1.0)
        glEnable(GL_DEPTH_TEST)
#        glEnable(GL_CULL_FACE)


    def resizeGL(self, width, height):
        if height == 0: height = 1

        glViewport(0, 0, width, height)
        aspect = width / float(height)
        self.proj_matrix = calcFrustum(45.0, aspect, 0.1, 10.0)


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_pipline)

        self.cubeVtx_vbo.bind()
        mv_matrix = np.identity(4)
        mv_matrix[2,3] = -5

        glUniformMatrix4dv(self.mv_loc, 1, GL_TRUE, mv_matrix)
        glUniformMatrix4dv(self.proj_loc, 1, GL_TRUE, self.proj_matrix)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glDrawArrays(GL_TRIANGLES, 0, 6)
        #glDrawElementsui(GL_QUADS, self.cubeIdxArray)

        glDisableVertexAttribArray(0)
        self.cubeVtx_vbo.unbind()
        glUseProgram(0)

        self.swapBuffers()

    def initGeometry(self):
        self.cubeVtx_vbo = VBO(np.array(
                [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]],'f'), GL_STATIC_DRAW,GL_ARRAY_BUFFER)
        self.cubeIdxArray = np.array(
               [0, 1, 2, 3,
                3, 2, 6, 7,
                1, 0, 4, 5,
                2, 1, 5, 6,
                0, 3, 7, 4,
                7, 6, 5, 4 ],'f')


    def spin(self):
        self.yRotDeg = (self.yRotDeg  + 1) % 360.0
        self.parent.statusBar().showMessage('rotation %f' % self.yRotDeg)
        self.updateGL()

    def mouseMoveEvent(self, mouseEvent):
        if int(mouseEvent.buttons()) != QtCore.Qt.NoButton :
            # user is dragging
            delta_x = mouseEvent.x() - self.oldx
            delta_y = self.oldy - mouseEvent.y()
            if int(mouseEvent.buttons()) & QtCore.Qt.LeftButton :
                    pass
#                    self.camera.orbit(self.oldx,self.oldy,mouseEvent.x(),mouseEvent.y())
            elif int(mouseEvent.buttons()) & QtCore.Qt.MidButton :
                pass
#                self.camera.translateSceneRightAndUp( delta_x, delta_y )
            self.update()
        self.oldx = mouseEvent.x()
        self.oldy = mouseEvent.y()

#    def mouseDoubleClickEvent(self, mouseEvent):
#        print "double click"
#
#    def mousePressEvent(self, e):
#        print "mouse press"
#        self.isPressed = True
#
#    def mouseReleaseEvent(self, e):
#        print "mouse release"
#        self.isPressed = False

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(640, 480)
        self.setWindowTitle('GL Cube Test')

        self.initActions()
        self.initMenus()

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