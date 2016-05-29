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
pis = plt.imshow
pf = plt.figure
def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy.io
#%%

def loaddata1():
    data = scipy.io.loadmat('data.mat')
    frames, = data['I']
    G, = data['G']
    K = data['K']
    Z, = data['Z']/100.0
    return frames, G, K, Z

class camera:
        def __init__(self, fovY, aspectRatio, front, back):
            self.fovY = fovY
            self.aspectRatio = aspectRatio
            self.front = front
            self.back = back

        def getProjection(self):
            return calcFrustum(self.fovY, self.aspectRatio, self.front, self.back)

        @staticmethod
        def lookAt( eyePoint, targetPoint=np.array([0,0,0]), upVector=np.array([0,1,0]), isInverted=False):
            # step one: generate a rotation matrix

            z = eyePoint-targetPoint
            if np.all(z==0):
                z = np.array([0,0,1])
            x = np.cross(upVector, z)   # cross product
            y = np.cross(z, x)   # cross product

            normalize = lambda x: x/np.linalg.norm(x)
            x = normalize(x)
            y = normalize(y)
            z = normalize(z)
            eRo =  np.vstack([x, y, z])

            eMo = np.identity(4,dtype='f')
            eMo[0:3,0:3] = eRo
            eMo[0:3,3] = eRo.dot(-eyePoint)

            return eMo if not isInverted else np.linalg.inv(eMo)


        @staticmethod
        def calcProjection(l, r, b, t, n, f):
            proj_matrix = np.identity(4,'f')
            proj_matrix[0,0] = 2 * n / (r - l)
            proj_matrix[0,2] = (r + l) / (r - l)
            proj_matrix[1,1] = 2 * n / (t - b)
            proj_matrix[1,2]  = (t + b) / (t - b)
            proj_matrix[2,2] = -(f + n) / (f - n)
            proj_matrix[2,3] = -(2 * f * n) / (f - n)
            proj_matrix[3,2] = -1
            proj_matrix[3,3] = 0
            return proj_matrix

        @staticmethod
        def calcFrustum(fovY, aspectRatio, front, back):
            """ This creates a symmetric frustum.
                It converts to 6 params (l, r, b, t, n, f) for glFrustum()
                from given 4 params (fovy, aspect, near, far)"""
            DEG2RAD = np.pi / 180

            tangent = np.tan(fovY/2 * DEG2RAD)   # tangent of half fovY
            height = front * tangent          # half height of near plane
            width = height * aspectRatio      # half width of near plane

            # params: left, right, bottom, top, near, far
            # glFrustum(-width, width, -height, height, front, back)
            return camera.calcProjection(-width, width, -height, height, front, back)
#%%
class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.yRotDeg = 0.0
        self.setMouseTracking(True)

        targetPoint = np.array([0,0,0])
        upVector = np.array([0,1,0])

        self.eyePoint = np.array([0,0,2],'f')
        self.view_matrix = camera.lookAt(self.eyePoint)
        self.model_matrix = np.identity(4,'f')
        self.proj_matrix = camera.calcFrustum(60, 6.4/4.8, 0.9, 20)

    def buildShaders(self):
#        vertex = shaders.compileShader("""#version 430 core
#            layout (location = 0) in vec3 position;
#            layout (location = 4) in vec2 tc;
#            uniform mat4 model;
#            uniform mat4 view;
#            uniform mat4 projection;
#            out VS_OUT {
#                vec2 tc;
#            } vs_out;
#            void main(void)
#            {
#                vs_out.tc = tc;
#                gl_Position = projection*(view*(model*vec4(position,1)));
#            }""",GL_VERTEX_SHADER)
#
#        fragment = shaders.compileShader("""#version 430 core
#            layout (binding = 0) uniform sampler2D tex_object;
#
#            in VS_OUT {
#                vec2 tc;
#            } fs_in;
#
#            out vec4 color;
#            void main(void)
#            {
#                color.rgb = vec3( texture(tex_object, fs_in.tc).r);
#            }""",GL_FRAGMENT_SHADER)
#
#        self.shader = shaders.compileProgram(vertex,fragment)
#        self.attri_tc = 4
#
#        self.model_loc = glGetUniformLocation(self.shader, "model")
#        self.proj_loc = glGetUniformLocation(self.shader, "projection")
#        self.view_loc = glGetUniformLocation(self.shader, "view")
        vertex = shaders.compileShader("""#version 430 core
            layout (location = 0) in vec3 position;
            layout (location = 4) in float color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat4 pTe,projection_pro;
            layout (binding = 0) uniform sampler2D tex_object;
            out VS_OUT {
                float c;
            } vs_out;
            void main(void)
            {
                vec4 pos_eye = model*vec4(position,1);
                gl_Position = projection*(view*pos_eye);
                vec4 pos_proj =  projection_pro*pTe*pos_eye;
                vec2 tc = pos_proj.rg / pos_proj.b;
                if(tc.r>-1 && tc.r<1 && tc.g>-1 && tc.g<1)
                    vs_out.c = texture(tex_object, tc).r;
                else
                    vs_out.c = 1;

            }""",GL_VERTEX_SHADER)

        fragment = shaders.compileShader("""#version 430 core
             in VS_OUT {
                float c;
            } fs_in;

            out vec4 color;
            void main(void)
            {
                color.rgb = vec3( fs_in.c);
            }""",GL_FRAGMENT_SHADER)

        self.shader = shaders.compileProgram(vertex,fragment)
        self.attri_tc = 4

        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.projection_pro_loc = glGetUniformLocation(self.shader, "projection_pro")
        self.pTe_loc = glGetUniformLocation(self.shader, "pTe")

        #tex_data = (plt.imread('/home/kaihong/workspace/0.png')*255).astype('uint8')
        tex_data = frames[1]
        if not np.can_cast(tex_data.dtype, 'uint8'):
            raise ValueError("texture data should be of type uint8")

        h,w = tex_data.shape[:2]
        self.tex = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0,
                        GL_RED,
                        w, h,
                        0,GL_RED,GL_UNSIGNED_BYTE,
                        tex_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0,  0))
        self.initGeometry()
        self.buildShaders()

        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)
#        self.camera.setViewportDimensions(width, height)


#    def paintGL(self):
#        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#
#        try:
#            self.quad.bind()
#            glUseProgram(self.shader)
#
#            glUniformMatrix4fv(self.model_loc, 1, GL_TRUE, self.model_matrix)
#            glUniformMatrix4fv(self.proj_loc, 1, GL_TRUE, self.proj_matrix)
#            glUniformMatrix4fv(self.view_loc, 1, GL_TRUE, self.view_matrix)
#
#            glActiveTexture(GL_TEXTURE0)
#            glBindTexture(GL_TEXTURE_2D, self.tex)
#
#            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, self.quad)
#            glEnableVertexAttribArray(0)
#            glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 20, self.quad+12)
#            glEnableVertexAttribArray(4)
#
#            glDrawArrays(GL_QUADS,0, 4)
#
#            glDisableVertexAttribArray(0)
#            glDisableVertexAttribArray(4)
#        finally:
#            self.quad.unbind()
#            glUseProgram(0)
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        try:
            self.p.bind()
            glUseProgram(self.shader)

            glUniformMatrix4fv(self.model_loc, 1, GL_TRUE, self.model_matrix)
            glUniformMatrix4fv(self.proj_loc, 1, GL_TRUE, self.proj_matrix)
            glUniformMatrix4fv(self.view_loc, 1, GL_TRUE, self.view_matrix)

            glUniformMatrix4fv(self.projection_pro_loc, 1, GL_TRUE, self.proj_matrix)
#            glUniformMatrix4fv(self.pTe_loc, 1, GL_TRUE, np.dot(np.linalg.inv(G[1]), G[0]))
            glUniformMatrix4fv(self.pTe_loc, 1, GL_TRUE,
                               camera.lookAt(np.array([-1,0,0]),
                                             np.array([0,0,0]),
                                             np.array([0,1,0])))
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glBindVertexArray(self.vao)

            glDrawArrays(GL_POINTS, 0, imheight*imwidth)

            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)

        finally:
            self.I.unbind()
            glUseProgram(0)

    def initGeometry(self):
        self.quad = VBO(np.array([[-1,-1, 0, 0, 1],
                                  [ 1,-1, 0, 1, 1],
                                  [ 1, 1, 0, 1, 0],
                                  [-1, 1, 0, 0, 0]],'f'), GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        self.p = VBO(P, GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        self.I = VBO(I, GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        """ (optional) setup VAO configuration Macro"""
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.p.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, self.p)
        glEnableVertexAttribArray(0)
        self.I.bind()
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, self.I)
        glEnableVertexAttribArray(4)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


    def spin(self):
        self.yRotDeg = (self.yRotDeg  + 1) % 360.0
        self.parent.statusBar().showMessage('rotation %f' % self.yRotDeg)
        self.updateGL()

    def mouseMoveEvent(self, mouseEvent):
        button = int(mouseEvent.buttons())

        if button != QtCore.Qt.NoButton :
            # user is dragging
            delta_x = (self.oldx - mouseEvent.x())/float(imwidth)
            delta_y = ( self.oldy - mouseEvent.y())/float(imheight)

            if button & QtCore.Qt.LeftButton :
                """ rotate the scene """
                deltaT = fromEuler(delta_y, delta_x).matrix()
                self.model_matrix=deltaT.dot(self.model_matrix)
            elif button & QtCore.Qt.MidButton :
                pass
            self.update()
        self.oldx = mouseEvent.x()
        self.oldy = mouseEvent.y()

    def wheelEvent(self, e):
        # QtGui.QWheelEvent(e)
        """ zoom in """
        self.eyePoint[2] += 0.1 if e.delta()>0 else -0.1
        self.view_matrix = camera.lookAt(self.eyePoint)
        self.updateGL()

    def closeEvent(self):
        glUseProgram(0)
        glDeleteProgram(self.shader)
        self.p.delete()
        self.I.delete()


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

        self.resize(imwidth, imheight)
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
    if 'frames' not in globals() or 1:
        frames, wGc, K, Z = loaddata1()
        imheight,imwidth = frames[0].shape[:2]
    plt.close('all')
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    refid, curid = 0,2
    Iref, G0, Z = frames[refid]/255.0, wGc[refid], wGc[refid]
    Icur, G1  = frames[curid]/255.0, wGc[curid]
    Ki = np.linalg.inv(K)
    cGr = np.dot(np.linalg.inv(G1), G0)
    R, T = cGr[0:3,0:3], cGr[0:3,3]

    dy,dx = np.gradient(Iref)
    dI = np.sqrt(dx**2+dy**2)
    valid_mask = dI>0.1
    u,v = np.where(valid_mask)
    color = dI[valid_mask]

    f,a = plt.subplots(1,1,num='epiline')
    a.imshow(sim(Icur,Iref))
    p = np.round(plt.ginput(1, timeout=-1)[0]).reshape(-1,1)
    a.plot(p[0], p[1],'*')
    ld = np.array( [-fx*T[0] + T[2]*(p[0]-cx),
                    -fy*T[1] + T[2]*(p[1]-cy)])
    ep = np.linspace(0,1,20)*ld+p
    a.plot(ep[0]+640,ep[1],'.')
    a.plot([T[0]/T[2]*fx+640,p[0]+640],
           [T[1]/T[2]*fy,p[1]],'r-')

#    app_created = False
#    app = QtCore.QCoreApplication.instance()
#    if app is None:
#        app = QtGui.QApplication(sys.argv)
#        app_created = True
#    app.references = set()
#    window = MainWindow()
#    app.references.add(window)
#    window.show()
#    if app_created:
#        app.exec_()
