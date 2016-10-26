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
h,w = frames[0].shape[:2]

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


class PlaneSweeper():
    def __init__(self, height, width):
        """ 1.Calculate NDC/Texture mapping matrix"""
        xmin,xmax,ymin,ymax = 0.0, width-1.0, 0.0, height-1.0

        mapNDC = np.array([[2/(xmin-xmax),   0, 1-2*xmin/(xmin-xmax)],
                           [0,   2/(ymin-ymax), 1-2*ymin/(ymin-ymax)]])

        mapTEX = np.array([[1/(xmax-xmin),   0, -xmin/(xmax-xmin)],
                           [0,   1/(ymax-ymin), -ymin/(ymax-ymin)]])

        """ 2.Create and compile shader code"""
        vsrc = """#version 130
            in vec2 p_ref;        // p_ref = [x, y]
            in float pcolor;
            uniform mat3 H;
            uniform sampler2DArray im_cur;

            out float c;

            const float width = %(width)d-1, height = %(height)d-1;
            bool isInImage(vec2 p)
            {    return p.x>0 && p.x<width && p.y>0 && p.y<height;  }   // 1 pixel border gap

            const mat3x2 mapNDC = mat3x2( %(mapNDC)s );
            vec2 toNDC(vec2 p)
            {   return mapNDC*vec3(p,1); }    // map from [0:h,0:w] to [-1:1,-1:1]

            const mat3x2 mapTEX = mat3x2( %(mapTEX)s );
            vec2 toTEX(vec2 p)
            {   return mapTEX*vec3(p,1); }    // map from [0:h,0:w] to [0:1,0:1]

            void main(void)
            {
                vec3 p_cur = H*vec3(p_ref,1);      // := K*R*inv(K)*p_ref+K*T/d
                vec2 tc = p_cur.xy / p_cur.z;
                bool isValid = isInImage(tc);
                if(isValid) {
                    c = abs( pcolor-texture(im_cur, vec3(toTEX(tc), 0)).r );
                }
                else
                    c = pcolor;
                gl_Position.xy = toNDC(p_ref);
                gl_Position.zw = vec2(0,1);

            }""" % {"mapNDC": "".join(str(v)+"," for v in mapNDC.ravel('F'))[:-1],
                    "mapTEX": "".join(str(v)+"," for v in mapTEX.ravel('F'))[:-1],
                    "width": width,
                    "height": height }
        vertex = shaders.compileShader(vsrc, GL_VERTEX_SHADER)
        fragment = shaders.compileShader("""#version 130
            in float c;
            out vec4 color;
            void main(void)
            {
                color.rgb = vec3(c);
            }
            """,GL_FRAGMENT_SHADER)
        shader = shaders.compileProgram(vertex,fragment)
        self.__shader = shader

        """ 3.Extract parameter locations"""
        attr = {"p_ref" : glGetAttribLocation(shader, "p_ref"),
                "color" : glGetAttribLocation(shader, "pcolor")}
        unif = {"H"     : glGetUniformLocation(shader, "H")}
        # set sampler in pos 0
        glUseProgram(shader)
        glUniform1i(glGetUniformLocation(shader, "im_cur"), 0)
        glUseProgram(0)

        """ 4.Create constant vbo for p_ref"""
        y, x = np.mgrid[0:height, 0:width]
        P = np.vstack([x.ravel(), y.ravel()]).T
        P = np.ascontiguousarray(P, 'f')
        p_ref_vbo = VBO(P, GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        self.__p_ref_vbo = p_ref_vbo

        """ 5.Create mutable vbo for reference pixel data """
        ref_im = np.ones((height,width), 'uint8')
        color_vbo = VBO(ref_im, GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        self.isRefSetted = False
        def setRefImage(image):
            if image.dtype != np.uint8:
                raise RuntimeError("image must be uint8")
            if image.shape != (height,width):
                raise RuntimeError("image size not matched")
            with color_vbo:
                color_vbo.set_array(image)
                color_vbo.copy_data()           # send data to gpu
                self.isRefSetted = True
        self.setRefImage = setRefImage
        self.__color_vbo = color_vbo

        """ 6.Create texture array for cur image"""
        max_images = 10
        im_cur_tex = glGenTextures(1)
#        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
        glTexStorage3D(GL_TEXTURE_2D_ARRAY,
                       1,             # mip map levels
                       GL_R8,         #
                       width, height, # shape
                       max_images)    # total num of layers/depth
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self.isCurSetted = False
        def setCurImage(images):
            if len(images) > max_images:
                raise RuntimeWarning("More than 10 images")
                images = images[:10]
#            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
            for layer,image in enumerate(images):
                if image.dtype != np.uint8:
                    raise RuntimeError("image must be uint8")
                if image.shape != (height,width):
                    raise RuntimeError("image size not matched")
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY,    # target,
                             0,0,0,layer,                   # mid map level, x/y/z-offset
                             w, h, 1,               # width, height,layers
                             GL_RED,            # format of the pixel data, i.e. GL_RED, GL_RG, GL_RGB,
                             GL_UNSIGNED_BYTE,  # data type of the pixel data, i.e GL_UNSIGNED_BYTE, GL_FLOAT ...
                             image)             # data pointer
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
            self.isCurSetted = True
        self.setCurImage = setCurImage
        self.__im_cur_tex = im_cur_tex

        """ 7.Create VAO configuration Macro """
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        with p_ref_vbo:   # auto bind
            glVertexAttribPointer(attr["p_ref"],   # index
                                  2,                # number of components per vertex attribute
                                  GL_FLOAT,         # type
                                  GL_FALSE,         # whether to normalize it to float or not
                                  0,                # stride, byte offset between consecutive vertex attributes
                                  p_ref_vbo)        # *pointer
            glEnableVertexAttribArray(attr["p_ref"])

        if attr["color"]!= -1:
            with color_vbo:
                glVertexAttribPointer(attr["color"],
                                      1,
                                      GL_UNSIGNED_BYTE,
                                      GL_TRUE,          # convert them to float when loaded in shader
                                      0,
                                      color_vbo)
                glEnableVertexAttribArray(attr["color"])
        glBindVertexArray(0)
        self.__vao = vao

        """ 8. The Draw Function"""
#        @timing
        def draw(H):
            if self.isCurSetted and self.isRefSetted:
                glUseProgram(shader)

                glUniformMatrix3fv(unif["H"], 1, GL_TRUE, H.astype('f')) # location,count,transpose,*value

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
                glBindVertexArray(vao)
                with p_ref_vbo,color_vbo: # auto vbo bind
                    glDrawArrays(GL_POINTS, 0, height*width)    # mode,first,count

                glBindVertexArray(0)
                glUseProgram(0)
        self.draw = draw

    def __delete__(self):
        glUseProgram(0)

        self.__p_ref_vbo.delete()
        self.__color_vbo.delete()
        glDeleteTextures(self.__im_cur_tex)
        glDeleteVertexArrays(self.__vao)
        glDeleteProgram(self.__shader)

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

        cGr = np.dot(np.linalg.inv(wGc[-1]), wGc[0])
        R,T = cGr[:3,:3], cGr[:3,3]
        self.M1 = K.dot(R).dot(inv(K))
        self.M2 = K.dot(T)
        self.d = 2.0

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0,  0))
        self.sweeper = PlaneSweeper(h,w)
        self.sweeper.setCurImage([frames[-1]])
        self.sweeper.setRefImage(frames[0])

        glViewport(0, 0, 640, 480)


    def resizeGL(self, width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        H = self.M1.copy()
        H[:, 2] += self.M2/self.d
        self.sweeper.draw(H)

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

        self.resize(w, h)
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