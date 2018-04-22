#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:36:17 2016

@author: kaihong
"""
import sys
from PyQt5 import QtCore,QtGui,QtOpenGL,QtWidgets
from OpenGL import GLU
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays.vbo import VBO
#from OpenGLContext.quaternion import  fromEuler
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tools import *


#%%
from OpenGL.GL import *
from OpenGL import GLX
from OpenGL.raw.GLX._types import struct__XDisplay
from ctypes import *
import Xlib
import Xlib.display
class OffScreenGL:
    """ these method do not seem to exist in python x11 library lets exploit the c methods """
    xlib = cdll.LoadLibrary('libX11.so')
    xlib.XOpenDisplay.argtypes = [c_char_p]
    xlib.XOpenDisplay.restype = POINTER(struct__XDisplay)
    xdisplay = xlib.XOpenDisplay(b'')
    display = Xlib.display.Display()

    def __init__(self, width, height):
        assert(self.xdisplay)
        """ lets setup are opengl settings and create the context for our window """
        def cAttrs(aList):
            aList += [0,0]
            return (c_int * len(aList))(*aList)

        xvinfo = GLX.glXChooseVisual(self.xdisplay,
                                     self.display.get_default_screen(),
                                     cAttrs([GLX.GLX_RGBA,
                                             GLX.GLX_DOUBLEBUFFER,
                                             GLX.GLX_DEPTH_SIZE,    24,
#                                             GLX.GLX_RED_SIZE,      1,
#                                             GLX.GLX_GREEN_SIZE,    1,
#                                             GLX.GLX_BLUE_SIZE,     1,

                                             ]))
        self.context = GLX.glXCreateContext(self.xdisplay, xvinfo, None, True)

        configs = GLX.glXChooseFBConfig(self.xdisplay, 0, None, byref(c_int()))
        self.pbuffer = GLX.glXCreatePbuffer(self.xdisplay, configs[0],
                                            cAttrs([GLX.GLX_PBUFFER_HEIGHT, height,
                                                    GLX.GLX_PBUFFER_WIDTH,  width]))

        if(not GLX.glXMakeContextCurrent(self.xdisplay, self.pbuffer, self.pbuffer, self.context)):
            raise RuntimeError("Failed to make GL context current!")
        glViewport(0, 0, width, height)
        print("GL context created!")

    def __call__(self):
        if(not GLX.glXMakeContextCurrent(self.xdisplay, self.pbuffer, self.pbuffer, self.context)):
            raise RuntimeError("Failed to make GL context current!")

    def __del__(self):
        GLX.glXMakeContextCurrent(self.xdisplay, 0, 0, None)
        GLX.glXDestroyContext(self.xdisplay, self.context)
        print("GL context destroyed!")




#%%
from tools import *
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays.vbo import VBO
import numpy as np
class PlaneSweeper():
    GLContext = None
    def __init__(self, height, width, max_images = 10, offscreen = True):
        if offscreen:
            self.GLContext = OffScreenGL(width, height)

        """ 1.Calculate NDC/Texture mapping matrix"""
        if offscreen:
            # flip image up-side down
            mapNDC = np.array([[2.0/width,   0, -1],
                               [0,   2.0/height, -1]])
        else:
            mapNDC = np.array([[2.0/width,   0, -1],
                               [0,  -2.0/height, 1]])

        mapTEX = np.array([[1.0/width,   0, 0],
                           [0,   1.0/height, 0]])

        """ 2.Create and compile shader code"""
        vsrc = """#version 130
            in vec2 p_ref;        // [x, y]

            uniform mat3 R[%(max_images)d];
            uniform vec3 t[%(max_images)d];
            uniform int im_cnt;
            uniform sampler2D im_ref;
            uniform sampler2DArray im_cur;
            uniform float idepth;
            out lowp float c;

            const float width = %(width)d-1, height = %(height)d-1;
            bool isInImage(vec2 p)
            {    return p.x>0 && p.x<width && p.y>0 && p.y<height;  }   // 1 pixel border gap

            const mat3x2 mapNDC = mat3x2( %(mapNDC)s );
            vec2 toNDC(vec2 p)
            {   return mapNDC*vec3(p,1); }    // map from [0:h,0:w] to [-1:1,-1:1]

            const mat3x2 mapTEX = mat3x2( %(mapTEX)s );
            vec2 toTEX(vec2 p)
            {   return mapTEX*vec3(p,1); }    // map from [0:h,0:w] to [0:1,0:1]

            vec2 dxy_local(vec3 Pe0, vec2 Pr) // Pe0 = K*Trc
            {
                vec2 res = -Pe0.z*Pr.xy+Pe0.xy;
                return res/length(res);
            }
            vec2 dxy(mat3 KRcrK_, vec3 KTcr, vec2 Pr, float dinv)
            {
                vec3 Pinf = KRcrK_*vec3(Pr,1);
                vec3 Pe = KTcr;
                vec2 dxy_raw = (-Pe.z/Pinf.z)*Pinf.xy+Pe.xy;
                float dxy_norm = length(dxy_raw);
                vec2 dxy = dxy_raw/dxy_norm;

                float denom = 1/(Pinf.z+dinv*Pe.z);
                float v = dinv*denom*dxy_norm;
                float x = (Pinf.x+dinv*Pe.x)*denom;
                float y = (Pinf.y+dinv*Pe.y)*denom;

                return vec2(x,y);
            }

            void main(void)
            {
                float pcolor = texture2D(im_ref, toTEX(p_ref)).r;

                lowp float err_sum = 0;
                int valid_cnt = 0;
                int i;
                for(i=0; i<im_cnt; ++i)
                {
                    highp mat3 H = R[i];
                    H[2] += t[i]*idepth;

                    highp vec3 p_cur = H*vec3(p_ref, 1);
                    highp vec2 tc = p_cur.xy / p_cur.z;
                    bool isValid = isInImage(tc);
                    lowp float value = texture(im_cur, vec3(toTEX(tc), i)).r;

                    if(isValid) {
                        ++valid_cnt;
                        err_sum += abs( pcolor-value );
                    }
                 }

                 if(valid_cnt>0)
                    c = err_sum/valid_cnt;
                 else
                    c = pcolor;

                gl_Position.xy = toNDC(p_ref);
                gl_Position.zw = vec2(0,1);
            }""" % {"mapNDC": "".join(str(v)+"," for v in mapNDC.ravel('F'))[:-1],
                    "mapTEX": "".join(str(v)+"," for v in mapTEX.ravel('F'))[:-1],
                    "width": width,
                    "height": height,
                    "max_images":max_images}

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
        glUseProgram(shader)

        """ 3.Extract parameter locations"""
        attr = {"p_ref" : glGetAttribLocation(shader, "p_ref"),
                "color" : glGetAttribLocation(shader, "pcolor")}
        unif = {"R"     : glGetUniformLocation(shader, "R"),
                "t"     : glGetUniformLocation(shader, "t"),
                "im_cnt": glGetUniformLocation(shader, "im_cnt"),
                "idepth": glGetUniformLocation(shader, "idepth")}
        # set sampler in pos 0
        glUniform1i(glGetUniformLocation(shader, "im_cur"), 0)
        glUniform1i(glGetUniformLocation(shader, "im_ref"), 1)

        """ 4.Create constant vbo for p_ref, used in step 7"""
        y, x = np.mgrid[0:height, 0:width]
        P = np.vstack([x.ravel(), y.ravel()]).T
        P = np.ascontiguousarray(P, 'f')
        p_ref_vbo = VBO(P, GL_STATIC_DRAW, GL_ARRAY_BUFFER)
        self.__p_ref_vbo = p_ref_vbo

        """ 5.Create texture for reference pixel data """
        im_ref_tex = glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, im_ref_tex)
        glTexImage2D(GL_TEXTURE_2D,     # target
                         0,                 # mid map level
                         GL_RED,            # number of color components in the texture
                         width, height,     # width, height
                         0,                 # border(must be 0)
                         GL_RED,            # format of the pixel data, i.e. GL_RED, GL_RG, GL_RGB,
                         GL_UNSIGNED_BYTE,  # data type of the pixel data, i.e GL_UNSIGNED_BYTE, GL_FLOAT ...
                         None)              # data pointer
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.isRefSetted = False
        def setRefImage(image, K, wGr):
            if image.dtype != np.uint8:
                raise RuntimeError("image must be uint8")
            if image.shape != (height,width):
                raise RuntimeError("image size not matched")
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, im_ref_tex)
            glTexImage2D(GL_TEXTURE_2D,     # target
                         0,                 # mid map level
                         GL_RED,            # number of color components in the texture
                         width, height,     # width, height
                         0,                 # border(must be 0)
                         GL_RED,            # format of the pixel data, i.e. GL_RED, GL_RG, GL_RGB,
                         GL_UNSIGNED_BYTE,  # data type of the pixel data, i.e GL_UNSIGNED_BYTE, GL_FLOAT ...
                         image)             # data pointer
            glBindTexture(GL_TEXTURE_2D, 0)
            self.isRefSetted = True
            self.K, self.wGr = K, wGr
        self.setRefImage = setRefImage
        self.__im_ref_tex = im_ref_tex

        """ 6.Create texture array for cur image"""
        im_cur_tex = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
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
        @timing
        def setCurImage(images):
            if isinstance(images,np.ndarray):
                images = [images]
            if len(images) > max_images:
                raise RuntimeWarning("More than 10 images")
                images = images[:max_images]
            glUseProgram(shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
            for layer,image in enumerate(images):
                if image.dtype != np.uint8:
                    raise RuntimeError("image must be uint8")
                if image.shape != (height,width):
                    raise RuntimeError("image size not matched")
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY,    # target,
                             0,0,0,layer,                   # mid map level, x/y/z-offset
                             width, height, 1,               # width, height,layers
                             GL_RED,            # format of the pixel data, i.e. GL_RED, GL_RG, GL_RGB,
                             GL_UNSIGNED_BYTE,  # data type of the pixel data, i.e GL_UNSIGNED_BYTE, GL_FLOAT ...
                             image)             # data pointer
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
            glUseProgram(0)

            self.isCurSetted = True
        self.setCurImage = setCurImage
        @timing
        def setCurImagePBO(images):
            if isinstance(images,np.ndarray):
                images = [images]
            if len(images) > max_images:
                raise RuntimeWarning("More than 10 images")
                images = images[:max_images]

            imsize = width*height
            imcnt = len(images)
            glUseProgram(shader)
            glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
            # create pbo
            pbo = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, imsize*imcnt, None, GL_STREAM_READ) # allocate space
            # copy data to it (cpu side)
            for layer,image in enumerate(images):
                if image.dtype != np.uint8:
                    raise RuntimeError("image must be uint8")
                if image.shape != (height,width):
                    raise RuntimeError("image size not matched")
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                                layer*imsize,     # offset
                                imsize,           # size
                                image)
            # now issue the transfer order (from CPU to GPU by DMA in the backgroung)
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY,    # target,
                             0,0,0,0,           # mid map level, x/y/z-offset
                             w, h, imcnt,               # width, height,layers
                             GL_RED,            # format of the pixel data, i.e. GL_RED, GL_RG, GL_RGB,
                             GL_UNSIGNED_BYTE,  # data type of the pixel data, i.e GL_UNSIGNED_BYTE, GL_FLOAT ...
                             None)             # data pointer
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            glDeleteBuffers(1, [pbo])
            glUseProgram(0)
            self.isCurSetted = True
        self.setCurImagePBO = setCurImagePBO
        self.__im_cur_tex = im_cur_tex

        def setCurImagePos(wGc):
            if isinstance(wGc,np.ndarray):
                wGc = [wGc]
            N = len(wGc)
            if N>max_images:
                raise RuntimeWarning("Too many matrix")
                wGc = wGc[:max_images]
            cGr = [inv(G).dot(self.wGr) for G in wGc]
            invK = inv(self.K)
            Rs = [ self.K.dot(G[:3,:3]).dot(invK) for G in cGr]
            ts = [ self.K.dot(G[:3,3])            for G in cGr]
            #Ms = np.vstack([ R+vec(t).dot(np.array([[0,0,idepth]]))  for R,t in zip(Rs,ts)])
#            self.GLContext()
            glUseProgram(shader)
            glUniformMatrix3fv(unif["R"], N, GL_TRUE, np.vstack(Rs).astype('f')) # location,count,transpose,*value
            glUniform3fv(unif["t"], N, np.vstack(ts).astype('f'))
            glUniform1i(unif["im_cnt"], N)
            glUseProgram(0)
        self.setCurImagePos = setCurImagePos
        self.setTargetDepth = lambda idepth: glProgramUniform1f(shader, unif["idepth"], float(idepth))

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
        glBindVertexArray(0)
        self.__vao = vao

        """ 8. The Draw Function"""
        @timing
        def draw():
            if self.isCurSetted and self.isRefSetted:
                glUseProgram(shader)

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, im_ref_tex)

                glBindVertexArray(vao)
                with p_ref_vbo: # auto vbo bind
                    glDrawArrays(GL_POINTS, 0, height*width)    # mode,first,count

                glBindVertexArray(0)
                glUseProgram(0)
        self.draw = draw

        """ 9. Readout the result/pixel"""
        @timing
        def getResult(res = np.empty((height,width),'f')):
            glReadPixels(0,  0,              # window coordinates of the first pixel
                         width,height,       # dimensions of the pixel rectangle
                         GL_RED,             # format
                         GL_FLOAT,           #	 data type,
                         res)
            return res
        self.getResult = getResult

        """ 10.All-in-one API Function"""
        @timing
        def process(ref_im, wGr, cur_ims, wGc, K, idepths):
            if not self.GLContext is None:
                self.GLContext()
            glUseProgram(shader)
            self.setRefImage(ref_im, K, wGr)
            self.setCurImage(cur_ims)
            self.setCurImagePos(wGc)

            idepths = np.atleast_1d(idepths)
            res = np.empty((len(idepths), height, width),'f')
            for i,idp in enumerate(idepths):
                self.setTargetDepth(idp)
                self.draw()
                self.getResult(res[i])
            glUseProgram(0)
            return res
        self.process = process
        glUseProgram(0)

    def __delete__(self):
        glUseProgram(0)
        glDeleteTextures(self.__im_cur_tex)
        glDeleteTextures(self.__im_ref_tex)
        glDeleteVertexArrays(self.__vao)
        glDeleteProgram(self.__shader)
        if not self.GLContext is None:
            del self.GLContext

if __name__ == "__main__":
    #%%
    frames, wGc, K, Z = loaddata1()
#    from orb_kfs import  loaddata4
#    frames, wGc, K, Z = loaddata4(10)
    h,w = frames[0].shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    def testOffscreen():
        sweep = PlaneSweeper(h,w)
        res = sweep.process(frames[0],wGc[0],frames[1:], wGc[1:], K, np.linspace(iD(0.1),iD(6),50))
        return res


    class GLWidget(QtOpenGL.QGLWidget):
        def __init__(self, parent=None):
            self.parent = parent
            QtOpenGL.QGLWidget.__init__(self, parent)

            self.d = 2.0

        def initializeGL(self):
            self.qglClearColor(QtGui.QColor(0, 0,  0))
            self.sweeper = PlaneSweeper(h,w, offscreen=False)
            self.sweeper.setRefImage(frames[0], K, wGc[0])
            self.sweeper.setCurImagePBO(frames[1:])
            self.sweeper.setCurImagePos(wGc[1:])

            glViewport(0, 0, 640, 480)

        def resizeGL(self, width, height):
            if height == 0: height = 1
            glViewport(0, 0, width, height)

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.sweeper.setTargetDepth(1/self.d)
            self.sweeper.draw()

        def wheelEvent(self, e):
            # QtGui.QWheelEvent(e)
            """ zoom in """
            sign = np.sign(e.angleDelta().y())
            if self.d < 1:
                inc = 0.02*sign
            else:
                inc = 0.2*sign

            self.d = np.clip(self.d+inc, 0.01, 8.0)
            print( self.d)
            self.updateGL()

    class MainWindow(QtWidgets.QMainWindow):

        def __init__(self):
            QtWidgets.QMainWindow.__init__(self)

            self.resize(w, h)
            self.setWindowTitle('GL Cube Test')

            self.initActions()

            self.glWidget = GLWidget(self)

            self.setCentralWidget(self.glWidget)

        def initActions(self):
            self.exitAction = QtWidgets.QAction('Quit', self)
            self.exitAction.setShortcut('Ctrl+Q')
            self.exitAction.setStatusTip('Exit application')
            self.exitAction.triggered.connect(self.close)
            
        def close(self):
            QtWidgets.qApp.quit()


    if 0:
        res = testOffscreen()
        IndexTracker(res)
    else:
        app_created = False
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            app_created = True
        app.references = set()
        window = MainWindow()
        app.references.add(window)
        window.show()
        if app_created:
            app.exec_()
#%%

#    sweep = PlaneSweeper(h,w)
#    res = sweep.process(frames[0],wGc[0],frames[1:], wGc[1:], K, np.linspace(iD(0.1),iD(6),50))
#
#    fig,(a1,a2) = plt.subplots(2,1,num='slice')
#    a1.imshow(frames[0])
#    l, = a2.plot([])
#    while 1:
#        p = np.array(plt.ginput(1,-1)[0],np.int)
#        a1.plot(p[0],p[1],'r.')
#        a2.plot(range(50),res[:,p[0],p[1]])
#        plt.pause(0.001)