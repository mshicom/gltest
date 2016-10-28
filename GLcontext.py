#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:18:28 2016

@author: kaihong
"""
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
    xdisplay = xlib.XOpenDisplay("")
    display = Xlib.display.Display()

    def __init__(self, width, height):
        """ lets setup are opengl settings and create the context for our window """
        def cAttrs(aList):
            aList += [0]
            return (c_int * len(aList))(*aList)

        xvinfo = GLX.glXChooseVisual(self.xdisplay,
                                     self.display.get_default_screen(),
                                     cAttrs([GLX.GLX_RGBA,          1,
#                                             GLX.GLX_RED_SIZE,      1,
#                                             GLX.GLX_GREEN_SIZE,    1,
#                                             GLX.GLX_BLUE_SIZE,     1,
                                             GLX.GLX_DOUBLEBUFFER,  0]))
        self.context = GLX.glXCreateContext(self.xdisplay, xvinfo, None, True)

        configs = GLX.glXChooseFBConfig(self.xdisplay, 0, None, byref(c_int()))
        self.pbuffer = GLX.glXCreatePbuffer(self.xdisplay, configs[0],
                                            cAttrs([GLX.GLX_PBUFFER_HEIGHT, height,
                                                    GLX.GLX_PBUFFER_WIDTH,  width]))

        if(not GLX.glXMakeContextCurrent(self.xdisplay, self.pbuffer, self.pbuffer, self.context)):
            raise RuntimeError("Failed to make GL context current!")
        glViewport(0, 0, width, height)
        print "GL context created!"

    def __call__(self):
        if(not GLX.glXMakeContextCurrent(self.xdisplay, self.pbuffer, self.pbuffer, self.context)):
            raise RuntimeError("Failed to make GL context current!")

    def __del__(self):
        GLX.glXMakeContextCurrent(self.xdisplay, 0, 0, None)
        GLX.glXDestroyContext(self.xdisplay, self.context)
        self.xlib.XCloseDisplay(self.xdisplay)
        print "GL context destroyed!"




#%%
from tools import *
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays.vbo import VBO
import numpy as np
class PlaneSweeper():
    GLContext = None
    def __init__(self, height, width, max_images = 10):
        self.GLContext = OffScreenGL(width, height)

        """ 1.Calculate NDC/Texture mapping matrix"""
        xmin,xmax,ymin,ymax = 0.0, width-1.0, 0.0, height-1.0

        mapNDC = np.array([[2/(xmin-xmax),   0, 1-2*xmin/(xmin-xmax)],
                           [0,   2/(ymin-ymax), 1-2*ymin/(ymin-ymax)]])

        mapTEX = np.array([[1/(xmax-xmin),   0, -xmin/(xmax-xmin)],
                           [0,   1/(ymax-ymin), -ymin/(ymax-ymin)]])

        """ 2.Create and compile shader code"""
        vsrc = """#version 130
            in vec2 p_ref;        // [x, y]
            in float pcolor;

            uniform mat3 R[%(max_images)d];
            uniform vec3 t[%(max_images)d];
            uniform int im_cnt;
            uniform sampler2DArray im_cur;
            uniform float idepth;
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
                float err_sum = 0;
                int valid_cnt = 0;
                int i;
                for(i=0; i<im_cnt; ++i)
                {
                    mat3 H = R[i];
                    H[2] += t[i]*idepth;

                    vec3 p_cur = H*vec3(p_ref, 1);
                    vec2 tc = p_cur.xy / p_cur.z;
                    bool isValid = isInImage(tc);
                    if(isValid) {
                        ++valid_cnt;
                        err_sum += abs( pcolor-texture(im_cur, vec3(toTEX(tc), i)).r );
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

        """ 3.Extract parameter locations"""
        attr = {"p_ref" : glGetAttribLocation(shader, "p_ref"),
                "color" : glGetAttribLocation(shader, "pcolor")}
        unif = {"R"     : glGetUniformLocation(shader, "R"),
                "t"     : glGetUniformLocation(shader, "t"),
                "im_cnt": glGetUniformLocation(shader, "im_cnt"),
                "idepth": glGetUniformLocation(shader, "idepth")}

        def setCurImagePos(cGr, K):
            if isinstance(cGr,np.ndarray):
                cGr = [cGr]
            N = len(cGr)
            if N>max_images:
                raise RuntimeWarning("Too many matrix")
                cGr = cGr[:max_images]

            invK = inv(K)
            Rs = [ K.dot(G[:3,:3]).dot(invK) for G in cGr]
            ts = [ K.dot(G[:3,3])            for G in cGr]
            #Ms = np.vstack([ R+vec(t).dot(np.array([[0,0,idepth]]))  for R,t in zip(Rs,ts)])
            self.GLContext()
            glUseProgram(shader)
            glUniformMatrix3fv(unif["R"], N, GL_TRUE, np.vstack(Rs).astype('f')) # location,count,transpose,*value
            glUniform3fv(unif["t"], N, np.vstack(ts).astype('f'))
            glUniform1i(unif["im_cnt"], N)
            glUseProgram(0)
        self.setCurImagePos = setCurImagePos

        def setTargetDepth(idepth):
            self.GLContext()
            glProgramUniform1f(shader, unif["idepth"], float(idepth))
        self.setTargetDepth = setTargetDepth

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
            self.GLContext()
            with color_vbo:
                color_vbo.set_array(image)
                color_vbo.copy_data()           # send data to gpu
                self.isRefSetted = True
        self.setRefImage = setRefImage
        self.__color_vbo = color_vbo

        """ 6.Create texture array for cur image"""
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
        @timing
        def setCurImage(images):
            if isinstance(images,np.ndarray):
                images = [images]
            if len(images) > max_images:
                raise RuntimeWarning("More than 10 images")
                images = images[:max_images]
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
        @timing
        def setCurImagePBO(images):
            if isinstance(images,np.ndarray):
                images = [images]
            if len(images) > max_images:
                raise RuntimeWarning("More than 10 images")
                images = images[:max_images]

            imsize = width*height
            imcnt = len(images)

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
            self.isCurSetted = True
        self.setCurImagePBO = setCurImagePBO
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
        @timing
        def draw():
            if self.isCurSetted and self.isRefSetted:
                glUseProgram(shader)

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D_ARRAY, im_cur_tex)
                glBindVertexArray(vao)
                with p_ref_vbo,color_vbo: # auto vbo bind
                    glDrawArrays(GL_POINTS, 0, height*width)    # mode,first,count

                glBindVertexArray(0)
                glUseProgram(0)
        self.draw = draw

        @timing
        def getResult(res = np.empty((height,width),'f')):
            glReadPixels(0,  0,              # window coordinates of the first pixel
                         width,height,       # dimensions of the pixel rectangle
                         GL_RED,             # format
                         GL_FLOAT,           #	 data type,
                         res)
            return res
        self.getResult = getResult

        def process(ref_im, cur_ims, cGr, K, idepths):
            self.GLContext()
            self.setRefImage(ref_im)
            self.setCurImagePBO(cur_ims)
            self.setCurImagePos(cGr, K)

            idepth = np.atleast_1d(idepth)
            res = np.empty((len(idepth), height, width),'f')
            for i,idp in enumerate(idepths):
                self.setTargetDepth(idp)
                self.draw()
                self.getResult(res[i]);
            return res
        self.process = process

    def __delete__(self):
        glUseProgram(0)

        self.__p_ref_vbo.delete()
        self.__color_vbo.delete()
        glDeleteTextures(self.__im_cur_tex)
        glDeleteVertexArrays(self.__vao)
        glDeleteProgram(self.__shader)
#%%
frames, wGc, K, Z = loaddata1()
#from orb_kfs import  loaddata4
#frames, wGc, K, Z = loaddata4(60)
h,w = frames[0].shape[:2]
fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
#%%

cGr = [inv(G).dot(wGc[0]) for G in wGc]
sweep = PlaneSweeper(h,w)
res = sweep.process(frames[0],frames[1:],cGr[1:], K, 1/2.0)

