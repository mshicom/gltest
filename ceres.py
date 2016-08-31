#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:07:04 2016

@author: kaihong
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.weave import inline
from tools import *

def ba(x,y, d0, frames,cGr, K):
    scode = r"""
    #include "ceres/ceres.h"
    #include "ceres/cubic_interpolation.h"
    #include "ceres/rotation.h"
    #include <csignal>
    #include "glog/logging.h"
    #include <opencv2/opencv.hpp>

    #include <Eigen/Core>
    #include <string>
    #include <vector>
    #include <string>
    #include <memory>

    using ceres::Grid2D;
    using ceres::BiCubicInterpolator;
    using ceres::SizedCostFunction;
    using ceres::AutoDiffCostFunction;
    using ceres::CostFunction;
    using ceres::Problem;
    using ceres::Solver;
    using ceres::Solve;

    typedef BiCubicInterpolator<Grid2D<double,1> > ImageInterpolator;

    #define SCALE 16.0
    class ImageData
    {
    public:
        ImageData(double *data, size_t h, size_t w)
        {
            array_ = new Grid2D<double,1>(data, 0, h, 0, w);
            interpolator_ = new ImageInterpolator(*array_);
        }
        ImageInterpolator & Interpolator()
        { return *interpolator_; }

        ~ImageData()
        {
            delete[] array_;
            delete[] interpolator_;
        }
    public:
        Grid2D<double,1> *array_;
        ImageInterpolator *interpolator_;
    };
    typedef std::shared_ptr<ImageData> pImageData;

    template<int nbrs=9>
    class PhotometricCostFunction
      : public SizedCostFunction<nbrs /* number of residuals */,
                                 1 /* size of first parameter */>
    {
     public:
        double observed_I[nbrs];
        const ImageInterpolator & it_cur_;
        double Pe[3],Pinfs[nbrs][3];
        //const double weight[9] = {0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05};
        const double weight[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

      PhotometricCostFunction(const ImageInterpolator &it_ref, const ImageInterpolator &it_cur,
                              const double *K, const double *cGr,
                              double observed_x, double observed_y)
      : it_cur_(it_cur)
      {
          const double off_x[9] = {0, 0,+1,+2,+1, 0,-1,-2,-1};
          const double off_y[9] = {0,-2,-1, 0,+1,+2,+1, 0,-1};
          const double  &fx = K[0], &cx = K[2];
          const double  &fy = K[4], &cy = K[5];
          const double  ifx = 1.0/fx, ify = 1.0/fy;

          // Pe = K*Tcr
          const double &tx=cGr[3], &ty=cGr[7], &tz=cGr[11];
          Pe[0] = fx*tx+cx*tz;
          Pe[1] = fy*ty+cy*tz;
          Pe[2] =          tz;

          for(int i=0;i<nbrs;i++)
          {
              double *Pinf = Pinfs[i];
              double x = observed_x + off_x[i];
              double y = observed_y + off_y[i];
              it_ref.Evaluate(y, x, &observed_I[i]);

              // back-projection
              double P[3];
              P[0] = (x-cx)*ifx;
              P[1] = (y-cy)*ify;
              P[2] = 1;

              // rotation, R.dot(invK.dot(P))
              Pinf[0] = P[0]*cGr[0] + P[1]*cGr[1] + P[2]*cGr[2];
              Pinf[1] = P[0]*cGr[4] + P[1]*cGr[5] + P[2]*cGr[6];
              Pinf[2] = P[0]*cGr[8] + P[1]*cGr[9] + P[2]*cGr[10];
              // K
              Pinf[0] = Pinf[0]*fx + Pinf[2]*cx;
              Pinf[1] = Pinf[1]*fy + Pinf[2]*cy;
          }
      }
      virtual bool Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const {
        double inv_depth = parameters[0][0];
        for(int i=0;i<nbrs;i++) {
            const double *Pinf = Pinfs[i];
            double denominator = Pinf[2]+ inv_depth*Pe[2];
            double p_x = (Pinf[0] + inv_depth*Pe[0])/denominator;
            double p_y = (Pinf[1] + inv_depth*Pe[1])/denominator;

            double cur;
            if (jacobians != NULL && jacobians[0] != NULL) {
                double dfdx, dfdy;
                it_cur_.Evaluate(p_y, p_x, &cur, &dfdy, &dfdx);

                jacobians[0][i] =(  dfdx*(Pe[0]*Pinf[2]-Pe[2]*Pinf[0])
                                  + dfdy*(Pe[1]*Pinf[2]-Pe[2]*Pinf[1])
                                  ) / (denominator*denominator) * weight[i];
            }
            else
                it_cur_.Evaluate(p_y, p_x, &cur);

            residuals[i] = weight[i]*(cur - observed_I[i]);
        }

        return true;
      }
    };
    """
    code = r"""
        if (!PyList_Check(py_frames) && !PyList_Check(py_cGr))
            py::fail(PyExc_TypeError, "frames and rGc must be a list");

        assert(frames.len() == cGr.len());

        size_t max = frames.len();
        std::vector<pImageData> image_set;
        std::vector<double*> G_set;

        //std::raise(SIGINT);

        for(size_t i=0; i<max; i++)
        {
            std::cout<<"creating image"<< i<<std::endl;

            PyObject *im= PyList_GET_ITEM(py_frames,i);
            int nd = PyArray_NDIM(im);
            npy_intp *dims = PyArray_DIMS(im);
            double *dptr = (double *)PyArray_DATA(im); // pointer to data.
            image_set.push_back(std::make_shared<ImageData>(dptr, dims[0], dims[1]));

            PyObject *G = PyList_GET_ITEM(py_cGr,i);
            double *g = (double *)PyArray_DATA(G);
            G_set.push_back(g);
        }
        Problem problem;
         double d = d0;
         for(size_t i=1; i<max; i++)
         {
              CostFunction* cost_function = new PhotometricCostFunction<9>(
                          image_set[0]->Interpolator() , image_set[i]->Interpolator(),
                          K, G_set[i], x, y);
              problem.AddResidualBlock(cost_function, /*NULL*/new ceres::HuberLoss(100), &d);
              problem.SetParameterLowerBound(&d, 0, 0.02);
              problem.SetParameterUpperBound(&d, 0, 1e2);
         }

         Solver::Options options;
          options.minimizer_progress_to_stdout = false;
        //  options.linear_solver_type = ceres::DENSE_QR;
          options.use_nonmonotonic_steps = 1;
          Solver::Summary summary;
          Solve(options, &problem, &summary);
          return_val = d;
    """

    frames = [np.ascontiguousarray(f, 'd') for f in frames]
    cGr = [np.ascontiguousarray(g, 'd') for g in cGr]
    K = np.ascontiguousarray(K, 'd')

    d = inline(code,['frames','cGr','K', 'd0','x','y'],
           support_code=scode,
           include_dirs = ["/usr/include/eigen3/"],
           libraries = ['ceres','glog','opencv_core','opencv_highgui','opencv_imgproc','opencv_imgproc'],
           extra_compile_args=['-std=gnu++11 -msse2 -O3'],
           extra_link_args = [r'-rdynamic -lspqr -ltbb -ltbbmalloc -lcholmod -lccolamd -lcamd -lcolamd -lamd -llapack'])
    return d