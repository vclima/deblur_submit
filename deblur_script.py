
import queue
from PIL import Image
from os import listdir,makedirs
from os.path import isfile,join,exists

import sys
import cv2 as cv
import numpy as np
import time
import multiprocessing as mp
from numpy.fft import fft2
from numpy.fft import ifft2
from scipy.fftpack import fftn, ifftn
from scipy import ndimage
from skimage.restoration import (denoise_tv_chambolle,denoise_wavelet,denoise_bilateral)
from skimage import exposure

from PIL import Image
from skimage.color import rgb2yuv, yuv2rgb

in_path = sys.argv[1]
out_path = sys.argv[2]
cat_nbr = sys.argv[3]

def hist_norm(img):
    p2, p98 = np.percentile(img, (4, 96))
    img_eq = exposure.rescale_intensity(img, in_range=(p2, p98))
    img_eq[img_eq<0.3] = 0
    img_eq[img_eq>0.3] = 1
    img_eq= 1-img_eq
    img_eq=(img_eq*255).astype(np.uint8)
    return Image.fromarray(img_eq,mode="L")

def deblur(input_path=in_path, output_path=out_path, categoryNbr=cat_nbr):
  exclude=["LSF","PSF"]
  ip1=join(input_path,"CAM01")
  ip2=join(input_path,"CAM02")

  
  img_files=[f for f in listdir(ip1) if (isfile(join(ip1,f)) and f.endswith(".tif"))]
  for img_path in img_files:
    if "PSF" in img_path:
      psf1=Image.open(join(ip1,img_path))
      psf2=Image.open(join(ip2,img_path))

  for img_path in img_files:
    if not any(ignore for ignore in exclude if ignore in img_path):
      im1=Image.open(join(ip1,img_path))
      im2=Image.open(join(ip2,img_path))

      scale=1
      start=time.time()
      Rcomb=deblur_fp_06(im1,im2,psf1,psf2,categoryNbr,scale)
      end=time.time()
      print("Elapsed time: "+time.strftime("%H:%M:%S",time.gmtime(end-start)))

      '''
      Rw=(Rw*255).astype(np.uint8)
      Rsparse=(Rsparse*255).astype(np.uint8)
      RL1Sparse=(RL1Sparse*255).astype(np.uint8)
      Ranti=(Ranti*255).astype(np.uint8)
      Rcomb=(Rcomb*255).astype(np.uint8)

      Rw_image=Image.fromarray(Rw,mode="L")
      Rsparse_image=Image.fromarray(Rsparse,mode="L")
      RL1Sparse_image=Image.fromarray(RL1Sparse,mode="L")
      Ranti_image=Image.fromarray(Ranti,mode="L")
      Rcomb_image=Image.fromarray(Rcomb,mode="L")
      '''

      if not exists(output_path):
        makedirs(output_path)

      img_path=img_path.replace('tif','png')
      Rcomb.save(join(output_path,img_path))


  return

def deblur_fp_06(cam01,cam02,psf1,psf2,categoryNbr,scale=1):#Arrumar os parametros daqui
  #im = Image.open('/content/sample_data/focusStep_5_timesR_size_30_sample_0001_cam01.tif')
  im = cam01
  t = np.asarray(im)
  print(t.shape)
  t = (t-np.min(t))/(np.max(t)-np.min(t))
  t = 1-t
  t_aux=t

  #im2  = Image.open('/content/sample_data/focusStep_5_timesR_size_30_sample_0001_cam02.tif')
  im2 = cam02
  X = np.asarray(im2)
  X = (X-np.min(X))/(np.max(X)-np.min(X))
  X = 1-X

  #im3 = Image.open('/content/sample_data/focusStep_5_PSF_cam01.tif')
  im3 = psf1
  tpsf = np.asarray(im3)
  tpsf = (tpsf-np.min(tpsf))/(np.max(tpsf)-np.min(tpsf))
  tpsf = 1-tpsf

  #im4 = Image.open('/content/sample_data/focusStep_5_PSF_cam02.tif')
  im4 = psf2
  Xpsf = np.asarray(im4)
  Xpsf = (Xpsf-np.min(Xpsf))/(np.max(Xpsf)-np.min(Xpsf))
  Xpsf = 1-Xpsf

  width = int(t.shape[1] * scale)
  height = int(t.shape[0] * scale)
  dim = (width, height)

  scaled_t = cv.resize(t, dim, interpolation = cv.INTER_AREA)
  scaled_X = cv.resize(X, dim, interpolation = cv.INTER_AREA)
  scaled_tpsf = cv.resize(tpsf, dim, interpolation = cv.INTER_AREA)
  scaled_Xpsf = cv.resize(Xpsf, dim, interpolation = cv.INTER_AREA)
  '''
  t    = t.copy()
  X    = X.copy()
  R0   = X.copy()
  tpsf = tpsf.copy()
  Xpsf = Xpsf.copy()
  '''
  t    = scaled_t.copy()
  X    = scaled_X.copy()
  R0   = scaled_X.copy()
  tpsf = scaled_tpsf.copy()
  Xpsf = scaled_Xpsf.copy()

  M,N=tpsf.shape
  sig={'0':0.08,'1':0.0825,'2':0.085,'3':0.0875,'4':0.09,'5':0.1,'6':0.11,'7':0.12,'8':0.13,
     '9':0.14,'10':0.15,'11':0.17,'12':0.17,'13':0.17,'14':0.17,'15':0.17,'16':0.17,
     '17':0.17,'18':0.17,'19':0.17}
  k={'0':(int(np.round(M*0.02)),int(np.round(N*0.006))),
     '1':(int(np.round(M*0.02)),int(np.round(N*0.005))),
     '2':(int(np.round(M*0.015)),int(np.round(N*0.006))),
     '3':(int(np.round(M*0.015)),int(np.round(N*0.006))),
     '4':(int(np.round(M*0.013)),int(np.round(N*0.005))),
     '5':(int(np.round(M*0.013)),int(np.round(N*0.006))),
     '6':(int(np.round(M*0.0125)),int(np.round(N*0.005))),
     '7':(int(np.round(M*0.0105)),int(np.round(N*0.006))),
     '8':(int(np.round(M*0.009)),int(np.round(N*0.006))),
     '9':(int(np.round(M*0.007)),int(np.round(N*0.006))),
     '10':(int(np.round(M*0.005)),int(np.round(-N*0.002))),
     '11':(int(np.round(M*0.0)),int(np.round(-N*0.001))),
     '12':(int(np.round(M*0.0)),int(np.round(-N*0.002))),
     '13':(int(np.round(M*0.0)),int(np.round(-N*0.0025))), 
     '14':(int(np.round(-M*0.005)),int(np.round(-N*0.003))),
     '15':(int(np.round(-M*0.005)),int(np.round(-N*0.003))),
     '16':(int(np.round(-M*0.015)),int(np.round(-N*0.001))),
     '17':(int(np.round(-M*0.015)),int(np.round(-N*0.0055))),
     '18':(int(np.round(-M*0.02)),int(np.round(-N*0.004))),
     '19':(int(np.round(-M*0.01)),int(np.round(-N*0.004)))}

  x = np.linspace(-1, 1, tpsf.shape[1])
  y = np.linspace(-1, 1, tpsf.shape[0])*tpsf.shape[0]/tpsf.shape[1]
  x, y = np.meshgrid(x, y) 
  wind=ptrans(gauss(tpsf.shape,sig[str(categoryNbr)]),k[str(categoryNbr)])

  tpsf = tpsf*wind
  Xpsf = Xpsf*wind

  fftR = fft2(tpsf)
  fftX = fft2(Xpsf)
  OTF0 = ((np.conjugate(fftR)*fftX)/(fftR*np.conjugate(fftR)+5))
  #OTF0=np.load('C:\Deblur_Challenge\OTF\OTF7.npy')

  #Niter     = 100
  #lamb      = 0.01
  #lambdaPSF = 5
  
  Niter     = 50
  lamb      = 0.005
  lambdaPSF = 5

  #Wavelet denoiser
  f1=denoise_wavelet
  #Rw,OTFEw=blind_decon2D_red_fp_wavelet(R0,X,OTF0,Niter,lamb,lambdaPSF,f,t)

  #Wavelet denoiser + sparse denoise
  #f=denoise_wavelet
  #Rsparse,OTFEsparse=blind_decon2D_red_fp_wavelet_sparse(R0,X,OTF0,Niter,lamb,lambdaPSF,f,t)

  # Sparse denoise (L1 soft threshold)
  f3         = denoise_L1
  radius3    = 0.15
  #RL1sparse,OTFEsparse = blind_decon2D_red_fp_L1(R0,X,OTF0,Niter,lamb,lambdaPSF,f,radius,t)
 
  # Anti sparse denoise (L1 projection)
  f4      = denoise_L1projection
  alpha  = 1
  radius4 = alpha*np.max(R0)
  #Ranti, OTFEanti = blind_decon2D_red_fp_L1(R0,X,OTF0,Niter,lamb,lambdaPSF,f,radius,t)

  qout=mp.Queue()
  jobs=(blind_decon2D_red_fp_L1,blind_decon2D_red_fp_L1)
  args=((R0,X,OTF0,Niter,lamb,lambdaPSF,f3,radius3,t,qout,1),(R0,X,OTF0,Niter,lamb,lambdaPSF,f4,radius4,t,qout,2))

  processes = [mp.Process(target=job, args=arg)
             for job, arg in zip(jobs, args)]
  
 
  for p in processes:
    p.start()

  unsorted=[]
  while True:
    try:
        op = qout.get(False)
        unsorted.append(op)
    except queue.Empty:
        pass
    allExited = True
    for t in processes:
        if t.exitcode is None:
            allExited = False
            break
    if allExited & qout.empty():
        break

  for p in processes:
    p.join()

  result=[t[1] for t in sorted(unsorted)]

  RL1sparse=result[0]
  Ranti=result[1]

  sigma = 0.5
  Rcomb = sigma*RL1sparse + (1-sigma)*Ranti

  Rcomb = hist_norm(Rcomb)
  
  return Rcomb

def ptrans(f,to=None):
  if to is None:
      to=f.shape[0]//2+1,f.shape[1]//2+1
  
  rr,cc = to
  H,W = f.shape

  r=H-rr%H
  c=W-cc%W

  h=np.zeros([2*H,2*W])
  h[0:H,0:W]=f
  h[H:2*H,0:W]=f
  h[0:H,W:2*W]=f
  h[H:2*H,W:2*W]=f

  g=np.zeros([H,W])
  g=h[r:r+H,c:c+W]

  return g

def PSNR(original, compressed):
  mse = np.mean((original - compressed) ** 2)
  if(mse == 0):  # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
      return 100
  original=(original-np.min(original))/(np.max(original)-np.min(original))
  compressed=(compressed-np.min(compressed))/(np.max(compressed)-np.min(compressed))
  max_pixel = 1.0
  psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
  return psnr

# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=0.01, sy=0.01):
  return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def gauss(s,sx,sy=None):
    #s: shape
    #sy, sx: sigma_y and sigma_x
    
    if sy is None:
        sy=sx
    mx=0
    my=0
    
    xx = np.linspace(-1, 1, s[1])
    yy = np.linspace(-1, 1, s[0])*s[0]/s[1]
    xx, yy = np.meshgrid(xx, yy)   
    
    wind = 1. / (2. * np.pi * sx * sy) * np.exp(-((xx - mx)**2. / (2. * sx**2.) + (yy - my)**2. / (2. * sy**2.)))
    wind = (wind-np.min(wind))/(np.max(wind)-np.min(wind))
    
    return wind

def psf2otf(psf, otf_size):
    # calculate otf from psf with size >= psf size
    
    if psf.any(): # if any psf element is non-zero    
        # pad PSF with zeros up to image size  
        pad_size = ((0,otf_size[0]-psf.shape[0]),(0,otf_size[1]-psf.shape[1]))
        psf_padded = np.pad(psf, pad_size, 'constant')    
        
        # circularly shift psf   
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[0]/2)), axis=0)    
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[1]/2)), axis=1)       
       
       #calculate otf    
        otf = fftn(psf_padded)
        # this condition depends on psf size    
        num_small = np.log2(psf.shape[0])*4*np.spacing(1)    
        if np.max(abs(otf.imag))/np.max(abs(otf)) <= num_small:
            otf = otf.real 
    else: # if all psf elements are zero
        otf = np.zeros(otf_size)
    return otf

def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size
    
    if otf.any(): # if any otf element is non-zero
        # calculate psf     
        psf = ifftn(otf)
        # this condition depends on psf size    
        num_small = np.log2(otf.shape[0])*4*np.spacing(1)    
        if np.max(abs(psf.imag))/np.max(abs(psf)) <= num_small:
            psf = psf.real 
        
        # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0]/2)), axis=0)    
        psf = np.roll(psf, int(np.floor(psf_size[1]/2)), axis=1) 
        
        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else: # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf

def denoise_L1(I,R):
    I = (I/(np.abs(I)+1e-10)) * np.maximum(np.abs(I) - R, 0)
    return I

def denoise_L1projection(I, R):
  #I: Image to be projected onto the L1 Ball
  #R: Ball Radius 

  [N1, N2] = I.shape
  I_ = I.ravel()
  sign_array = np.sign(I_)

  #Determining lbda
  idx = np.argwhere((I_- R) > 0)
  K = 0
  sum = 0
  for id in idx:
    K += 1
    sum += I_[id] - R
  if(K > 0):
    lbda = sum/K
  else:
    lbda = 0

  #Projection
  I_[idx] = I_[idx] - lbda
  I_ = np.multiply(I_, sign_array)
  I = np.reshape(I_, (N1, N2))

  return I

def blind_decon2D_red_fp_L1(R0,X,OTF,Niter,lamb,lambdaPSF,f,radius,R_true,q,sa):
  print('L1 '+str(sa))
  R=R0.copy()
  OTF0 = OTF.copy()
  scale_percent = 100
  width = int(OTF0.shape[1] * scale_percent / 100)
  height = int(OTF0.shape[0] * scale_percent / 100)
  dim = (height, width)
  PSF  = otf2psf(OTF0, dim)
  OTF  = psf2otf(PSF, OTF0.shape)

  fftX=fft2(X)
  fftR=fft2(R)
  fftHtH=np.conjugate(OTF)*OTF
  fftHtX=np.conjugate(OTF)*fftX 

  for ii in range(Niter):
    # denoise R - L1

    RD = denoise_wavelet(R)
    RD=f(RD, radius)
    #RD=ndimage.median_filter(RD, size=3)

    b=fftHtX+lamb*(fft2(RD))
    A=fftHtH+lamb
    fftR=np.nan_to_num(b/A)
    R=np.real(ifft2(fftR))

    # update OTF
    #OTF0 = ((np.conjugate(fftR)*fftX)/(fftR*np.conjugate(fftR)+lambdaPSF))
    # if not (ii+1)%10:
    #   print("Update OTF")
    #   OTF0 = ((np.conjugate(fftR)*fftX)/(fftR*np.conjugate(fftR)+lambdaPSF))
    #   PSF = otf2psf(OTF0, dim)
    #   OTF = psf2otf(PSF, OTF0.shape)
    #   fftHtH=np.conjugate(OTF)*OTF
    #   fftHtX=np.conjugate(OTF)*fftX 

    psnr=PSNR(R_true,R)
    #print('iter: ',ii,'\t PSNR=',psnr)
  
  q.put((sa,R))
  return


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def YUV(endereco):
    return rgb2yuv(endereco)

def RGB(endereco):
    return yuv2rgb(endereco)
 
if __name__ == "__main__":
  deblur()
