import numpy as np
import cv2
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.filters.rank import mean
from skimage.morphology import square

# Funkcija za dodavanje speckle šuma na sliku koristeći skimage
def add_speckle_noise(image, var, mean):
    """
    Blagi šum: var=0.01
    Umjereni šum: var=0.1
    Visoki šum: var=0.3
    Vrlo visoki šum: var=0.5

    'speckle'
            Multiplicative noise using ``out = image + n * image``, where ``n``
            is Gaussian noise with specified mean & variance.
    """
    image = image / 255.0  
    noisy_image = random_noise(image, mode='speckle', var=var, mean=mean)
    noisy_image = (255 * noisy_image).astype(np.uint8)
    
    return noisy_image

# Funkcija za primjenu metode prostornog usrednjavanja s 3x3 kernelom koristeći skimage
def mean_filter_skimage(image):
    return mean(image, square(3))

# Funkcija za izračunavanje PSNR
def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=img1.max() - img1.min())

# Funkcija za izračunavanje SSIM
def calculate_ssim(img1, img2):
    s, _ = ssim(img1, img2, full=True)
    return s

def choose_image():
    Tk().withdraw() 
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.gif;*.tiff")])
    return file_path

def main():
    # Odabir slike
    file_path = choose_image()
    if not file_path:
        print("Nijedna datoteka nije odabrana. Izlazim...")
        return
    
    # Učitavanje slike u sivoj skali
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Učitavanje slike nije uspjelo. Izlazim...")
        return
    
    # Dodavanje speckle šuma na sliku 
    noisy_image = add_speckle_noise(image, var=0.2, mean=0.0)
    
    # Primjena metode prostornog usrednjavanja na sliku sa šumom koristeći skimage
    denoised_image_skimage = mean_filter_skimage(noisy_image)
    
    # Izračunavanje PSNR i SSIM između originalne slike i slike sa šumom
    psnr_noisy = calculate_psnr(image, noisy_image)
    ssim_noisy = calculate_ssim(image, noisy_image)
    
    # Izračunavanje PSNR i SSIM između originalne slike i slike nakon uklanjanja šuma koristeći skimage mean filter
    psnr_denoised_skimage = calculate_psnr(image, denoised_image_skimage)
    ssim_denoised_skimage = calculate_ssim(image, denoised_image_skimage)
    
    print(f'PSNR između originalne slike i slike sa šumom: {psnr_noisy} dB')
    print(f'SSIM između originalne slike i slike sa šumom: {ssim_noisy}')
    print(f'PSNR između originalne slike i slike nakon uklanjanja šuma: {psnr_denoised_skimage} dB')
    print(f'SSIM između originalne slike i slike nakon uklanjanja šuma: {ssim_denoised_skimage}')
    
    # Prikazivanje originalne slike, slike sa šumom i slike nakon uklanjanja šuma
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Originalna slika')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Slika sa šumom\nPSNR: {:.2f} dB\nSSIM: {:.4f}'.format(psnr_noisy, ssim_noisy))
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Slika nakon uklanjanja šuma\n PSNR: {:.2f} dB\nSSIM: {:.4f}'.format(psnr_denoised_skimage, ssim_denoised_skimage))
    plt.imshow(denoised_image_skimage, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
