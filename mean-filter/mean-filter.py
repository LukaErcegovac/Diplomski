import numpy as np
import cv2
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

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

# Funkcija za primjenu metode prostornog usrednjavanja s 3x3 kernelom
def mean_filter(image):
    kernel_size = 3
    # Dodavanje rubova nula oko slike za rukovanje pikselima na rubu
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), 'constant', constant_values=0)
    # Inicijalizacija prazne slike za filtriranu sliku
    filtered_image = np.zeros_like(image)
    
    # Iteracija kroz sve piksele u slici
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Izdvajanje područja interesa
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Izračunavanje srednje vrijednosti područja
            filtered_image[i, j] = np.mean(region)
    
    return filtered_image

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
    
    # Primjena metode prostornog usrednjavanja na sliku sa šumom
    denoised_image = mean_filter(noisy_image)
    
    # Izračunavanje PSNR i SSIM između originalne slike i slike sa šumom
    psnr_noisy = calculate_psnr(image, noisy_image)
    ssim_noisy = calculate_ssim(image, noisy_image)
    
    # Izračunavanje PSNR i SSIM između originalne slike i slike bez šuma
    psnr_denoised = calculate_psnr(image, denoised_image)
    ssim_denoised = calculate_ssim(image, denoised_image)
    
    print(f'PSNR između originalne slike i slike sa šumom: {psnr_noisy} dB')
    print(f'SSIM između originalne slike i slike sa šumom: {ssim_noisy}')
    print(f'PSNR između originalne slike i slike bez šuma: {psnr_denoised} dB')
    print(f'SSIM između originalne slike i slike bez šuma: {ssim_denoised}')
    
    # Prikaz originalne, slike sa šumom i bez šuma na slici
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
    plt.title('Slika bez šuma\nPSNR: {:.2f} dB\nSSIM: {:.4f}'.format(psnr_denoised, ssim_denoised))
    plt.imshow(denoised_image, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
