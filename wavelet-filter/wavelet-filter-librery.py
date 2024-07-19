import numpy as np
import cv2
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_wavelet

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

# Funkcija za denoising pomoću valne transformacije koristeći skimage
def wavelet_denoising(image):
    denoised_image = denoise_wavelet(image, method='BayesShrink', mode='soft',
                                     wavelet_levels=3, wavelet='db1', rescale_sigma=True)
    denoised_image = (255 * denoised_image).astype(np.uint8)
    return denoised_image

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

    # Dodavanje speckle šuma na sliku s povećanom varijansom
    noisy_image = add_speckle_noise(image, var=0.2, mean=0.0)

    # Primjena valnog denoisinga na sliku sa šumom koristeći skimage
    denoised_image_wavelet = wavelet_denoising(noisy_image)

    # Izračunavanje PSNR i SSIM između originalne slike i slike sa šumom
    psnr_noisy = calculate_psnr(image, noisy_image)
    ssim_noisy = calculate_ssim(image, noisy_image)

    # Izračunavanje PSNR i SSIM između originalne slike i slike bez šuma koristeći valni denoising
    psnr_denoised_wavelet = calculate_psnr(image, denoised_image_wavelet)
    ssim_denoised_wavelet = calculate_ssim(image, denoised_image_wavelet)

    print(f'PSNR između originalne slike i slike sa šumom: {psnr_noisy} dB')
    print(f'SSIM između originalne slike i slike sa šumom: {ssim_noisy}')
    print(f'PSNR između originalne slike i slike bez šuma (Wavelet): {psnr_denoised_wavelet} dB')
    print(f'SSIM između originalne slike i slike bez šuma (Wavelet): {ssim_denoised_wavelet}')

    # Prikaz originalne, slike sa šumom i bez šuma na slici
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Originalna slika')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f'Slika sa šumom\nPSNR: {psnr_noisy:.2f} dB\nSSIM: {ssim_noisy:.4f}')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'Slika bez šuma (Wavelet)\nPSNR: {psnr_denoised_wavelet:.2f} dB\nSSIM: {ssim_denoised_wavelet:.4f}')
    plt.imshow(denoised_image_wavelet, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
