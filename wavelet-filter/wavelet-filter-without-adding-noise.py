import numpy as np
import cv2
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Funkcija za razinski DWT s ciljnim razinama
def dwt2(image, target_levels):
    h, w = image.shape
    if h % 2 != 0:
        image = np.pad(image, ((0, 1), (0, 0)), 'constant')
        h += 1
    if w % 2 != 0:
        image = np.pad(image, ((0, 0), (0, 1)), 'constant')
        w += 1
    
    LL = (image[0::2, 0::2] + image[1::2, 0::2] + image[0::2, 1::2] + image[1::2, 1::2]) / 4
    LH = (image[0::2, 0::2] - image[1::2, 0::2] + image[0::2, 1::2] - image[1::2, 1::2]) / 4
    HL = (image[0::2, 0::2] + image[1::2, 0::2] - image[0::2, 1::2] - image[1::2, 1::2]) / 4
    HH = (image[0::2, 0::2] - image[1::2, 0::2] - image[0::2, 1::2] + image[1::2, 1::2]) / 4
    
    if target_levels == 1:
        return LL, [(LH, HL, HH)]
    
    LL, sub_details = dwt2(LL, target_levels - 1)
    return LL, sub_details + [(LH, HL, HH)]

# Funkcija za razinski IDWT
def idwt2(LL, details):
    if not details:
        return LL
    
    LH, HL, HH = details[0]
    
    h, w = LL.shape
    h_lh, w_lh = LH.shape
    h_hl, w_hl = HL.shape
    h_hh, w_hh = HH.shape
    
    if (h, w) != (h_lh, w_lh):
        LH = cv2.resize(LH, (w, h))
    if (h, w) != (h_hl, w_hl):
        HL = cv2.resize(HL, (w, h))
    if (h, w) != (h_hh, w_hh):
        HH = cv2.resize(HH, (w, h))
    
    image = np.zeros((h * 2, w * 2), dtype=np.float64)
    image[0::2, 0::2] = LL + LH + HL + HH
    image[1::2, 0::2] = LL - LH + HL - HH
    image[0::2, 1::2] = LL + LH - HL - HH
    image[1::2, 1::2] = LL - LH - HL + HH
    
    return idwt2(image, details[1:])

# Funkcija za soft thresholding
def soft_thresholding(data, threshold):
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

# Funkcija za hard thresholding
def hard_thresholding(data, threshold):
    return data * (np.abs(data) > threshold)

# Funkcija za wavelet filtriranje s thresholdingom
def wavelet_filter(image, target_levels, threshold, threshold_type):
    # Pretvaranje slike u float64 radi preciznosti
    image = image.astype(np.float64)
    
    # Primijena DWT
    LL, details = dwt2(image, target_levels)
    
    # Primijena odabranog thresholdinga 
    if threshold_type == 'soft':
        details = [(soft_thresholding(d[0], threshold), 
                    soft_thresholding(d[1], threshold), 
                    soft_thresholding(d[2], threshold)) for d in details]
    elif threshold_type == 'hard':
        details = [(hard_thresholding(d[0], threshold), 
                    hard_thresholding(d[1], threshold), 
                    hard_thresholding(d[2], threshold)) for d in details]
    
    # Primijena IDWT
    denoised_image = idwt2(LL, details)
    
    # Vraćanje slike na originalne dimenzije ako su promijenjene
    denoised_image = denoised_image[:image.shape[0], :image.shape[1]]
    
    # Pretvaranje slike natrag u uint8
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    return denoised_image, LL, details

# Funkcija za izračun PSNR
def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=img1.max() - img1.min())

# Funkcija za izračun SSIM
def calculate_ssim(img1, img2):
    s, _ = ssim(img1, img2, full=True)
    return s

# Funkcija za odabir slike
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
    
    # Učitavanje slike u grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Učitavanje slike nije uspjelo. Izlazim...")
        return
    
    # Provjera dimenzija ulazne slike
    original_shape = image.shape
    
    # Primijena wavelet filtriranja na originalnu sliku
    target_levels = 2  # Broj razina za DWT
    denoised_image_soft, LL_soft, details_soft = wavelet_filter(image, target_levels, threshold=25, threshold_type='soft')  
    denoised_image_hard, LL_hard, details_hard = wavelet_filter(image, target_levels, threshold=25, threshold_type='hard')
    
    # Provjera da li su dimenzije jednake
    if denoised_image_soft.shape != original_shape or denoised_image_hard.shape != original_shape:
        print("Dimenzije filtriranih slika se ne podudaraju s originalom. Izlazim...")
        return
    
    # Izračun PSNR i SSIM između originalne slike i slike nakon uklanjanja šuma 
    psnr_denoised_soft = calculate_psnr(image, denoised_image_soft)
    ssim_denoised_soft = calculate_ssim(image, denoised_image_soft)
    psnr_denoised_hard = calculate_psnr(image, denoised_image_hard)
    ssim_denoised_hard = calculate_ssim(image, denoised_image_hard)
    
    print(f'PSNR između originalne i slike nakon uklanjanja šuma (soft thresholding): {psnr_denoised_soft} dB')
    print(f'SSIM između originalne i slike nakon uklanjanja šuma (soft thresholding): {ssim_denoised_soft}')
    print(f'PSNR između originalne i slike nakon uklanjanja šuma (hard thresholding): {psnr_denoised_hard} dB')
    print(f'SSIM između originalne i slike nakon uklanjanja šuma (hard thresholding): {ssim_denoised_hard}')
    
    # Prikaz rezultata
    plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 5, 1)
    plt.title('Originalna slika')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 2)
    plt.title('Slika nakon uklanjanja šuma (Soft)\nPSNR: {:.2f} dB\nSSIM: {:.4f}'.format(psnr_denoised_soft, ssim_denoised_soft))
    plt.imshow(denoised_image_soft, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 3)
    plt.title('Slika nakon uklanjanja šuma (Hard)\nPSNR: {:.2f} dB\nSSIM: {:.4f}'.format(psnr_denoised_hard, ssim_denoised_hard))
    plt.imshow(denoised_image_hard, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 4)
    plt.title('LL')
    plt.imshow(LL_soft, cmap='gray')
    plt.axis('off')

    # Prikaz LH, HL, HH potpodručja za soft thresholding
    LH_soft, HL_soft, HH_soft = details_soft[0]
    plt.subplot(3, 5, 5)
    plt.title('LH (Soft)')
    plt.imshow(LH_soft, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 6)
    plt.title('HL (Soft)')
    plt.imshow(HL_soft, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 7)
    plt.title('HH (Soft)')
    plt.imshow(HH_soft, cmap='gray')
    plt.axis('off')

    # Prikaz LH, HL, HH potpodručja za hard thresholding
    LH_hard, HL_hard, HH_hard = details_hard[0]
    plt.subplot(3, 5, 8)
    plt.title('LH (Hard)')
    plt.imshow(LH_hard, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 9)
    plt.title('HL (Hard)')
    plt.imshow(HL_hard, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 5, 10)
    plt.title('HH (Hard)')
    plt.imshow(HH_hard, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
