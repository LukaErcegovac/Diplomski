import numpy as np
import cv2
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Funkcija za izdvajanje susjedstva piksela
def extract_neighborhood(image, p, f):
    padded_image = np.pad(image, ((f, f), (f, f)), 'reflect')
    i, j = p
    neighborhood = padded_image[i:i + 2 * f + 1, j:j + 2 * f + 1]
    return neighborhood

# Funkcija za izračunavanje težine 
def compute_weight(image, p, q, f, sigma, h):
    B_p_f = extract_neighborhood(image, p, f)
    B_q_f = extract_neighborhood(image, q, f)
    d_squared = np.sum((B_p_f - B_q_f) ** 2) / (2 * f + 1) ** 2
    weight = np.exp(-max(d_squared - 2 * sigma ** 2, 0.0) / h ** 2)
    return weight

# Funkcija za primjenu filtra nelokalnog usrednjavanja (non-local means filter)
def nonlocal_means_filter(image, f=1, h=10.0, sigma=10.0):
    filtered_image = np.zeros_like(image, dtype=np.float64)
    padded_image = np.pad(image, ((f, f), (f, f)), 'reflect')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            p = (i + f, j + f)
            weights = []
            neighborhood_weights = []
            for m in range(-f, f + 1):
                for n in range(-f, f + 1):
                    q = (i + m + f, j + n + f)
                    weight = compute_weight(padded_image, p, q, f, sigma, h)
                    weights.append(weight)
                    neighborhood_weights.append(padded_image[q[0], q[1]])
            weights = np.array(weights)
            neighborhood_weights = np.array(neighborhood_weights)
            C_p = np.sum(weights)
            filtered_image[i, j] = np.sum(weights * neighborhood_weights) / C_p
    return filtered_image.astype(np.uint8)

# Funkcija za izračunavanje PSNR (peak signal-to-noise ratio)
def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=img1.max() - img1.min())

# Funkcija za izračunavanje SSIM (structural similarity index)
def calculate_ssim(img1, img2):
    s, _ = ssim(img1, img2, full=True)
    return s

# Funkcija za odabir slike 
def choose_image():
    Tk().withdraw()  
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.gif;*.tiff")])
    return file_path

# Glavna funkcija
def main():
    file_path = choose_image()
    if not file_path:
        print("Nijedna datoteka nije odabrana. Izlazim...")
        return
    
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
    if image is None:
        print("Učitavanje slike nije uspjelo. Izlazim...")
        return
    
    sigma = 10.0
    
    denoised_image = nonlocal_means_filter(image, f=1, h=0.40*sigma, sigma=10.0) 

    psnr_denoised = calculate_psnr(image, denoised_image)
    ssim_denoised = calculate_ssim(image, denoised_image)
    
    print(f'PSNR između originalne slike i Slike nakon uklanjanja šuma: {psnr_denoised} dB')
    print(f'SSIM između originalne slike i Slike nakon uklanjanja šuma: {ssim_denoised}')
    
    # Prikaz slika
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Originalna slika')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Slika nakon uklanjanja šuma\nPSNR: {:.2f} dB\nSSIM: {:.4f}'.format(psnr_denoised, ssim_denoised))
    plt.imshow(denoised_image, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()
