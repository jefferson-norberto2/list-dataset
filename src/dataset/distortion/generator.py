import io
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class DistortionGenerator:
    def __init__(self):
        self.img_uint8 = None
        self.img = None

        self.height = None
        self.width = None
        self.channels = None

        self.img_name = None
        self.folder_path = None
        self.root_path = None

    def change_image(self, new_image_path: str):
        """Permite alterar a imagem de entrada."""
        self.img_uint8 = cv2.imread(new_image_path)

        if self.img_uint8 is None:
            raise ValueError(f"Image not found at path: {new_image_path}.")

        self.img = self.img_uint8.astype(np.float32) / 255.0
        self.height, self.width, self.channels = self.img.shape
        self.img_name = self._get_image_name(new_image_path)
        self.folder_path = os.path.dirname(new_image_path)
        self.root_path = os.path.dirname(self.folder_path)

    def _get_image_name(self, image_path: str):
        return os.path.splitext(os.path.basename(image_path))[0]

    def _check_image_loaded(self):
        if self.img_uint8 is None:
            raise ValueError("No image loaded. Use change_image() to load an image.")

    def save_output(self, distorted_img, distortion_name, level):
        """Saves the resulting image converting back to uint8."""
        self._check_image_loaded()
        img_clipped = np.clip(distorted_img, 0, 1)
        img_out = (img_clipped * 255).astype(np.uint8)
        save_path = f"{self.root_path}/{distortion_name}"
        os.makedirs(save_path, exist_ok=True)
        file_path = f"{save_path}/dist_{level}_{self.img_name}.png"
        cv2.imwrite(file_path, img_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # 1. Gaussian Blur
    def add_gaussian_blur(self, sigma=2.0, force_kernel=None):
        self._check_image_loaded()
        if force_kernel is not None:
            k = int(force_kernel)
            k = k if k % 2 != 0 else k + 1
            kernel_size = (k, k)
        else:
            kernel_size = (0, 0)
        blurred = cv2.GaussianBlur(self.img, kernel_size, sigma)
        return blurred

    # 2. Poisson-Gaussian Noise
    def add_poisson_gaussian_noise(self, shot_noise=0.01, read_noise=0.001):
        self._check_image_loaded()
        variance_map = (self.img * shot_noise) + read_noise
        sigma_map = np.sqrt(np.clip(variance_map, 0, None))
        noise = np.random.normal(0, 1, self.img.shape) * sigma_map
        noisy_img = self.img + noise
        return noisy_img

    # 3. JPEG Compression
    def add_jpeg_compression(self, quality=10):
        self._check_image_loaded()
        pil_img = Image.fromarray(np.clip(self.img * 255, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        pil_img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        jpeg_img = Image.open(buffer)
        img_np = np.array(jpeg_img)
        return img_np.astype(np.float32) / 255.0

    # 4. JPEG-2000 Compression
    def add_jpeg2000_compression(self, compression_ratio=20):
        self._check_image_loaded()
        img_u8 = np.clip(self.img * 255, 0, 255).astype(np.uint8)
        temp_filename = f"temp_dist_{np.random.randint(1000, 9999)}.jp2"
        try:
            cv2.imwrite(
                temp_filename,
                img_u8,
                [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio * 10],
            )
            jp2_img = cv2.imread(temp_filename)
            os.remove(temp_filename)
            if jp2_img is not None:
                return jp2_img.astype(np.float32) / 255.0
            return self.img
        except Exception:  # noqa: BLE001
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return self.img

    # 5. Global Contrast Decrements
    def change_contrast(self, alpha=0.5):
        self._check_image_loaded()
        gray_mean = 0.5
        contrast_img = alpha * (self.img - gray_mean) + gray_mean
        return contrast_img

    # 6. Additive Pink Gaussian Noise
    def add_pink_noise(self, intensity=0.05, spatial_scale=1.0):
        self._check_image_loaded()

        def pink_noise_2d(shape):
            rows, cols = shape
            u = np.fft.fftfreq(rows)
            v = np.fft.fftfreq(cols)
            u_grid, v_grid = np.meshgrid(u, v, indexing="ij")

            f = np.sqrt(u_grid**2 + v_grid**2)
            f[0, 0] = 1.0

            scale = 1.0 / (f**spatial_scale + 1e-5)

            white_noise_spec = np.fft.fft2(np.random.normal(0, 1, shape))
            pink_spec = white_noise_spec * scale
            pink_noise = np.fft.ifft2(pink_spec).real

            pink_noise = (pink_noise - np.mean(pink_noise)) / (
                np.std(pink_noise) + 1e-8
            )
            return pink_noise

        noise_layer = np.zeros_like(self.img)
        for ch in range(self.channels):
            noise_layer[:, :, ch] = pink_noise_2d((self.height, self.width))

        noisy_img = self.img + (noise_layer * intensity)
        return noisy_img


if __name__ == "__main__":
    try:
        generator = DistortionGenerator()

        folder_path = "Datasets/LIVE_RGB/src"
        if not os.path.exists(folder_path):
            print(f"Diretório não encontrado: {folder_path}")
            files = []
        else:
            files = [
                file
                for file in os.listdir(folder_path)
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]

        # --- DEFINIÇÃO DOS 5 NÍVEIS DE DISTORÇÃO ---
        levels = {
            "blur_sigma": [1.0, 1.5, 2.0, 3.0, 4.0],
            "pgn_shot": [0.005, 0.015, 0.03, 0.05, 0.08],
            "pgn_read": [0.0005, 0.001, 0.002, 0.003, 0.005],
            "jpeg_quality": [20, 15, 10, 8, 3],  # Qualidade DIMINUI com o nível
            "jp2k_ratio": [1, 4, 7, 10, 13],  # Compressão AUMENTA com o nível
            "contrast_alpha": [
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
            ],  # Alpha DIMINUI com o nível (menor = menos contraste)
            "pink_intensity": [0.03, 0.07, 0.12, 0.18, 0.25],
        }

        for file in tqdm(files, desc="Processando imagens"):
            image_path = os.path.join(folder_path, file)
            try:
                generator.change_image(image_path)
            except Exception as e:  # noqa: BLE001
                print(f"Erro ao carregar imagem {file}: {e}")
                continue

            # Aplica os 5 níveis para cada tipo de distorção
            for lvl in range(5):
                level_name = lvl + 1  # Salvará como 1, 2, 3, 4, 5

                # 1. Blur
                res_blur = generator.add_gaussian_blur(
                    sigma=levels["blur_sigma"][lvl], force_kernel=None
                )
                generator.save_output(res_blur, "blur", level_name)

                # 2. Poisson-Gaussian Noise
                res_pgn = generator.add_poisson_gaussian_noise(
                    shot_noise=levels["pgn_shot"][lvl],
                    read_noise=levels["pgn_read"][lvl],
                )
                generator.save_output(res_pgn, "awgn", level_name)

                # 3. JPEG
                res_jpg = generator.add_jpeg_compression(
                    quality=levels["jpeg_quality"][lvl]
                )
                generator.save_output(res_jpg, "jpeg", level_name)

                # 4. JPEG 2000
                res_jp2 = generator.add_jpeg2000_compression(
                    compression_ratio=levels["jp2k_ratio"][lvl]
                )
                generator.save_output(res_jp2, "jpeg2000", level_name)

                # 5. Contrast Decrement
                res_contrast = generator.change_contrast(
                    alpha=levels["contrast_alpha"][lvl]
                )
                generator.save_output(res_contrast, "contrast", level_name)

                # 6. Pink Noise (Escala espacial fixa para consistência, apenas a intensidade muda)
                res_pink = generator.add_pink_noise(
                    intensity=levels["pink_intensity"][lvl], spatial_scale=1.0
                )
                generator.save_output(res_pink, "fnoise", level_name)

    except Exception as e:  # noqa: BLE001
        print(f"Erro fatal: {e}")
