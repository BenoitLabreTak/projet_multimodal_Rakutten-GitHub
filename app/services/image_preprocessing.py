# === IMPORTS DES LIBRAIRIES ===
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from torchvision import transforms

import app.core.config as config 


def preprocess_image_from_pil(image: Image.Image) -> Image.Image:
    """
    Prétraite une image PIL et retourne l'image transformée (PIL).
    """
    # Étapes PIL
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)

    # Ajustement des canaux
    img_array = np.array(image).astype(np.float32)
    img_array[:, :, 0] *= 0.9
    img_array[:, :, 1] *= 1.05
    img_array[:, :, 2] *= 1.05
    image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    # Transforms PyTorch
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)

    # Retourner image PIL finale
    processed_image = transforms.ToPILImage()(tensor.squeeze(0))
    return processed_image

def crop_borders(image, ratio=0.5):
    """Crops borders, leaving a maximum border of (1-ratio)%."""
    gray_image = tf.image.rgb_to_grayscale(image)
    gray_image = tf.cast(gray_image, dtype=tf.float32) / 255.0

    edges = tf.image.sobel_edges(tf.expand_dims(gray_image, 0))[0, :, :, 0]
    edges = tf.abs(edges)
    edges = tf.reduce_sum(edges, axis=-1)

    threshold = tf.reduce_max(edges) * 0.1
    binary_edges = tf.cast(edges > threshold, dtype=tf.bool)

    rows = tf.reduce_any(binary_edges, axis=1)
    cols = tf.reduce_any(binary_edges, axis=0)

    if tf.reduce_any(rows) and tf.reduce_any(cols):
        rmin = tf.reduce_min(tf.where(rows)[:, 0], axis=0)
        rmax = tf.reduce_max(tf.where(rows)[:, 0], axis=0)
        cmin = tf.reduce_min(tf.where(cols)[:, 0], axis=0)
        cmax = tf.reduce_max(tf.where(cols)[:, 0], axis=0)

        cropped_image = image[rmin:rmax + 1, cmin:cmax + 1, :]
        return cropped_image
    else:
        return image

def preprocess_image_enhance(image_path, sharpness_factor=2.5, global_enhancement_factor=1, brightness_factor_range=(0.85, 1.15)):
    """
    Améliore une image via un pipeline adaptatif incluant :
    - recadrage intelligent
    - ajustements de netteté, saturation, luminosité
    - ajustement des canaux rouge, vert, bleu

    Args:
        image_path (str): Chemin vers l’image
        sharpness_factor (float): Intensité de la netteté
        global_enhancement_factor (float): Multiplicateur général
        brightness_factor_range (tuple): Min et max de facteur de luminosité

    Returns:
        np.ndarray: Image traitée (uint8), ou None si erreur
    """
    try:
        # Chargement de l’image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32) * 255.0

        original_sharpness = tf.reduce_mean(tf.abs(tf.image.sobel_edges(tf.expand_dims(image / 255.0, 0))[0]))
        mean_brightness = tf.reduce_mean(image)
        mean_saturation = tf.reduce_mean(tf.image.rgb_to_hsv(image / 255.0)[..., 1])
        mean_red = tf.reduce_mean(image[..., 0])
        mean_green = tf.reduce_mean(image[..., 1])
        mean_blue = tf.reduce_mean(image[..., 2])

        # Recadrage intelligent
        cropped_image = crop_borders(tf.cast(image, tf.uint8))
        cropped_image = tf.image.resize(cropped_image, [500, 500], method='bicubic')

        cropped_sharpness = tf.reduce_mean(tf.abs(tf.image.sobel_edges(tf.expand_dims(cropped_image / 255.0, 0))[0]))
        sharpness_diff = cropped_sharpness - original_sharpness

        # Netteté adaptative
        adaptive_sharpness_factor = tf.clip_by_value(
            (sharpness_factor * original_sharpness) * global_enhancement_factor + sharpness_diff * 0.5,
            1.0, 5.0
        )
        pil_image = Image.fromarray(tf.cast(cropped_image, tf.uint8).numpy())
        pil_image = ImageEnhance.Sharpness(pil_image).enhance(adaptive_sharpness_factor.numpy())
        cropped_image = tf.convert_to_tensor(np.array(pil_image), dtype=tf.float32)

        # Ajustements RGB
        red_factor = tf.clip_by_value(1 - (mean_red / 255.0) * 0.05 * global_enhancement_factor, 0.9, 1.0)
        green_factor = tf.clip_by_value(1 - (mean_green / 255.0) * 0.05 * global_enhancement_factor, 0.9, 1.0)
        blue_factor = tf.clip_by_value(1 - (mean_blue / 255.0) * 0.05 * global_enhancement_factor, 0.9, 1.0)

        red_channel = cropped_image[..., 0] * red_factor
        green_channel = cropped_image[..., 1] * green_factor
        blue_channel = cropped_image[..., 2] * blue_factor

        adjusted_rgb = tf.stack([red_channel, green_channel, blue_channel], axis=-1)
        adjusted_rgb = tf.clip_by_value(adjusted_rgb, 0, 255)

        # Saturation + Brightness
        adaptive_saturation = 1 + (mean_saturation - 0.5) * global_enhancement_factor
        adaptive_brightness = tf.clip_by_value(
            1 + (brightness_factor_range[1] - 1) * (1 - mean_brightness / 255.0), 1.0, 1.4
        )
        brightness_delta = adaptive_brightness - 1.0

        final_img = tf.image.adjust_saturation(adjusted_rgb / 255.0, adaptive_saturation) * 255.0
        final_img = tf.image.adjust_brightness(final_img / 255.0, brightness_delta) * 255.0
        final_img = tf.clip_by_value(final_img, 0, 255)

        return final_img.numpy().astype(np.uint8)

    except Exception as e:
        print(f"❌ Erreur dans enhance_images_per_image : {e}")
        return None

def display_image_transformation(full_path):
    try:
        transformed_np = enhance_images_per_image(full_path)
        if transformed_np is None:
            return None
        return Image.fromarray(transformed_np)
    except Exception as e:
        print(f"Erreur affichage transformation : {e}")
        return None
