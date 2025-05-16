import os
from PIL import Image
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import hashlib
import config

pio.renderers.default = 'browser'


def get_image_files(source_folder, limit=500):
    all_files = sorted(os.listdir(source_folder))
    return [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:limit]


def check_image_sizes(source_folder, image_files):
    sizes = []
    for img_name in image_files:
        image_path = os.path.join(source_folder, img_name)
        try:
            with Image.open(image_path) as img:
                sizes.append(img.size)
        except Exception as e:
            continue

    unique_sizes = set(sizes)
    return unique_sizes



def detect_duplicates(source_folder, image_files, show_examples=True):
    print("\n🔍 Vérification des doublons d'images...")

    hash_dict = {}
    duplicates = []
    duplicate_pairs = []

    for img_name in image_files:
        image_path = os.path.join(source_folder, img_name)
        try:
            with open(image_path, 'rb') as f:
                file_data = f.read()
                file_hash = hashlib.md5(file_data).hexdigest()

            if file_hash in hash_dict:
                original = hash_dict[file_hash]
                duplicates.append(img_name)
                duplicate_pairs.append((original, img_name))
            else:
                hash_dict[file_hash] = img_name

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture de {img_name}: {e}")

    total_images = len(image_files)
    total_duplicates = len(duplicates)
    duplicate_percentage = (total_duplicates / total_images) * 100

    if total_duplicates > 0:
        print(f"\n🚨 {total_duplicates} doublon(s) détecté(s) sur {total_images} images.")
        print(f"📊 Cela représente {duplicate_percentage:.2f}% du dataset.")
        print("🗂️ Fichiers dupliqués :")
        print(", ".join(duplicates))

        if show_examples:
            print("\n🖼️ Affichage de 2 exemples de doublons...")
            for i, (original, dup) in enumerate(duplicate_pairs[:2]):
                orig_path = os.path.join(source_folder, original)
                dup_path = os.path.join(source_folder, dup)

                img1 = cv2.imread(orig_path)
                img2 = cv2.imread(dup_path)

                if img1 is not None and img2 is not None:
                    combined = np.hstack((img1, img2))
                    cv2.imshow(f"Exemple de doublon {i+1}: {original} | {dup}", combined)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:
        print("✅ Aucun doublon détecté.")



def plot_color_distribution(source_folder, image_files):
    avg_colors = []
    for img_name in image_files:
        image_path = os.path.join(source_folder, img_name)
        image = cv2.imread(image_path)
        if image is not None:
            mean_color = np.mean(image, axis=(0, 1))  # BGR
            avg_colors.append(mean_color)

    avg_colors = np.array(avg_colors)
    bins = np.linspace(0, 255, 50)

    fig = go.Figure()
    colors = ['blue', 'green', 'red']
    channel_names = ['Bleu', 'Vert', 'Rouge']
    color_stats = {}

    for i in range(3):
        hist, bin_edges = np.histogram(avg_colors[:, i], bins=bins)
        percentages = (hist / len(avg_colors)) * 100
        color_stats[channel_names[i]] = avg_colors[:, i].mean()

        fig.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=percentages,
            name=channel_names[i],
            marker_color=colors[i],
            hovertemplate=f'{channel_names[i]}: %{{x:.0f}}<br>% d\'images: %{{y:.2f}}%',
        ))

    fig.update_layout(
        title="Distribution des couleurs moyennes (par canal)",
        xaxis_title="Valeur d'intensité (0-255)",
        yaxis_title="Pourcentage d'images",
        barmode='overlay',
        bargap=0.1,
        template="plotly_white"
    )

    return fig, color_stats

def plot_sharpness_distribution(source_folder, image_files):
    sharpness_values = []
    for img_name in image_files:
        image_path = os.path.join(source_folder, img_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            variance = laplacian.var()
            sharpness_values.append(variance)

    if len(sharpness_values) == 0:
        return None, None

    bins = np.linspace(0, max(sharpness_values), 50)
    hist, bin_edges = np.histogram(sharpness_values, bins=bins)
    percentages = (hist / len(sharpness_values)) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_edges[:-1],
        y=percentages,
        name='Netteté',
        marker_color='purple',
        hovertemplate='Variance: %{x:.2f}<br>% d\'images: %{y:.2f}%',
    ))

    fig.add_annotation(
        x=bins[3],
        y=max(percentages) * 0.9,
        text="⬅️ Plus c'est à gauche, plus c'est flou",
        showarrow=False,
        font=dict(size=13, color="gray"),
        align="left",
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="gray",
        borderwidth=1,
    )

    fig.update_layout(
        title="Distribution de la netteté dans le dataset",
        xaxis_title="Variance du Laplacien (netteté)",
        yaxis_title="Pourcentage d'images",
        template="plotly_white"
    )

    mean_sharpness = np.mean(sharpness_values)

    return fig, mean_sharpness


if __name__ == "__main__":
    source_folder = config.DATASET_IMAGE_DIR_TEST
    image_files = get_image_files(source_folder)

    check_image_sizes(source_folder, image_files)
    detect_duplicates(source_folder, image_files)
    plot_color_distribution(source_folder, image_files)
    plot_sharpness_distribution(source_folder, image_files)
