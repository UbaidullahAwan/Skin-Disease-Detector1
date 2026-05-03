#!/usr/bin/env python3
"""
Train a skin disease classifier from your dataset
Run this first: python3 train_model.py
"""

import numpy as np
from PIL import Image
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys


def extract_detailed_features(image_path):
    """
    Extract comprehensive features from a skin image
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0

        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]

        features = []

        # Color features
        features.append(np.mean(r_channel))
        features.append(np.mean(g_channel))
        features.append(np.mean(b_channel))
        features.append(np.std(r_channel))
        features.append(np.std(g_channel))
        features.append(np.std(b_channel))

        # Redness (r - g)
        redness = np.mean(r_channel - g_channel)
        features.append(max(0, redness))

        # Brightness
        brightness = (np.mean(r_channel) + np.mean(g_channel) +
                      np.mean(b_channel)) / 3
        features.append(brightness)

        # Color variation
        features.append(np.std(img_array))

        # Saturation
        max_rgb = np.maximum(np.maximum(r_channel, g_channel), b_channel)
        min_rgb = np.minimum(np.minimum(r_channel, g_channel), b_channel)
        saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 0.001))
        features.append(saturation)

        # Texture features (edge detection)
        gray = np.mean(img_array, axis=2)
        h_edges = np.abs(np.diff(gray, axis=1))
        v_edges = np.abs(np.diff(gray, axis=0))

        if h_edges.size > 0:
            features.append(np.mean(h_edges))
        else:
            features.append(0)

        if v_edges.size > 0:
            features.append(np.mean(v_edges))
        else:
            features.append(0)

        total_edges = (np.mean(h_edges) if h_edges.size > 0 else 0) + \
                      (np.mean(v_edges) if v_edges.size > 0 else 0)
        features.append(total_edges / 2)

        # Asymmetry (split into quadrants)
        h, w = gray.shape
        if h >= 2 and w >= 2:
            q1 = gray[:h//2, :w//2].mean()
            q2 = gray[:h//2, w//2:].mean()
            q3 = gray[h//2:, :w//2].mean()
            q4 = gray[h//2:, w//2:].mean()

            asymmetry_h = abs(q1 - q2) / (q1 + q2 + 0.001)
            asymmetry_v = abs(q3 - q4) / (q3 + q4 + 0.001)
            features.append(asymmetry_h)
            features.append(asymmetry_v)
            features.append((asymmetry_h + asymmetry_v) / 2)
        else:
            features.extend([0, 0, 0])

        # Color histograms
        hist_r, _ = np.histogram(r_channel, bins=10, range=(0, 1))
        hist_g, _ = np.histogram(g_channel, bins=10, range=(0, 1))
        hist_b, _ = np.histogram(b_channel, bins=10, range=(0, 1))

        features.extend(hist_r / len(r_channel.flatten()))
        features.extend(hist_g / len(g_channel.flatten()))
        features.extend(hist_b / len(b_channel.flatten()))

        # Texture uniformity
        features.append(np.std(gray))
        features.append(np.mean(gray) - np.std(gray))

        return features

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_dataset(dataset_path):
    """Load all images from dataset folders"""
    X = []
    y = []

    print(f"\n📂 Loading dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        return None, None

    disease_folders = [f for f in os.listdir(dataset_path)
                       if os.path.isdir(os.path.join(dataset_path, f))]

    if not disease_folders:
        print(f"❌ No disease folders found in '{dataset_path}'")
        return None, None

    print(f"\nFound {len(disease_folders)} disease categories:")

    total_images = 0
    for disease in disease_folders:
        disease_path = os.path.join(dataset_path, disease)
        image_files = [f for f in os.listdir(disease_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        print(f"  {disease}: {len(image_files)} images")
        total_images += len(image_files)

        for img_file in image_files:
            img_path = os.path.join(disease_path, img_file)
            features = extract_detailed_features(img_path)

            if features:
                X.append(features)
                y.append(disease)

    if len(X) == 0:
        print("\n❌ No valid images found!")
        return None, None

    print(
        f"\n✅ Successfully loaded {len(X)} images out of {total_images} total")
    return np.array(X), np.array(y)


def train_model(X, y):
    """Train the Random Forest classifier"""
    print("\n🚀 Training Random Forest classifier...")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\n📊 Validation Accuracy: {accuracy:.2%}")
    print("\n📈 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_))

    return clf, encoder


def save_model(clf, encoder):
    """Save the trained model"""
    with open('skin_disease_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("\n✅ Model saved as 'skin_disease_model.pkl'")
    print("✅ Label encoder saved as 'label_encoder.pkl'")


def main():
    print("=" * 60)
    print("🧠 SKIN DISEASE CLASSIFIER TRAINER")
    print("=" * 60)

    # Get dataset path
    dataset_path = input(
        "\n📁 Enter dataset path (default: 'dataset'): ").strip()
    if not dataset_path:
        dataset_path = "dataset"

    # Load dataset
    X, y = load_dataset(dataset_path)

    if X is None:
        print("\n❌ Training failed! Please check your dataset folder.")
        print("\nExpected structure:")
        print("  dataset/")
        print("    Acne_Vulgaris/")
        print("    Psoriasis/")
        print("    Melanoma/")
        print("    Impetigo/")
        print("    Eczema_Atopic_Dermatitis/")
        print("    Ringworm_Tinea_Corporis/")
        print("    Rosacea/")
        print("    Basal_Cell_Carcinoma/")
        print("    Vitiligo/")
        return

    # Train model
    clf, encoder = train_model(X, y)

    # Save model
    save_model(clf, encoder)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(
        f"\n✅ Model trained on {len(encoder.classes_)} diseases: {list(encoder.classes_)}")
    print("\nNow run: streamlit run app.py")


if __name__ == "__main__":
    main()
