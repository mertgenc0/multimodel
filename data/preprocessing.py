import os
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


def create_dummy_pisc_dataset(raw_path, num_samples=1000):
    """
    GERÇEK DATASET YOKSA: Dummy dataset oluştur (test için)

    Bu fonksiyon sadece kod test etmek için kullanılır.
    Gerçek PISC dataset'i indirdikten sonra bu fonksiyonu KULLANMAYIN!
    """
    print("🎭 Creating DUMMY dataset for testing...")
    print("⚠️  This is NOT real data! Only for code testing.")

    images_dir = os.path.join(raw_path, 'images')
    annotations_dir = os.path.join(raw_path, 'annotations')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Dummy annotations
    annotations = {}

    relationship_types = {
        'coarse': ['intimate', 'non-intimate', 'no-relation'],
        'fine': ['friends', 'family', 'couple', 'professional', 'commercial', 'no-relation']
    }

    captions = [
        "Two friends laughing together",
        "A couple holding hands",
        "Family members at dinner",
        "Colleagues in a meeting",
        "Two people at a store",
        "Happy friends playing sports"
    ]

    for i in tqdm(range(num_samples), desc="Creating dummy data"):
        img_name = f"dummy_image_{i:05d}.jpg"

        # Create dummy image (red square)
        from PIL import Image
        img = Image.new('RGB', (224, 224), color=(200, 200, 200))
        img.save(os.path.join(images_dir, img_name))

        # Create dummy annotation
        num_pairs = random.randint(1, 3)
        pairs = {}

        for p in range(num_pairs):
            pair_id = f"pair_{p}"
            pairs[pair_id] = {
                'person1_bbox': [10, 10, 100, 200],
                'person2_bbox': [120, 10, 210, 200],
                'coarse_label': random.choice(relationship_types['coarse']),
                'fine_label': random.choice(relationship_types['fine']),
                'caption': random.choice(captions)
            }

        annotations[img_name] = {'pairs': pairs}

    # Save annotations
    with open(os.path.join(annotations_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ Dummy dataset created: {num_samples} images")
    return annotations


def organize_pisc_dataset(raw_path, processed_path, test_size=0.2, val_size=0.1):
    """
    PISC dataset'ini train/val/test olarak organize et

    Args:
        raw_path: data/raw yolu
        processed_path: data/processed yolu
        test_size: Test set oranı (0.2 = %20)
        val_size: Validation set oranı (0.1 = %10)
    """

    print("🔄 Organizing PISC dataset...")
    print("=" * 60)

    images_dir = os.path.join(raw_path, 'images')
    annotations_file = os.path.join(raw_path, 'annotations', 'annotations.json')

    # Check if dataset exists
    if not os.path.exists(annotations_file):
        print("⚠️  Real PISC dataset not found!")
        print("   Creating DUMMY dataset for testing...")
        create_dummy_pisc_dataset(raw_path, num_samples=1000)
        annotations_file = os.path.join(raw_path, 'annotations', 'annotations.json')

    # Load annotations
    print(f"📂 Loading annotations from {annotations_file}")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    print(f"📊 Total images: {len(annotations)}")

    # Get all image names
    all_images = list(annotations.keys())

    # Train/Val/Test split (80/10/10)
    train_imgs, temp_imgs = train_test_split(
        all_images,
        test_size=test_size,
        random_state=42
    )

    val_ratio = val_size / test_size  # 0.1 / 0.2 = 0.5
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,  # Split temp equally
        random_state=42
    )

    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    print(f"\n📊 Split Statistics:")
    print(f"   Train: {len(train_imgs)} images ({len(train_imgs) / len(all_images) * 100:.1f}%)")
    print(f"   Val:   {len(val_imgs)} images ({len(val_imgs) / len(all_images) * 100:.1f}%)")
    print(f"   Test:  {len(test_imgs)} images ({len(test_imgs) / len(all_images) * 100:.1f}%)")

    # Organize each split
    for split_name, img_list in splits.items():
        print(f"\n🔨 Processing {split_name} split...")

        split_path = os.path.join(processed_path, split_name)
        os.makedirs(split_path, exist_ok=True)

        split_annotations = {}

        # Copy images and collect annotations
        for img_name in tqdm(img_list, desc=f"Copying {split_name} images"):
            # Source and destination paths
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(split_path, img_name)

            # Copy image
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"⚠️  Image not found: {src_img}")

            # Store annotation
            split_annotations[img_name] = annotations[img_name]

        # Save split annotations
        annotations_path = os.path.join(split_path, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(split_annotations, f, indent=2)

        print(f"✅ {split_name}: {len(img_list)} images saved")

    print("\n" + "=" * 60)
    print("✅ Dataset organization complete!")
    print("\n📁 Dataset structure:")
    print(f"   {processed_path}/")
    print(f"   ├── train/ ({len(train_imgs)} images)")
    print(f"   ├── val/   ({len(val_imgs)} images)")
    print(f"   └── test/  ({len(test_imgs)} images)")
    print("=" * 60)


def verify_dataset(processed_path):
    """Dataset'in doğru şekilde organize edildiğini kontrol et"""

    print("\n🔍 Verifying dataset...")
    print("=" * 60)

    splits = ['train', 'val', 'test']
    all_good = True

    for split in splits:
        split_path = os.path.join(processed_path, split)
        annotations_path = os.path.join(split_path, 'annotations.json')

        # Check if annotations exist
        if not os.path.exists(annotations_path):
            print(f"❌ {split}: annotations.json not found")
            all_good = False
            continue

        # Load annotations
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Count pairs
        total_pairs = sum(
            len(img_data.get('pairs', {}))
            for img_data in annotations.values()
        )

        # Count images
        num_images = len([f for f in os.listdir(split_path) if f.endswith('.jpg')])

        print(f"✅ {split:5s}: {num_images:4d} images, {total_pairs:5d} pairs")

    print("=" * 60)

    if all_good:
        print("✅ Dataset verification passed!")
    else:
        print("⚠️  Some issues found. Please check above.")

    return all_good


if __name__ == "__main__":
    import sys

    # Paths
    raw_path = "raw"
    processed_path = "processed"

    print("🚀 PISC Dataset Preprocessing")
    print("=" * 60)

    # Check if raw data exists
    if not os.path.exists(os.path.join(raw_path, 'images')):
        print("⚠️  Raw data not found!")
        print("\n📥 Please either:")
        print("   1. Download PISC dataset and place in data/raw/")
        print("   2. Or let me create DUMMY data for testing (type 'y')")

        choice = input("\nCreate dummy data? (y/n): ").lower()

        if choice == 'y':
            os.makedirs(raw_path, exist_ok=True)
            create_dummy_pisc_dataset(raw_path, num_samples=1000)
        else:
            print("❌ Exiting. Please download dataset first.")
            sys.exit(1)

    # Organize dataset
    organize_pisc_dataset(raw_path, processed_path)

    # Verify
    verify_dataset(processed_path)

    print("\n🎉 All done! Dataset is ready for training.")