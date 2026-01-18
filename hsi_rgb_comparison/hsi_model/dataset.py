import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not found. Multispectral/Hyperspectral data might not load correctly.")

class HSIDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_dict=None, transform=None, selected_bands=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            label_dict (dict): Dictionary mapping class names to integers.
            transform (callable, optional): Optional transform to be applied on a sample.
            selected_bands (list, optional): List of band indices to select (0-based).
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_dict = label_dict
        self.transform = transform
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # CSV'den resim yolunu al
        img_name = self.annotations.iloc[idx, 0] # image_path column
        
        # Dosya yolunu oluştur
        img_path = os.path.join(self.root_dir, img_name)

        # Hata ayıklama: Dosya var mı kontrol et
        if not os.path.exists(img_path):
             # Alternatif: dosya ismine _hyperspectral ekleyip dene
             name_without_ext = os.path.splitext(img_name)[0]
             ext = os.path.splitext(img_name)[1]
             # Check if it already has _hyperspectral to avoid doubling it if logic fails elsewhere
             if "_hyperspectral" not in name_without_ext:
                 alt_name = f"{name_without_ext}_hyperspectral{ext}"
                 alt_path = os.path.join(self.root_dir, alt_name)
                 
                 if os.path.exists(alt_path):
                     img_path = alt_path
                 else:
                    raise FileNotFoundError(f"Image not found at: {img_path} (checked also {alt_path}). Check --root_train/val arguments.")
             else:
                 raise FileNotFoundError(f"Image not found at: {img_path}. Check --root_train/val arguments.")

        # --- GÖRÜNTÜ OKUMA KISMI ---
        image = None
        
        # Multispectral/Hyperspectral okuma (Rasterio varsa ve TIF ise)
        if HAS_RASTERIO and (img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff')):
            try:
                with rasterio.open(img_path) as src:
                    # Spesifik bantlar seçildiyse onları oku
                    if self.selected_bands:
                        # Rasterio 1-based index kullanır, listemiz 0-based ise +1 eklemeliyiz
                        # Ancak src.read() listeyi olduğu gibi alırsa:
                        # indices = [x + 1 for x in self.selected_bands]
                        # image = src.read(indices)
                        
                        # Daha güvenli yöntem: Tümünü oku, sonra seç
                        full_image = src.read() # (Channels, Height, Width)
                        image = full_image[self.selected_bands, :, :]
                    else:
                        image = src.read() # (C, H, W)
                        
                # Float32'ye çevir ve normalize et (Opsiyonel: Veri tipine göre değişir)
                image = image.astype(np.float32)
                image = torch.from_numpy(image)
                
            except Exception as e:
                print(f"Rasterio read failed for {img_path}, falling back to PIL. Error: {e}")
        
        # Eğer yukarıda okunmadıysa (JPG/PNG veya rasterio yoksa) PIL ile RGB oku
        if image is None:
            pil_image = Image.open(img_path).convert("RGB")
            image = np.array(pil_image)
            image = image.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W) yap
            image = torch.from_numpy(image).float()
            
            # Eğer 10 bant istendiyse ama RGB (3 bant) okunduysa hata verebiliriz
            # veya duplicate edebiliriz. Şimdilik olduğu gibi bırakıyoruz.

        # --- KOORDİNAT OKUMA ---
        # csv yapısı: image_path, xmin, ymin, xmax, ymax, label
        xmin = float(self.annotations.iloc[idx, 1])
        ymin = float(self.annotations.iloc[idx, 2])
        xmax = float(self.annotations.iloc[idx, 3])
        ymax = float(self.annotations.iloc[idx, 4])
        class_name = str(self.annotations.iloc[idx, 5])
        
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        
        # Label mapping
        label_id = 1 # Varsayılan: Tek sınıf varsa 1
        if self.label_dict and class_name in self.label_dict:
            label_id = self.label_dict[class_name]
            
        labels = torch.tensor([label_id], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transform:
            # Not: Transform uygulanacaksa HSI uyumlu olmalı
            pass

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))