import pandas as pd
import rasterio
import spectral.io.envi as envi
from spectral import *
import numpy as np
import time

#programın başından sonuna kadar geçen süreyi ölçmek için başlangıç zamanı
start_time = time.time()

# CSV dosyanı oku 
df = pd.read_csv('neon_aop_bands.csv')
wavelengths = df['nanometer'].tolist() 

# Eğer bant isimleri de varsa onları da alabilirsin
band_names = df['BandName'].tolist() if 'BandName' in df.columns else None

# 1. TIF dosyasını oku
with rasterio.open('hsi_examples\\BONA_017_2019_hyperspectral.tif') as src:
    # Rasterio (Bands, Rows, Samples) okur, 
    # Spectral ise (Rows, Samples, Bands) bekler. Transpose şart.
    img_data = src.read().transpose(1, 2, 0)
    
    # Veri tipini kontrol et (int16, float32 vb.)
    dtype = img_data.dtype

# 2. ENVI Header (Üstbilgi) sözlüğünü oluştur
metadata = {
    'wavelength': wavelengths,
    'description': 'TIF dosyasindan donusturuldu',
    'interleave': 'bil' # Genelde tercih edilen formattır
}

if band_names:
    metadata['band names'] = band_names

# 3. Dosyayı kaydet (Bu işlem bir .hdr bir de .dat/raw dosyası oluşturur)
envi.save_image('BONA_017_2019_hyperspectral.hdr', img_data, metadata=metadata, force=True)

# Programın toplam çalışma süresini hesapla
end_time = time.time()
print(f"Dosya başarıyla kaydedildi. Toplam süre: {end_time - start_time:.2f} saniye.")

time.sleep(2)  # Kullanıcıya mesajı görmesi için zaman tanıma