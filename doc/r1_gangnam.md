# R1 Gangnam: kısa referans ve tracking görevi

Bu değişiklik `Unitree-R1-Tracking` ve
`Unitree-R1-Tracking-No-State-Estimation` görevlerini ekler. Mevcut R1 robot/motor
ayarları ve tracking PPO altyapısı kullanılır; `train.py` ve `play.py` değişmez.

Depoda kaynak G1 CSV'si, dönüştürülmüş R1 CSV/NPZ dosyaları ve eğitilmiş
`model_3500.pt` checkpoint'i birlikte bulunur.

## 1. Ortamı aç

```bash
conda activate unitree_rl_mjlab
cd unitree_rl_mjlab
```

## 2. İlk üç saniyeyi dönüştür

```bash
python scripts/r1_gangnam.py convert --start 5.2 --duration 22.0 --overwrite
```

Varsayılan girdi `src/assets/motions/r1/gangnam_g1_source.csv` dosyasıdır.
Başka dosya için `--input /tam/yol/motion.csv` ver.

Çıktılar:

- `src/assets/motions/r1/gangnam_short.csv`: 60 Hz, 1321 kare; gövde xyz [m],
  quaternion xyzw, 24 eklem açısı [rad].
- `src/assets/motions/r1/gangnam_short.json`: kaynak özeti, eklem sırası,
  kırpılan açı sayısı, bel uyarlama hatası ve yükseklik düzeltme aralığı.

Mevcut çıktıyı değiştirmek için komuta `--overwrite` ekle.

Dönüştürücü isimle eşleme yapar, R1 sınırlarına uymayan açıları kırpar,
G1'in gövde yönelimini R1'in iki bel eklemiyle yaklaşık eşler. Her karede R1'in
en alttaki ayak tabanı yüksekliğini G1'inkiyle eşleştirir; kaynakta iki ayak
havadaysa yerden açıklığı korur. Negatif kaynak açıklığı sıfıra kırpılır.
El ve ayakların yatay konumları için ters kinematik uygulanmaz.

## 3. Referansı izle

```bash
python scripts/r1_gangnam.py preview
```

Yavaş izlemek için:

```bash
python scripts/r1_gangnam.py preview --speed 0.5
```

Pencereyi kapatarak çık. Bu önizleme MuJoCo'dadır. Gövde ve eklemler doğrudan
referans konumlara yazılır; politika ve fizik adımlaması yoktur.
**Ayakta görünmesi dengeyi öğrendiği anlamına gelmez.**

Ayak kaymasını, kolların gövdeye girmesini, hareket sıçramalarını ve belin
uygunluğunu kontrol et. Bu ilk uyarlama ayak temasını yatayda kilitlemez,
öz-çarpışmaları çözmez ve hareketin dinamik olarak yapılabilirliğini doğrulamaz.
Sorun varsa uzun eğitimden önce referans düzeltilmelidir. JSON'daki hata
değerleri bir başarı/onay ölçütü değildir.

## 4. Önizlemeyi değerlendirdikten sonra NPZ oluştur

```bash
python scripts/csv_to_npz.py --robot r1 \
  --input-file src/assets/motions/r1/gangnam_short.csv \
  --output-name gangnam_short.npz --input-fps 60 --output-fps 50 --device cpu
```

Mevcut dönüştürücü R1 modelinin gövde konumları/hızlarını ve eklem sırasını
kullanır. Çıktı `src/assets/motions/r1/gangnam_short.npz` olur. G1 NPZ veya
ONNX dosyasını R1 için kullanma.

## Eğitim ve politika oynatma

```bash
bash scripts/train_r1_gangnam.sh
```

Varsayılan 256 ortamdır. Örneğin 512 ortam için
`R1_NUM_ENVS=512 bash scripts/train_r1_gangnam.sh` kullan. Checkpoint'ler
`logs/rsl_rl/r1_tracking/` altına kaydedilir. Varsayılan tracking eğitimindeki
gürültü ve dış itmeler korunur; kısa bölümün öğrenilmesine göre ayarlanabilir.

```bash
python scripts/play.py Unitree-R1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/r1/gangnam_short.npz \
  --checkpoint-file artifacts/r1_gangnam/checkpoints/model_3500.pt \
  --num-envs 1 --viewer native
```

Bu son komut öğrenilmiş politikayı fizik içinde çalıştırır; `preview` komutundan farklıdır.
Modelin değişmesi durumunda CSV ve NPZ'yi yeniden üret.

## Kaynak ve sınırlar

Gangnam kaydı, Unitree RL Lab
`deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/params/G1_gangnam_style_V01.bvh_60hz.csv`
dosyasından alınmıştır. Bu G1 eklem hareketidir; özgün insan BVH kaydı değildir.

R1 tracking yükleyicisi kaynak XML'i değiştirmeden, mevcut olmayan bilek
gövdelerine ait eski contact exclude kayıtlarını bellek içinde ayıklar.
R1 yürüyüş görevlerinin kaynakları değiştirilmez.
