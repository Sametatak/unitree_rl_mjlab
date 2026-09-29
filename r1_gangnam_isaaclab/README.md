# R1 Gangnam policy — Isaac Lab

Eğitilmiş Unitree R1 Gangnam policy'sini Isaac Sim'de fizik ile oynatır.
Kod, ONNX policy, dans referansı ve robotun bütün USD katmanları bu klasördedir.
Eğitim veya mjlab kurulumu gerekmez.

## Gereken kurulum

- Linux ve çalışan **Isaac Lab 3 / Isaac Sim 6** kurulumu.
- Isaac Lab'in kendi Python ortamındaki PyTorch ve aşağıdaki ek paketler.
- Grafik arayüz için çalışan NVIDIA sürücüsü ve Isaac Sim uyumlu GPU.

Bu kod yerel Isaac Lab 3 API'sini kullanır (`.torch`, `*_to_sim_index`, xyzw
quaternion). Isaac Lab 2.x ile doğrudan uyumlu değildir. Isaac Lab/Sim'in kendisi
ve GPU sürücüsü bu arşive dahil değildir; ayrı kurulmalıdır.

## Çalıştırma

Klasörü istediğin yere indir/aç. Varsayılan Isaac Lab yolu `$HOME/IsaacLab`.

```bash
cd r1_gangnam_isaaclab
bash run.sh --device cpu
```

Isaac Lab başka bir klasördeyse:

```bash
ISAACLAB_PYTHON=/kurulum/.venv/bin/python bash run.sh --device cpu
```

Eksik Python paketi hatası alırsan, Isaac Lab'in Python ortamına kur:

```bash
bash "${ISAACLAB_PATH:-$HOME/IsaacLab}/isaaclab.sh" -p -m pip install -r requirements.txt
```

### Seçenekler

```bash
bash run.sh --device cpu --once        # Bir dans sonunda duraklat
bash run.sh --device cuda:0            # GPU fiziği
bash run.sh --device cpu --headless --once  # Arayüzsüz tek tur
```

Varsayılan olarak dans bitince robot başlangıç pozu ve hızlarına dönüp tekrar
eder. Düşmeye bağlı otomatik reset yoktur. `--once` sonrası arayüzde Play yeni
turu başlatır. Küple müdahale/GUI kuvvet araçları için CPU fiziğini kullan;
GPU fiziğinde bazı araçlar Direct GPU API kısıtına takılabilir.

## Paket içeriği

```text
play.py                         Isaac Lab çalıştırıcısı
run.sh                          Başlatma komutu
requirements.txt                Ek Python bağımlılıkları
assets/model_3500/policy.onnx     Öğrenilmiş actor + gözlem normalizasyonu
assets/model_3500/motion.npz      R1'e uyarlanmış dans referansı
assets/model_3500/config.json     Eklem sırası ve motor/kontrol ayarları
assets/robot/r1/r1.usda           Robotun giriş USD dosyası
assets/robot/r1/payloads/         Geometri ve fizik katmanları
SHA256SUMS                      Paket dosyalarının bütünlük değerleri
THIRD_PARTY_NOTICES.md           Kaynak bilgileri
```

Policy kaynağı: `r1_tracking/2026-09-15_17-26-16/model_3500.pt`.
24 eklem, 129 gözlem; fizik 200 Hz, policy 50 Hz. `.pt` dosyası çıkarım için
gerekmez: dışa aktarılmış ONNX pakete dahil edilmiştir. Robot ve model yolları
klasöre göredir; eski `/home/forkon/...` dizinlerine ihtiyaç duyulmaz.

`--bundle /yol/model_klasoru` ile aynı gözlem/eklem yapısındaki başka bir
dışa aktarımı, `--robot_usd /yol/r1.usda` ile eşdeğer robot USD'sini seçebilirsin.

MuJoCo'da öğrenilen policy PhysX'te çalışır; temas ve çözücü farkları nedeniyle
denge birebir aynı olmayabilir. Paketleme sırasında simülasyon veya eğitim
çalıştırılmadı. Bu depo eğitim görevi değil, eğitilmiş policy çalıştırıcısıdır.
