# Unitree G1 Halay

Bu repo, Unitree G1 robotu için eğitilmiş halay hareketini MuJoCo
simülasyonunda çalıştırmak için gerekli hareket verisini ve modeli içerir.

## Hazır dosyalar

- Hareket: `src/assets/motions/g1/halay_loop.npz`
- Eğitim checkpoint'i: `artifacts/g1_halay/checkpoints/model_5000.pt`
- Dağıtım modeli: `artifacts/g1_halay/exported/policy.onnx`
- Eğitim ayarları: `artifacts/g1_halay/params/`

## Kurulum

Ubuntu 22.04, Python 3.11 ve NVIDIA GPU önerilir.

```bash
git clone https://github.com/Sametatak/unitree_rl_mjlab.git
cd unitree_rl_mjlab

conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab

sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
pip install -e .
```

Projede daha önce oluşturulmuş `.venv` kullanılıyorsa:

```bash
source .venv/bin/activate
```

## MP4 videodan eğitim başlatma

Repo, aşağıdaki uçtan uca akışı otomatikleştirir:

`MP4 → GVHMR → SMPL-X → GMR → G1 CSV → loop düzeltme → NPZ → eğitim`

Pipeline, varsayılan olarak şu iki harici projeyi kullanır:

- [GVHMR](https://github.com/ryanrudes/gvhmr): `~/GVHMR-blackwell`
- [GMR](https://github.com/YanjieZe/GMR): `~/GMR`

Her iki projeyi kendi kurulum talimatlarıyla hazırlayın. GMR'nin
`assets/body_models/smplx/` dizininde kişisel olarak indirdiğiniz SMPL-X gövde
modelleri bulunmalıdır. Model dosyaları ve ağırlıklar bu repoya eklenmez.

Sabit kamerayla çekilmiş, tek kişinin net göründüğü bir halay videosundan motion
üretip eğitimi doğrudan başlatmak için:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/halay.mp4 \
  --motion-name halay_loop \
  --gvhmr-root ~/GVHMR-blackwell \
  --gmr-root ~/GMR
```

Bu komut sırasıyla:

1. GVHMR ile videodan dünya koordinatlarında SMPL-X hareketi çıkarır.
2. GMR ile hareketi 29-DoF Unitree G1 eklemlerine retarget eder.
3. Birbirine en yakın başlangıç/bitiş pozlarını otomatik bulur, XY sürüklenmesini
   kaldırır ve 15 karelik quaternion/joint crossfade uygular.
4. `src/assets/motions/g1/halay_loop.csv` ve `halay_loop.npz` üretir.
5. `Unitree-G1-Tracking-No-State-Estimation` eğitimini 4096 ortamla başlatır.

Yalnızca motion dosyasını üretmek, eğitimi başlatmamak için:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/halay.mp4 \
  --motion-name halay_loop \
  --prepare-only
```

Hareketli kamera kullanıldıysa örneğin:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/halay.mp4 \
  --motion-name halay_loop \
  --moving-camera \
  --gvhmr-camera simplevo
```

Daha önce GVHMR sonucu üretildiyse pahalı video analizini atlayabilirsiniz:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/halay.mp4 \
  --motion-name halay_loop \
  --gvhmr-result ~/GVHMR-blackwell/outputs/demo/halay/hmr4d_results.pt
```

Ara dosyalar ve kullanılan GVHMR/GMR Git commit'leri
`pipeline_runs/<motion-name>/pipeline_run.json` içinde kaydedilir. Bu dizin Git'e
eklenmez.

> GVHMR'nin lisansı eğitim, araştırma ve ticari olmayan kullanımla sınırlıdır.
> SMPL-X gövde modelleri ayrıca kayıt/lisans gerektirir. Bu repo yalnızca
> orkestrasyon kodunu içerir; üçüncü taraf kodu, ağırlıkları ve gövde modellerini
> yeniden dağıtmaz.

## Halayı çalıştırma

Repo ana dizinindeyken aşağıdaki komutu çalıştırın:

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/halay_loop.npz \
  --checkpoint-file artifacts/g1_halay/checkpoints/model_5000.pt
```

Pencereli bir masaüstünde MuJoCo görüntüleyicisi otomatik açılır. Ekransız veya
uzak bir makinede web görüntüleyicisini kullanmak için komuta şunu ekleyin:

```bash
--viewer viser
```

Belirli bir GPU kullanmak için:

```bash
--device cuda:0
```

## Video kaydetme

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/halay_loop.npz \
  --checkpoint-file artifacts/g1_halay/checkpoints/model_5000.pt \
  --video True \
  --video-length 500
```

Video, checkpoint klasörünün altındaki `videos/play/` dizinine kaydedilir.

> Gerçek robota aktarmadan önce modeli simülasyonda doğrulayın. Fiziksel robot
> çalıştırılırken güvenli askı, acil durdurma ve uygun çevre önlemleri kullanın.
