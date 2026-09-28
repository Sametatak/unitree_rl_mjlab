# Unitree G1 Human Motion Imitation Pipeline

Bu proje, tek kameralı bir MP4 videodaki insan hareketlerini Unitree G1 robotuna
aktarmak ve hareket takip politikası eğitmek için uçtan uca bir pipeline sunar.

```text
MP4 video
  → GVHMR ile dünya koordinatlarında 3B insan hareketi
  → GMR ile 29-DoF Unitree G1 retargeting
  → G1 CSV ve NPZ hareket verisi
  → MuJoCo üzerinde reinforcement learning eğitimi
```

## Gereksinimler

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU
- [GVHMR](https://github.com/ryanrudes/gvhmr)
- [GMR](https://github.com/YanjieZe/GMR)
- Kişisel olarak indirilmiş SMPL-X gövde modelleri

Pipeline varsayılan olarak GVHMR'yi `~/GVHMR-blackwell`, GMR'yi ise `~/GMR`
altında arar. İki proje de kendi Python ortamında çalışmaya devam eder; büyük
model ağırlıkları bu repoya kopyalanmaz.

## Ana proje kurulumu

```bash
git clone https://github.com/Sametatak/unitree_rl_mjlab.git
cd unitree_rl_mjlab

conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab

sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
pip install -e .
```

Projede hazır `.venv` bulunuyorsa bunun yerine:

```bash
source .venv/bin/activate
```

GMR'nin `assets/body_models/smplx/` dizininde gerekli SMPL-X model dosyalarının
bulunduğundan emin olun.

## Hazır eğitilmiş dosyalar: G1 halay

Repo, doğrudan denenebilen hazır bir G1 halay politikası içerir:

- Hareket: `src/assets/motions/g1/halay_loop.npz`
- Eğitim checkpoint'i: `artifacts/g1_halay/checkpoints/model_5000.pt`
- ONNX politikası: `artifacts/g1_halay/exported/policy.onnx`
- Eğitim ayarları: `artifacts/g1_halay/params/`

Hazır politikayı MuJoCo simülasyonunda çalıştırmak için:

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/halay_loop.npz \
  --checkpoint-file artifacts/g1_halay/checkpoints/model_5000.pt
```

Video kaydetmek için:

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/halay_loop.npz \
  --checkpoint-file artifacts/g1_halay/checkpoints/model_5000.pt \
  --video True \
  --video-length 500
```

## Videodan motion üretme ve eğitimi başlatma

Sabit kamerayla çekilmiş, tek kişinin net göründüğü bir video için:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/human_motion.mp4 \
  --motion-name human_motion_loop \
  --gvhmr-root ~/GVHMR-blackwell \
  --gmr-root ~/GMR
```

Pipeline aşağıdaki işlemleri otomatik gerçekleştirir:

1. GVHMR ile videodan SMPL-X hareketi çıkarır.
2. GMR ile insan hareketini Unitree G1 eklem hareketlerine dönüştürür.
3. Döngüye uygun başlangıç ve bitiş karelerini otomatik seçer.
4. XY sürüklenmesini kaldırır ve quaternion/joint crossfade uygular.
5. `src/assets/motions/g1/human_motion_loop.csv` üretir.
6. Hareketi 50 FPS NPZ biçimine dönüştürür.
7. `Unitree-G1-Tracking-No-State-Estimation` eğitimini başlatır.

Eğitimi başlatmadan yalnızca CSV ve NPZ üretmek için:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/human_motion.mp4 \
  --motion-name human_motion_loop \
  --prepare-only
```

Döngü oluşturulmaması gereken bir hareket için:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/human_motion.mp4 \
  --motion-name human_motion \
  --no-loop
```

Hareketli kamera kullanıldıysa:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/human_motion.mp4 \
  --motion-name human_motion_loop \
  --moving-camera \
  --gvhmr-camera simplevo
```

Önceden oluşturulmuş bir GVHMR sonucu kullanarak video analizini atlamak için:

```bash
python scripts/video_to_g1_pipeline.py ~/Videos/human_motion.mp4 \
  --motion-name human_motion_loop \
  --gvhmr-result /path/to/hmr4d_results.pt
```

Pipeline ara dosyalarıyla birlikte kullanılan GVHMR, GMR ve ana repo Git
commit'lerini `pipeline_runs/<motion-name>/pipeline_run.json` dosyasına kaydeder.
Bu çalışma dizini Git'e eklenmez.

## Eğitilmiş hareketi simülasyonda çalıştırma

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/human_motion_loop.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_<iteration>.pt
```

Ekransız veya uzak bir makinede web görüntüleyicisini kullanmak için komuta
`--viewer viser` ekleyin. Belirli bir GPU seçmek için `--device cuda:0`
kullanabilirsiniz.

Video kaydetmek için:

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/human_motion_loop.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_<iteration>.pt \
  --video True \
  --video-length 500
```

## Lisans ve güvenlik

GVHMR'nin lisansı eğitim, araştırma ve ticari olmayan kullanımla sınırlıdır.
SMPL-X gövde modelleri ayrıca kayıt ve lisans kabulü gerektirir. Bu repo yalnızca
orkestrasyon kodunu içerir; üçüncü taraf ağırlıkları ve gövde modellerini yeniden
dağıtmaz.

Gerçek robota aktarmadan önce politikayı simülasyonda doğrulayın. Fiziksel robot
çalıştırılırken güvenli askı, acil durdurma ve uygun çevre önlemleri kullanın.
