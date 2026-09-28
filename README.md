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
