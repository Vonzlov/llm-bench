# k3s с GPU на одной машине

Цель — чтобы под в кластере видел видеокарту. Инструкция проверена на Windows 10 с WSL2
(Ubuntu 24.04), RTX 3060, k3s v1.36.4+k3s1 и NVIDIA Container Toolkit 1.20.1. На обычном Linux
шаги те же, пропускаются только замечания про WSL.

Все настройки узла лежат в [`deploy/node`](../deploy/node), команды выполняются из корня
репозитория. Длинных команд с переносами через `\` здесь нет намеренно: при вставке
в терминал они ломаются.

## Что нужно

- NVIDIA GPU с драйвером: `nvidia-smi` на хосте работает. В WSL2 драйвер ставится в Windows,
  а внутри WSL — нет.
- В WSL включён systemd: в `/etc/wsl.conf` есть секция `[boot]` со строкой `systemd=true`.
- В WSL на Windows 10 включённый VPN может отключить сеть. Если имена не резолвятся или
  `curl` висит, первым делом выключите VPN.

## 1. NVIDIA Container Toolkit

Прослойка, через которую контейнеры получают доступ к видеокарте. Репозиторий тот же, что
в [официальной инструкции NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
но записан в формате deb822 — по полю на строку.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o /tmp/nvidia.key
sudo gpg --dearmor -o /usr/share/keyrings/nvidia.gpg /tmp/nvidia.key
sudo cp deploy/node/nvidia.sources /etc/apt/sources.list.d/
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk --version
```

Настраивать containerd руками не нужно: k3s сделает это сам при запуске.

## 2. Настройки и образы k3s до установки

k3s читает настройки при старте, поэтому они кладутся заранее.

```bash
sudo mkdir -p /etc/rancher/k3s /var/lib/rancher/k3s/agent/images
sudo cp deploy/node/k3s-config.yaml /etc/rancher/k3s/config.yaml
sudo cp deploy/node/registries.yaml /etc/rancher/k3s/registries.yaml
```

`config.yaml` открывает kubeconfig на чтение без sudo. `registries.yaml` отправляет образы
Docker Hub через зеркало Google: из России Docker Hub отдаёт описание образа, но обрывает
скачивание слоёв. Подробности — в [ADR 0002](adr/0002-image-sources.md).

Системные образы k3s (DNS, провижнер томов, метрики, traefik) тоже лежат на Docker Hub. Чтобы
не зависеть от того, какие из них есть в зеркале, берём их из архива для машин без интернета —
он лежит в релизе k3s на GitHub:

```bash
R=https://github.com/k3s-io/k3s/releases/download/v1.36.4%2Bk3s1
curl -fL -o /tmp/k3s-images.tar.zst $R/k3s-airgap-images-amd64.tar.zst
sudo mv /tmp/k3s-images.tar.zst /var/lib/rancher/k3s/agent/images/
```

Архив подходит только к своей версии k3s. При обновлении k3s его нужно скачать заново.

## 3. k3s

Версия зафиксирована, чтобы совпадать с архивом из шага 2:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.36.4+k3s1 sh -
```

Проверки:

```bash
sudo grep -c nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
kubectl get nodes
kubectl get runtimeclass nvidia
kubectl get pods -A
```

`grep` должен вывести число больше нуля: k3s нашёл рантайм NVIDIA и прописал его в containerd.
Узел — `Ready`, RuntimeClass `nvidia` существует, все поды в `kube-system` — `Running`.

Копия kubeconfig для инструментов, которые ищут его в домашней папке, например helm:

```bash
mkdir -p ~/.kube
cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
```

По метке `gpu=true` серверы моделей находят узел с видеокартой. Имя узла — из вывода
`kubectl get nodes`:

```bash
kubectl label node <имя-узла> gpu=true
```

## 4. Smoke-тест

```bash
kubectl apply -f deploy/smoke/gpu-smoke.yaml
kubectl logs -f pod/gpu-smoke
kubectl delete -f deploy/smoke/gpu-smoke.yaml
```

В логах должна быть таблица `nvidia-smi` с вашей картой. Образ в тесте — обычный `ubuntu:24.04`:
`nvidia-smi` и библиотеки драйвера прокидывает в контейнер рантайм `nvidia`, и проверяется
именно это.

## GPU как ресурс кластера — на этапе 3

Чтобы планировщик Kubernetes считал видеокарту ресурсом `nvidia.com/gpu` и выдавал её подам
по запросу, нужен [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin). Его образ
публикуется только в `nvcr.io`, а тот из России отвечает `403 Forbidden`. На Docker Hub версии
плагина не обновляются с `v0.5.0`.

План из [ADR 0002](adr/0002-image-sources.md): на этапе 3 CI копирует официальный образ в реестр
проекта на GitHub, и узел берёт его оттуда. До этого поды получают карту через
`runtimeClassName: nvidia`, без запроса ресурса. На одной машине с одной видеокартой и
собственной очередью прогонов этого достаточно. Если плагин не заведётся под WSL, запасной путь —
объявить видеокарту узлу вручную как extended resource; манифесты от этого не изменятся.

## Если не заработало

| Симптом | Что сделать |
| --- | --- |
| Поды в `ImagePullBackOff`, в событиях `connection reset by peer` на `production.cloudfront.docker.com` | Docker Hub не отдаёт слои. Нужны `registries.yaml` и архив образов из шага 2, затем `sudo systemctl restart k3s` |
| `kubectl`: `permission denied` на `/etc/rancher/k3s/k3s.yaml` | Нет `/etc/rancher/k3s/config.yaml` из шага 2. После копирования — `sudo systemctl restart k3s` |
| `grep` из шага 3 выводит `0` | Toolkit поставлен после k3s — `sudo systemctl restart k3s` |
| `gpu-smoke` висит в `Pending` | На узле нет метки `gpu=true` |
| `apt-get update`: `Malformed entry` | Файл репозитория испорчен при ручном наборе — скопируйте `deploy/node/nvidia.sources` |
| `nvcr.io`: `403 Forbidden` | Реестр NVIDIA закрыт для России — см. раздел про этап 3 |

Источники: [k3s — рантайм NVIDIA](https://docs.k3s.io/advanced),
[k3s — настройка реестров](https://docs.k3s.io/installation/private-registry),
[k3s — установка без интернета](https://docs.k3s.io/installation/airgap),
[NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin).
