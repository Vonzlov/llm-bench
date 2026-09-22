# k3s с GPU на одной машине

Цель — чтобы под в кластере видел видеокарту как ресурс `nvidia.com/gpu`. Это самый рискованный
шаг проекта, поэтому он идёт первым: если здесь тупик, запасной план включается сразу, а не в конце.

## Что нужно

- Linux с NVIDIA GPU и драйвером: команда `nvidia-smi` на хосте работает. В WSL2 драйвер ставится
  в Windows, а внутри WSL — нет.
- [Helm](https://helm.sh/docs/intro/install/).

## 1. NVIDIA Container Toolkit

Ставится по [официальной инструкции NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Настраивать containerd руками не нужно: k3s сделает это сам на следующем шаге.

## 2. k3s

```bash
curl -sfL https://get.k3s.io | sh -

# k3s должен найти рантайм NVIDIA и прописать его в конфиг containerd:
sudo grep -r nvidia /var/lib/rancher/k3s/agent/etc/containerd/
```

Если toolkit ставился после k3s, k3s нужно перезапустить: `sudo systemctl restart k3s`.

kubeconfig лежит в `/etc/rancher/k3s/k3s.yaml`. Чтобы работать обычным kubectl:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$USER" ~/.kube/config
```

По метке `gpu=true` серверы моделей находят узел с видеокартой:

```bash
kubectl get nodes
kubectl label node <имя-узла> gpu=true
```

## 3. NVIDIA device plugin

k3s регистрирует RuntimeClass `nvidia`, но не делает её рантаймом по умолчанию. Поэтому плагин
ставится с `runtimeClassName=nvidia`, иначе он не увидит карту.

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --create-namespace \
  --version 0.17.1 \
  --set runtimeClassName=nvidia
```

Проверка — у узла должна появиться GPU:

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

## 4. Smoke-тест

```bash
kubectl apply -f deploy/smoke/gpu-smoke.yaml
kubectl logs -f pod/gpu-smoke   # в конце должно быть «Test PASSED»
kubectl delete -f deploy/smoke/gpu-smoke.yaml
```

## Если не заработало

| Симптом | Что проверить |
| --- | --- |
| Плагин пишет `No devices found` | `runtimeClassName=nvidia` у плагина. Другой путь — `default-runtime: nvidia` в `/etc/rancher/k3s/config.yaml` и перезапуск k3s |
| `gpu-smoke` висит в `Pending` | Есть ли у узла `nvidia.com/gpu` (шаг 3) и метка `gpu=true` |
| `grep` из шага 2 ничего не нашёл | Toolkit не установлен или k3s не перезапущен после установки |
| `CUDA driver version is insufficient` | Драйвер старше CUDA 12.5 из примера — обновить драйвер |

Источники: [k3s — рантайм NVIDIA](https://docs.k3s.io/advanced),
[NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin).
