# Benchmark do SLOF em Docker

Este contrato foi adaptado para rodar o SLOF/FLOW360 em uma maquina Linux com GPU NVIDIA via Docker, sem alterar o codigo original do metodo.

## Premissas

- Host Linux com driver NVIDIA funcional.
- `nvidia-container-toolkit` instalado.
- GPU alvo semelhante a uma RTX 3060.
- O host pode expor CUDA 13 no driver; o container usa runtime CUDA 11.3, o que eh um fluxo valido por compatibilidade retroativa do driver.

## Estrutura esperada do dataset

Monte o dataset `FLOW360_train_test` no container em `/datasets/FLOW360_train_test`, preservando a estrutura:

```text
/datasets/FLOW360_train_test/
  train/
    000/
      frames/
      fflows/
      bflows/
  test/
    000/
      frames/
      fflows/
      bflows/
```

O caminho de referencia fora do container, no workspace atual, eh:

```text
/Volumes/External SSD/Mestrado/Datasets/FLOW360_train_test
```

Na maquina Linux de execucao, monte o dataset local correspondente nesse mesmo layout dentro de `/datasets/FLOW360_train_test`.

## Checkpoint padrao

O contrato usa por padrao:

```text
/app/weights/singlerotation.pt
```

Para trocar o checkpoint sem editar YAML:

```bash
-e SLOF_CHECKPOINT=/app/weights/ktn.pt
```

Se trocar a variante, ajuste tambem `config/experiment.yaml` ou mantenha nomes de arquivo que permitam deteccao automatica (`ktn`, `raft`, `raftfinetune`, `singlerotation`, `switchrotation`, `doublerotation`).

## Build

Rode na raiz do repositorio:

```bash
docker build -f benchmark_contrato/Dockerfile.benchmark -t slof-bench .
```

## Execucao

Monte dataset e resultados:

```bash
docker run --rm --gpus all \
  -v /caminho/linux/FLOW360_train_test:/datasets/FLOW360_train_test \
  -v "$(pwd)/benchmark_contrato/results:/app/benchmark_contrato/results" \
  -v "$(pwd)/benchmark_contrato/outputs:/app/benchmark_contrato/outputs" \
  slof-bench \
  ./benchmark_contrato/entrypoint.sh official_reproduction
```

Profiling padronizado:

```bash
docker run --rm --gpus all \
  -v /caminho/linux/FLOW360_train_test:/datasets/FLOW360_train_test \
  -v "$(pwd)/benchmark_contrato/results:/app/benchmark_contrato/results" \
  -v "$(pwd)/benchmark_contrato/outputs:/app/benchmark_contrato/outputs" \
  slof-bench \
  ./benchmark_contrato/entrypoint.sh standardized_efficiency
```

Robustez regional:

```bash
docker run --rm --gpus all \
  -v /caminho/linux/FLOW360_train_test:/datasets/FLOW360_train_test \
  -v "$(pwd)/benchmark_contrato/results:/app/benchmark_contrato/results" \
  -v "$(pwd)/benchmark_contrato/outputs:/app/benchmark_contrato/outputs" \
  slof-bench \
  ./benchmark_contrato/entrypoint.sh regional_robustness
```

## Saidas geradas

Ao final, os artefatos obrigatorios ficam em `benchmark_contrato/results/`:

- `metadata.json`
- `quality_metrics.json`
- `efficiency_metrics.json`
- `run_config.json`
- `environment.json`

Artefatos auxiliares ficam em `benchmark_contrato/outputs/`, incluindo uma predição de amostra em `.npy`.

## Observacoes

- O caminho de avaliacao replica as formulas de `evaluate_raft.py`.
- O loader do contrato reproduz o layout real do FLOW360 sem depender das partes legadas de I/O que puxam dependencias desnecessarias para o benchmark.
- O profiling usa tensores sinteticos em batch 1 na resolucao oficial `320x640`.
- O calculo de FLOPs tenta usar `fvcore`; se a analise falhar por incompatibilidade do grafo, o resultado eh gravado como `null`.
