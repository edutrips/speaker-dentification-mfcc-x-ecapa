# Identificação de Locutores — MFCC vs ECAPA-TDNN

Versão pública do projeto acadêmico de processamento de voz que identifica um dos locutores
cadastrados (V01–V04) ou rejeita uma gravação como `UNKNOWN`. A comparação usa
uma representação acústica clássica e uma representação neural pré-treinada.

## Objetivo

Comparar MFCC e ECAPA-TDNN com o mesmo tipo de classificador, um SVM RBF, em
gravações de sessões e textos distintos dos usados no treinamento. A decisão
open-set é feita por gravação, com um limiar de rejeição para cada representação.

## Arquitetura

```text
WAV ou microfone → áudio mono, 16 kHz → segmentos de 3 s
    → MFCC 120D ou ECAPA-TDNN 192D → StandardScaler → SVM RBF
    → média dos scores dos segmentos → limiar → V01–V04 ou UNKNOWN
```

Os segmentos completos avançam 1,5 s. Uma cauda com pelo menos 1,5 s é
completada com zeros; uma cauda menor é descartada. Para a demonstração, o áudio
de entrada deve ter pelo menos 3 s.

## Implementação MFCC

Para cada segmento, usam-se quadros de 25 ms com avanço de 10 ms, 40 filtros
Mel e 20 coeficientes MFCC. Acrescentam-se delta e delta-delta; a média e o
desvio padrão temporal das 60 séries formam um vetor de 120 dimensões. O
[librosa](https://librosa.org/) fornece as operações fundamentais de MFCC e
derivadas; o projeto define parametrização, segmentação, composição, validação e
integração. Código: [`mfcc.py`](src/speaker_id/features/mfcc.py),
[`segmentation.py`](src/speaker_id/features/segmentation.py) e
[`mfcc.yaml`](configs/mfcc.yaml).

## Classificação SVM e rejeição

Cada representação possui um pipeline independente com `StandardScaler` e
`SVC` RBF do [scikit-learn](https://scikit-learn.org/). Os classificadores são
ajustados com segmentos de V01–V04. Para cada gravação, calcula-se a média dos
scores por classe: o maior score aceita a classe correspondente quando é maior
ou igual ao limiar; caso contrário, a saída é `UNKNOWN`. Scores de decisão não
são probabilidades.

O código está em [`mfcc_svm.py`](src/speaker_id/models/mfcc_svm.py),
[`open_set_models.py`](src/speaker_id/training/open_set_models.py) e
[`rejection.py`](src/speaker_id/open_set/rejection.py).

## Integração ECAPA-TDNN

O encoder [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
é fornecido pelo SpeechBrain, pré-treinado e carregado com parâmetros congelados,
sem fine-tuning no experimento. Cada segmento produz uma embedding de 192
dimensões, usada por um pipeline do mesmo tipo `StandardScaler` + SVM RBF. O
projeto integra o encoder; não implementa a arquitetura original. Código:
[`ecapa.py`](src/speaker_id/features/ecapa.py) e
[`ecapa_pipeline.py`](src/speaker_id/features/ecapa_pipeline.py).

## Protocolo experimental

| Etapa | Sessão e textos | Gravações |
|---|---|---:|
| Treinamento | S01, T01–T05, V01–V04 | 20 conhecidas |
| Validação e calibração dos limiares | S02, T06–T10, V01–V04 e U01–U03 | 20 conhecidas e 15 desconhecidas |
| Teste final | S03, T11–T15, V01–V04 e U04–U06 | 20 conhecidas e 15 desconhecidas |

A calibração em S02 minimiza `0,5 × FAR + 0,5 × FRR`; em empate exato, escolhe
o maior limiar. S03 é avaliada com os modelos e limiares congelados. A unidade
de avaliação é a **gravação**, não o segmento. O [manifesto de inferência](docs/FINAL_OPEN_SET_A_FREEZE.json) preserva apenas
os campos necessários do contrato congelado; o manifesto científico completo
permanece no arquivo privado do experimento.

## Resultados finais em S03

| Métrica | MFCC + SVM | ECAPA + SVM |
|---|---:|---:|
| Acurácia geral | 77,14% | 97,14% |
| Identificação correta de conhecidos | 65,00% | 95,00% |
| Rejeição de desconhecidos | 93,33% | 100,00% |
| FAR — aceitação de desconhecidos | 6,67% | 0,00% |
| FRR — rejeição de conhecidos | 35,00% | 5,00% |
| Macro-F1 | 0,6933 | 0,9713 |

Os resultados resumem 35 gravações: 20 conhecidas e 15 desconhecidas. O
experimento local preserva as matrizes de confusão em
`artifacts/final_open_set_a/final_test/figures/mfcc_test_confusion_matrix.png` e
`artifacts/final_open_set_a/final_test/figures/ecapa_test_confusion_matrix.png`.
As figuras mostram contagens agregadas, sem gravações ou predições individuais.
A [comparação numérica](artifacts/final_open_set_a/final_test/results/comparison.csv)
e as [matrizes MFCC](artifacts/final_open_set_a/final_test/results/mfcc_confusion_matrix.csv)
e [ECAPA](artifacts/final_open_set_a/final_test/results/ecapa_confusion_matrix.csv)
estão incluídas nesta distribuição.

## Instalação e execução

Os dois SVMs congelados acompanham esta distribuição. As gravações históricas
não são necessárias. Use a interface de [`live_open_set.py`](scripts/live_open_set.py)
e consulte o [guia de inferência](docs/PORTABLE_INFERENCE.md).

O destino público planejado é `https://github.com/edutrips/speaker-identification-mfcc-ecapa`;
sua disponibilidade ainda precisa ser verificada antes do envio. A partir da pasta
da distribuição:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/inference-mfcc.txt
python -m pip install --no-deps -e .
```

Para ECAPA em Linux x86-64 com CPU, use o índice oficial adicional do PyTorch:

```bash
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/inference-ecapa.txt
```

A instalação limpa dos dois caminhos foi validada com Python 3.12.3 em Linux
x86-64. O `pyproject.toml` declara também as dependências de ambos os caminhos e
`matplotlib`; para verificar todas com `pip check`, instale-as no mesmo ambiente.
As dependências transitivas não estão fixadas. Em outros sistemas, a disponibilidade
de wheels e a compatibilidade precisam ser verificadas. No Windows, a ativação do
ambiente é `.venv\Scripts\activate`.

Os SVMs estão em `artifacts/final_open_set_a/mfcc/model/model.joblib` e
`artifacts/final_open_set_a/ecapa/model/model.joblib`. A CLI verifica os hashes
antes de carregá-los. Para reinstalar cópias de um pacote local confiável, use:

```bash
python scripts/install_inference_models.py --source-root /caminho/para/pacote-confiavel
```

Para ECAPA, obtenha separadamente os cinco recursos do encoder da
[revisão candidata](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286)
e instale-os de um diretório local confiável:

```bash
python scripts/install_inference_models.py --source-root . --ecapa-source /caminho/para/encoder
```

O instalador e a aplicação verificam os hashes em
[`ecapa_resources.json`](configs/ecapa_resources.json); não há download
automático nem substituição silenciosa. A revisão do cache foi validada por
comparação numérica, mas sua identidade histórica exata não foi comprovada.
Detalhes, arquivos necessários e solução de problemas estão em
[`PORTABLE_INFERENCE.md`](docs/PORTABLE_INFERENCE.md).

Com um WAV novo de pelo menos 3 s, ou com um microfone disponível:

```bash
python scripts/live_open_set.py --representation mfcc --wav exemplo.wav
python scripts/live_open_set.py --representation ecapa --wav exemplo.wav
python scripts/live_open_set.py --representation ecapa --list-devices
python scripts/live_open_set.py --representation ecapa --device 0 --duration 6
```

Use um índice de dispositivo retornado pela listagem. Em modo WAV, a aplicação
não precisa de microfone. A execução com microfone depende de SoundDevice e
PortAudio; a captura física ainda não foi validada nesta preparação.

## Estrutura do repositório

- `src/speaker_id/`: carregamento de áudio, características, modelos e avaliação.
- `scripts/`: construção de características, treinamento, avaliação e inferência.
- `configs/`: parâmetros de áudio, características e recursos do encoder.
- `tests/`: testes automatizados.
- `docs/`: contratos, protocolo e instruções operacionais.
- `artifacts/`: SVMs autorizados e resultados agregados; dados históricos não são distribuídos.

## Limitações

O experimento usa poucos locutores e gravações coletadas em ambiente controlado;
os resultados não estabelecem generalização para outros grupos ou ambientes.
As amostras da mesma pessoa não devem ser interpretadas como observações
independentes de uma população geral. Os scores SVM não são probabilidades e a
demonstração não é um autenticador biométrico geral. A proveniência histórica
exata do encoder permanece pendente de comprovação. A instalação limpa foi
validada apenas no ambiente Linux x86-64 descrito acima; o acesso a um futuro
clone público ainda não foi testado. A distribuição dos SVMs não autoriza a publicação de
áudios, metadados pessoais ou outros dados identificáveis.

## Referências

- Davis, S.; Mermelstein, P. “Comparison of Parametric Representations for
  Monosyllabic Word Recognition in Continuously Spoken Sentences” (1980).
  [DOI: 10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420).
- Cortes, C.; Vapnik, V. “Support-vector networks” (1995).
  [DOI: 10.1007/BF00994018](https://doi.org/10.1007/BF00994018).
- Desplanques, B.; Thienpondt, J.; Demuynck, K. “ECAPA-TDNN” (2020).
  [arXiv:2005.07143](https://arxiv.org/abs/2005.07143).
- Ravanelli, M. et al. “SpeechBrain: A General-Purpose Speech Toolkit” (2021).
  [arXiv:2106.04624](https://arxiv.org/abs/2106.04624).
- [Documentação do librosa](https://librosa.org/doc/0.11.0/) e
  [documentação de SVM do scikit-learn](https://scikit-learn.org/stable/modules/svm.html).
