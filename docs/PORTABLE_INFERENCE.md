# Inferência open-set portátil

A aplicação usa modelos e limiares congelados para as classes V01–V04 e a saída
`UNKNOWN`. Execute os comandos na raiz do projeto. Não são necessários os WAVs
históricos, `recordings.csv`, matrizes de treino, CSVs de calibração ou resultados
S03. O WAV de entrada deve ter pelo menos 3 segundos. Os scores são valores de
decisão SVM, não probabilidades.

## Ambiente

Use Python 3.12. Para MFCC, crie e ative um ambiente virtual e execute:

```bash
python -m pip install -r requirements/inference-mfcc.txt
python -m pip install --no-deps -e .
```

Para ECAPA, no Linux x86-64 com CPU, instale as versões fixadas usando também o
índice oficial de wheels CPU do PyTorch:

```bash
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/inference-ecapa.txt
```

Em um venv novo sem pacotes de sistema, essa instalação foi validada com Python
3.12.3, NumPy 2.5.3, librosa 0.11.0, scikit-learn 1.9.1, torch 2.15.0+cpu,
torchaudio 2.11.1+cpu e SpeechBrain 1.1.2. O PyPI sozinho não ofereceu o wheel
`torch==2.15.0+cpu`; o índice adicional acima resolveu o pacote fixado. As versões
transitivas não estão fixadas, e a compatibilidade em outros sistemas não foi
verificada. Não substitua versões ou pesos silenciosamente. A CLI trata avisos
de incompatibilidade de versão dos modelos scikit-learn como erros.

O `pyproject.toml` declara as dependências completas dos dois pipelines e
`matplotlib`. Para um `pip check` sem ausências do projeto após a instalação
editável, instale também as dependências ECAPA e `matplotlib==3.11.2`:

```bash
python -m pip install matplotlib==3.11.2
python -m pip check
```

Para executar os testes da distribuição, instale adicionalmente `pytest==8.4.2`
e `ruff==0.16.8` e execute `python -m pytest -q` e `ruff check .`. Instalar
somente os requisitos MFCC permite executar esse caminho, mas `pip check` após a
instalação editável apontará as dependências completas ausentes do projeto.

MFCC não importa torch, torchaudio ou SpeechBrain. SoundFile requer libsndfile;
a listagem e a captura por microfone também precisam de SoundDevice e PortAudio.
Em modo WAV não é aberto dispositivo de áudio.

## Instalação local dos SVMs

A distribuição dos dois SVMs foi autorizada pelo responsável e os arquivos estão
incluídos nesta distribuição. Para reinstalá-los em outra cópia, obtenha de fonte
confiável um pacote local com esta estrutura:

```text
pacote-confiavel/
  artifacts/final_open_set_a/mfcc/model/model.joblib
  artifacts/final_open_set_a/ecapa/model/model.joblib
```

O pacote não precisa conter o dataset. Para reinstalar os SVMs na aplicação:

```bash
python scripts/install_inference_models.py --source-root /caminho/para/pacote-confiavel
```

O instalador verifica os dois hashes antes de copiar, preserva os bytes, verifica
os destinos e recusa substituir arquivos diferentes. Para outra cópia da
aplicação, acrescente `--destination-root /caminho/para/aplicacao`. Ele não usa
Git, não baixa arquivos e não desserializa modelos. Carregue arquivos `joblib`
somente de fonte confiável: um hash em manifesto não confiável não autentica o
modelo.

| Representação | SHA256 | Limiar |
|---|---|---:|
| MFCC 120D | `417beadded3b37f8ddc7f95771011422440e1d373ac65905a1b6f0f52f1d3e95` | 3.1804021896256245 |
| ECAPA 192D | `15c6d934b4bfc0919c295b8ab729d7c4638ed0da63b8054474632ce2643503e4` | 3.1886667008008707 |

Os destinos são os caminhos acima na raiz da aplicação. Hashes, limiares e ordem
das classes vêm de `docs/FINAL_OPEN_SET_A_FREEZE.json`; preserve-o junto com
`configs/mfcc.yaml`. Esse é um manifesto mínimo derivado, contendo os campos de
inferência copiados do contrato congelado; o manifesto científico completo é
preservado no arquivo privado do experimento.

## Recursos ECAPA

A CLI carrega os recursos ECAPA somente de arquivos locais e verifica os cinco
hashes em `configs/ecapa_resources.json`. O diretório de origem deve conter:

- `hyperparams.yaml`;
- `embedding_model.ckpt`;
- `classifier.ckpt`;
- `mean_var_norm_emb.ckpt`;
- `label_encoder.txt` — ou o alias de cache `label_encoder.ckpt`.

Uma alternativa para obtê-los é a [revisão candidata no Hugging Face](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286).
Obtenha apenas esses cinco arquivos em uma pasta confiável e instale-os, junto com
os SVMs, por exemplo:

```bash
python scripts/install_inference_models.py --source-root . --ecapa-source /caminho/para/encoder
```

O instalador verifica os hashes e copia os recursos para
`artifacts/ecapa_encoder/`. A CLI usa esse diretório quando existe; caso contrário,
verifica `~/.cache/speechbrain/spkrec-ecapa-voxceleb`. Também é possível indicar
`--ecapa-dir /caminho/para/encoder`. Recursos ausentes ou incompatíveis causam
falha: não há download ou fallback remoto durante a inferência. O SpeechBrain
carrega cópias temporárias com acesso à rede desabilitado.

A revisão `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` identifica o snapshot
observado no cache; **não comprova a revisão histórica exata** utilizada pelo
experimento. Os hashes descrevem os arquivos locais que foram comparados. Uma
verificação offline dos 197 segmentos S01 de 20 gravações encontrou diferença
máxima de embedding de `3.814697265625e-05` e diferença máxima de score médio por
Gravação de `1.635581381442819e-08`; as classes candidatas e decisões de
aceitação coincidiram. A verificação sustenta compatibilidade numérica nesse
conjunto, sem provar identidade histórica nem exatidão bit a bit. Os WAVs usados
nessa comparação não integram o pacote portátil.

O SpeechBrain fornece arquitetura e pesos; o projeto mantém os parâmetros
congelados e integra a inferência. O [cartão do modelo](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
informa licença Apache-2.0 e referências para ECAPA-TDNN e SpeechBrain. A
autorização de distribuição dos SVMs não se estende automaticamente a áudios,
metadados ou outros dados identificáveis.

## Execução

```bash
python scripts/live_open_set.py --representation mfcc --wav exemplo.wav
python scripts/live_open_set.py --representation ecapa --wav exemplo.wav
python scripts/live_open_set.py --representation ecapa --list-devices
python scripts/live_open_set.py --representation ecapa --device 0 --duration 6
```

Use `--verbose` para exibir todos os scores médios. `--device` seleciona o
microfone, não uma GPU; ECAPA usa CPU. A captura passa por um WAV temporário,
removido ao final. A CLI não grava features nem adiciona entradas ao manifest de
coleta.

## Validação e solução de problemas

Uma cópia isolada executou anteriormente os dois caminhos com recursos locais e
um WAV sintético, com rede e dados históricos bloqueados. Depois, a distribuição
persistente foi instalada em venv novo sem pacotes de sistema, em Linux x86-64,
Python 3.12.3, usando o PyPI e o índice CPU do PyTorch. `pip check` passou após a
instalação das dependências completas e de matplotlib; os dois caminhos foram
executados com WAV sintético, e 107 testes e Ruff passaram. Os scores desse WAV
coincidiram com os do ambiente anterior. A listagem posterior não encontrou
microfones. Áudio sintético testa a estrutura, não a acurácia; captura física e
instalação em outros sistemas não foram verificadas. Nenhum clone público foi testado.

- **SVM ausente:** instale um pacote confiável nos caminhos esperados.
- **Hash divergente:** recupere os bytes congelados corretos; não altere o
  manifesto, o limiar ou o classificador para aceitar outro arquivo.
- **ECAPA ausente:** instale os cinco recursos compatíveis ou informe
  `--ecapa-dir`; a CLI não baixa substitutos.
- **Erro de versão ou importação:** confira a versão de Python, a arquitetura e
  o acesso aos índices documentados. O wheel `+cpu` requer o índice PyTorch CPU;
  fora do ambiente Linux x86-64 testado, sua disponibilidade pode ser diferente.
- **Sem microfone ou erro PortAudio:** configure o áudio do sistema ou use WAV.
- **Resultado inesperado:** o sistema foi calibrado para os locutores cadastrados
  e não é um autenticador biométrico de uso geral.
