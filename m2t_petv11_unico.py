#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
m2t_petv11_unico.py  (rev. 2.0)

Pipeline experimental único para Model-to-Text (M2T) sobre PETv11.

Mudanças em relação à versão anterior (resumo):
- B1: O_OM e O_OO agora medem fenômenos distintos (omissão vs. má-representação).
- B2: O_OT documentado e renomeado conceitualmente como Kendall-tau distance normalizada.
- B3: MR analisado via GLM logístico + teste pareado de McNemar (em vez de OLS gaussiana).
- B4: novo subcomando `validate-models` para checagem rápida de IDs.
- S1: seed propagada para random, numpy e OpenAI (quando suportado); temperature=0 por padrão.
- S2: PROMPT_VERSION e CODE_VERSION incluídos na chave do cache.
- S3: TF-IDF agora é ajustado no corpus completo (CorpusTfidf), não em pares.
- S5: detokenização robusta + recomendação de embeddings multilíngues (PETv11 é em inglês).
- S6: retries só para erros transitórios (429/5xx/timeout); permanentes (4xx) falham rápido.
- S7: coluna `diagnostic_parse_ok` no CSV.
- S8: tratamento de Wilcoxon quando todos os pares são iguais.
- M5: logging estruturado.
- M6: salvamento incremental de results.csv (checkpoint por modelo).
- M10: flag `--validate-only` no subcomando `experiment`.
- NOVO: subcomando `plots` que gera figuras IEEE-ready a partir de results.csv,
  incluindo distribuições por KPI, deltas pareados com IC bootstrap, heatmap de
  tamanho de efeito Wilcoxon, diagrama de diferença crítica (Friedman + Nemenyi),
  decomposição de erros por tipo e barras de MR com McNemar.

Uso:
    python m2t_petv11_unico.py validate-models
    python m2t_petv11_unico.py experiment --sample-size 45
    python m2t_petv11_unico.py analysis --input outputs/results.csv
    python m2t_petv11_unico.py plots    --input outputs/results.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import requests
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return None

try:
    from sacrebleu.metrics import BLEU
except Exception:
    BLEU = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.contingency_tables import mcnemar
except Exception:
    sm = None
    smf = None
    anova_lm = None
    mcnemar = None


# ============================================================
# 0. VERSIONAMENTO
# ============================================================

PROMPT_VERSION = "ptbr_v1.0"
CODE_VERSION = "2.1.1"


# ============================================================
# 1. CONFIGURAÇÕES (dataclasses imutáveis)
# ============================================================

Provider = Literal["anthropic", "openai", "deepseek"]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "patriziobellan/PETv11"
    split: str = "test"
    sample_size: int = 45
    random_seed: int = 42
    cache_enabled: bool = True


@dataclass(frozen=True)
class GenerationStyleConfig:
    language: str = "pt-BR"
    style: str = "formal_academic"
    forbidden_terms: tuple[str, ...] = (
        "bpmn",
        "mermaid",
        "flowchart",
        "nó",
        "node",
        "gateway",
        "grafo",
        "aresta",
    )


@dataclass(frozen=True)
class DiagnosticThresholds:
    """
    Limiares para classificar a similaridade entre tarefa esperada e melhor
    sentença correspondente no texto gerado.

    score < weak_match     ⇒ tarefa OMITIDA            → contribui para O_OM
    weak_match ≤ score < strong_match  ⇒ MAL-REPRESENTADA   → contribui para O_OO
    score ≥ strong_match   ⇒ bem representada
    """
    weak_match: float = 0.18
    strong_match: float = 0.40


CONFIG = ExperimentConfig()
STYLE_CONFIG = GenerationStyleConfig()
THRESHOLDS = DiagnosticThresholds()


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: Provider
    display_name: str
    family: str
    temperature: float = 0.0  # determinístico por padrão (S1)
    max_tokens: int = 1200
    # Modelos com extended thinking (ex.: Claude Opus 4.7) recusam temperature.
    send_temperature: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "claude_opus_4_7": ModelSpec(
        "claude-opus-4-7", "anthropic", "Claude Opus 4.7", "Anthropic",
        send_temperature=False,  # extended thinking model
    ),
    "claude_sonnet_4_6": ModelSpec(
        "claude-sonnet-4-6", "anthropic", "Claude Sonnet 4.6", "Anthropic"
    ),
    "claude_haiku_4_5": ModelSpec(
        "claude-haiku-4-5", "anthropic", "Claude Haiku 4.5", "Anthropic"
    ),
    "gpt_5_5": ModelSpec(
        "gpt-5.5", "openai", "GPT-5.5", "OpenAI",
        send_temperature=False,  # reasoning model
    ),
    "gpt_5_4": ModelSpec("gpt-5.4", "openai", "GPT-5.4", "OpenAI"),
    "gpt_4": ModelSpec("gpt-4", "openai", "GPT-4", "OpenAI"),
    "deepseek_v4_flash": ModelSpec(
        "deepseek-v4-flash", "deepseek", "DeepSeek v4 Flash", "DeepSeek"
    ),
    "deepseek_v4_pro": ModelSpec(
        "deepseek-v4-pro", "deepseek", "DeepSeek v4 Pro", "DeepSeek"
    ),
}

METRIC_NAMES = ["TS_C", "TS_NC", "BLEU_1", "MR", "O_OM", "O_OT", "O_OO", "O_OG"]
METRIC_HIGHER_IS_BETTER = {
    "TS_C": True,
    "TS_NC": True,
    "BLEU_1": True,
    "MR": False,
    "O_OM": False,
    "O_OT": False,
    "O_OO": False,
    "O_OG": False,
}


# ============================================================
# 2. PATHS + LOGGING + SEEDS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CACHE_DIR = ROOT_DIR / ".cache"
GEN_DIR = OUTPUT_DIR / "generations"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
FIGS_DIR = OUTPUT_DIR / "figures"


def setup_directories() -> None:
    for path in [DATA_DIR, OUTPUT_DIR, CACHE_DIR, GEN_DIR, DIAG_DIR, FIGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    setup_directories()
    handlers = [
        logging.FileHandler(OUTPUT_DIR / "experiment.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("m2t")


def apply_seed(seed: int = CONFIG.random_seed) -> None:
    random.seed(seed)
    np.random.seed(seed)


log = setup_logging()
apply_seed()
load_dotenv(override=True)


# ============================================================
# 3. PROMPTS PT-BR VERSIONADOS
# ============================================================

def single_shot_prompt(mermaid_model: str) -> str:
    return f"""
Você é um especialista em transformação Model-to-Text.

Tarefa:
Gere uma descrição textual em português brasileiro, clara e formal, a partir do modelo de processo abaixo.

Regras:
1. Descreva o processo em linguagem natural.
2. Preserve a ordem lógica das atividades.
3. Não mencione termos internos de modelagem, como Mermaid, BPMN, flowchart, nó, aresta ou gateway.
4. Não invente atividades que não estejam representadas no modelo.
5. Não omita atividades relevantes.
6. Produza apenas o texto final, sem comentários adicionais.

Modelo Mermaid LR:
```mermaid
{mermaid_model}
```
""".strip()


def diagnostic_prompt(
    mermaid_model: str, generated_text: str, expected_tasks: list[str]
) -> str:
    tasks = "\n".join(f"- {task}" for task in expected_tasks)
    return f"""
Você é um avaliador crítico de transformação Model-to-Text.

Modelo Mermaid LR:
```mermaid
{mermaid_model}
```

Tarefas esperadas:
{tasks}

Texto gerado:
\"\"\"
{generated_text}
\"\"\"

Classifique os problemas nas categorias:
1. O-OM: tarefas esperadas omitidas no texto.
2. O-OT: problemas de ordem, sequência ou topologia.
3. O-OO: tarefas presentes mas mal representadas.
4. O-OG: tarefas espúrias/inventadas.
5. MR: vazamento de jargão de modelagem.

Responda em JSON válido:
{{
  "missing_tasks": [],
  "order_topology_errors": [],
  "lost_original_tasks": [],
  "spurious_tasks": [],
  "modeling_register_leakage": [],
  "summary": "diagnóstico breve"
}}
""".strip()


def self_refine_prompt(
    mermaid_model: str,
    first_output: str,
    diagnostic_json: str,
    expected_tasks: list[str],
) -> str:
    tasks = "\n".join(f"- {task}" for task in expected_tasks)
    return f"""
Você é um especialista em revisão de saídas Model-to-Text.

Gere uma versão revisada do texto, corrigindo os problemas do diagnóstico.

Modelo Mermaid LR:
```mermaid
{mermaid_model}
```

Tarefas esperadas:
{tasks}

Primeira saída:
\"\"\"
{first_output}
\"\"\"

Diagnóstico automático:
```json
{diagnostic_json}
```

Instruções:
1. Inclua tarefas esperadas omitidas.
2. Remova tarefas inventadas ou não sustentadas pelo modelo.
3. Corrija a ordem lógica.
4. Evite qualquer menção a Mermaid, BPMN, flowchart, nó, aresta, gateway.
5. Produza apenas o texto revisado final.

Texto revisado:
""".strip()


# ============================================================
# 4. CLIENTE LLM: cache + retry seletivo + erros tipados
# ============================================================

class LLMClientError(RuntimeError):
    pass


class TransientLLMError(LLMClientError):
    """Erros que devem disparar retry (429, 5xx, timeouts)."""


class PermanentLLMError(LLMClientError):
    """Erros que NÃO devem disparar retry (auth, modelo inexistente, etc.)."""


def _classify_http_error(status: int, body: str) -> LLMClientError:
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return TransientLLMError(f"HTTP {status}: {body[:300]}")
    return PermanentLLMError(f"HTTP {status}: {body[:300]}")


class Cache:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: dict[str, Any]) -> Path:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, payload: dict[str, Any]) -> str | None:
        if not CONFIG.cache_enabled:
            return None
        path = self._key(payload)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)["text"]

    def set(self, payload: dict[str, Any], text: str) -> None:
        if not CONFIG.cache_enabled:
            return
        path = self._key(payload)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"text": text, "payload": payload}, f, ensure_ascii=False, indent=2
            )


class UnifiedLLMClient:
    def __init__(self):
        self.cache = Cache()

    def generate(self, model: ModelSpec, prompt: str) -> str:
        payload = {
            "provider": model.provider,
            "model_id": model.model_id,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
            "prompt": prompt,
            "prompt_version": PROMPT_VERSION,
            "code_version": CODE_VERSION,
        }
        cached = self.cache.get(payload)
        if cached is not None:
            return cached

        if model.provider == "openai":
            text = self._call_openai(model, prompt)
        elif model.provider == "anthropic":
            text = self._call_anthropic(model, prompt)
        elif model.provider == "deepseek":
            text = self._call_deepseek(model, prompt)
        else:
            raise PermanentLLMError(f"Provider não suportado: {model.provider}")

        self.cache.set(payload, text)
        return text

    @staticmethod
    def _extract_openai_responses_text(data: dict) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
        parts: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if isinstance(content, dict):
                    if isinstance(content.get("text"), str):
                        parts.append(content["text"])
                    elif isinstance(content.get("output_text"), str):
                        parts.append(content["output_text"])
        text = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
        if text:
            return text
        raise PermanentLLMError(
            f"Não foi possível extrair texto da resposta OpenAI: {data}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(TransientLLMError),
    )
    def _call_openai(self, model: ModelSpec, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise PermanentLLMError("OPENAI_API_KEY não configurada.")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        use_responses = os.getenv("OPENAI_USE_RESPONSES_API", "1") != "0"

        body: dict[str, Any] = {"model": model.model_id}
        if use_responses:
            url = f"{base_url}/responses"
            body.update({
                "input": prompt,
                "max_output_tokens": model.max_tokens,
            })
            if model.send_temperature:
                body["temperature"] = model.temperature
        else:
            url = f"{base_url}/chat/completions"
            body.update({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": model.max_tokens,
            })
            if model.send_temperature:
                body["temperature"] = model.temperature

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
        except requests.Timeout as exc:
            raise TransientLLMError(f"Timeout OpenAI: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientLLMError(f"Conexão OpenAI: {exc}") from exc

        if not response.ok:
            raise _classify_http_error(response.status_code, response.text)

        data = response.json()
        if use_responses:
            return self._extract_openai_responses_text(data)
        return data["choices"][0]["message"]["content"].strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(TransientLLMError),
    )
    def _call_deepseek(self, model: ModelSpec, prompt: str) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise PermanentLLMError("DEEPSEEK_API_KEY não configurada.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        url = f"{base_url}/chat/completions"

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": model.temperature,
                    "max_tokens": model.max_tokens,
                },
                timeout=120,
            )
        except requests.Timeout as exc:
            raise TransientLLMError(f"Timeout DeepSeek: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientLLMError(f"Conexão DeepSeek: {exc}") from exc

        if not response.ok:
            raise _classify_http_error(response.status_code, response.text)

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(TransientLLMError),
    )
    def _call_anthropic(self, model: ModelSpec, prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise PermanentLLMError("ANTHROPIC_API_KEY não configurada.")
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        url = f"{base_url}/messages"

        body = {
            "model": model.model_id,
            "max_tokens": model.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.send_temperature:
            body["temperature"] = model.temperature

        try:
            response = requests.post(
                url,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
        except requests.Timeout as exc:
            raise TransientLLMError(f"Timeout Anthropic: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientLLMError(f"Conexão Anthropic: {exc}") from exc

        if not response.ok:
            raise _classify_http_error(response.status_code, response.text)

        data = response.json()
        return "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ).strip()


# ============================================================
# 5. DATA LOADER PETv11
# ============================================================

@dataclass
class ProcessExample:
    process_id: str
    original_text: str
    tasks: list[str]
    mermaid_lr: str


_DETOKENIZER = None


def _detokenize(tokens: list[str]) -> str:
    """
    Destokenização robusta. Usa sacremoses (Moses) se disponível;
    caso contrário aplica regras básicas para pontuação.
    """
    global _DETOKENIZER
    if _DETOKENIZER is None:
        try:
            from sacremoses import MosesDetokenizer
            _DETOKENIZER = MosesDetokenizer(lang="en")
        except Exception:
            _DETOKENIZER = False  # marca tentativa malsucedida

    if _DETOKENIZER:
        return _DETOKENIZER.detokenize(tokens)

    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_task(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else text


def iob_to_tasks(tokens: list[str], tags: list[str]) -> list[str]:
    """
    Extrai atividades de tags IOB. Match EXATO em "B-Activity"/"I-Activity"
    para não capturar "B-Activity Data".
    """
    tasks: list[str] = []
    current: list[str] = []

    for token, tag in zip(tokens, tags):
        tag = str(tag)
        if tag == "B-Activity":
            if current:
                tasks.append(normalise_task(" ".join(current)))
                current = []
            current.append(str(token))
        elif tag == "I-Activity" and current:
            current.append(str(token))
        else:
            if current:
                tasks.append(normalise_task(" ".join(current)))
                current = []
    if current:
        tasks.append(normalise_task(" ".join(current)))

    seen, unique = set(), []
    for task in tasks:
        key = task.lower()
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


def tasks_to_mermaid_lr(tasks: list[str]) -> str:
    lines = ["flowchart LR", "    Start([Start])"]
    previous = "Start"
    for idx, task in enumerate(tasks, start=1):
        node = f"T{idx}"
        safe_label = task.replace('"', "'")
        lines.append(f'    {node}["{safe_label}"]')
        lines.append(f"    {previous} --> {node}")
        previous = node
    lines.append("    End([End])")
    lines.append(f"    {previous} --> End")
    return "\n".join(lines)


def _synthetic_examples() -> list[ProcessExample]:
    items = [
        {
            "process_id": "synthetic_001",
            "text": (
                "O cliente envia o pedido. O sistema valida os dados. "
                "A equipe aprova a solicitação. O sistema notifica o cliente."
            ),
            "tasks": [
                "Cliente envia o pedido",
                "Sistema valida os dados",
                "Equipe aprova a solicitação",
                "Sistema notifica o cliente",
            ],
        },
        {
            "process_id": "synthetic_002",
            "text": (
                "O usuário preenche o formulário. A plataforma verifica os campos. "
                "O pagamento é processado. O comprovante é emitido."
            ),
            "tasks": [
                "Usuário preenche o formulário",
                "Plataforma verifica os campos",
                "Pagamento é processado",
                "Comprovante é emitido",
            ],
        },
    ]
    return [
        ProcessExample(
            process_id=item["process_id"],
            original_text=item["text"],
            tasks=item["tasks"],
            mermaid_lr=tasks_to_mermaid_lr(item["tasks"]),
        )
        for item in items
    ]


def load_petv11(sample_size: int | None = None) -> list[ProcessExample]:
    sample_size = sample_size or CONFIG.sample_size
    try:
        from datasets import load_dataset
        dataset = load_dataset(CONFIG.dataset_name, split=CONFIG.split)

        tag_column = "ner-tags" if "ner-tags" in dataset.column_names else "ner_tags"
        token_column = "tokens"
        doc_column = (
            "document name" if "document name" in dataset.column_names else "document_name"
        )
        sent_column = (
            "sentence-ID" if "sentence-ID" in dataset.column_names else "sentence_id"
        )

        label_names = None
        try:
            feature = dataset.features.get(tag_column)
            if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                label_names = feature.feature.names
            elif hasattr(feature, "names"):
                label_names = feature.names
        except Exception:
            label_names = None

        df = pd.DataFrame(dataset)
        if doc_column not in df.columns:
            raise ValueError(
                f"Coluna de documento ausente. Disponíveis: {list(df.columns)}"
            )

        df = (
            df.sort_values([doc_column, sent_column])
            if sent_column in df.columns
            else df.sort_values([doc_column])
        )

        examples: list[ProcessExample] = []
        for doc_name, group in df.groupby(doc_column, sort=False):
            all_tokens: list[str] = []
            all_tasks: list[str] = []
            for _, row in group.iterrows():
                tokens = row.get(token_column, [])
                tags = row.get(tag_column, [])
                if isinstance(tokens, str):
                    tokens = tokens.split()
                if tags and isinstance(tags[0], (int, np.integer)):
                    if label_names is not None:
                        tags = [label_names[int(t)] for t in tags]
                    else:
                        continue
                all_tokens.extend(map(str, tokens))
                if tokens and tags:
                    all_tasks.extend(iob_to_tasks(tokens, tags))

            seen, unique_tasks = set(), []
            for task in all_tasks:
                key = task.lower()
                if key not in seen:
                    seen.add(key)
                    unique_tasks.append(task)

            if not unique_tasks:
                continue

            examples.append(
                ProcessExample(
                    process_id=str(doc_name),
                    original_text=_detokenize(all_tokens),
                    tasks=unique_tasks,
                    mermaid_lr=tasks_to_mermaid_lr(unique_tasks),
                )
            )
            if len(examples) >= sample_size:
                break

        if examples:
            log.info(
                "PETv11 carregado: %d processos (split=%s).",
                len(examples), CONFIG.split,
            )
            log.info("Primeiro processo: %s", examples[0].process_id)
            log.info("Atividades (preview): %s", examples[0].tasks[:5])
            return examples

        log.warning("PETv11 sem atividades extraídas; usando fallback sintético.")
        return _synthetic_examples()

    except Exception as exc:
        log.warning("Falha ao carregar PETv11: %s — usando fallback sintético.", exc)
        return _synthetic_examples()


# ============================================================
# 6. SCORERS (TF-IDF de corpus + embeddings multilíngues)
# ============================================================

class CorpusTfidf:
    """
    TF-IDF ajustado em um corpus completo de referências, evitando o
    problema de IDF degenerado quando se ajusta em apenas 2 documentos.
    """

    def __init__(self, references: Iterable[str]):
        refs = [r for r in references if r and r.strip()]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        if refs:
            self.vectorizer.fit(refs)
            self._fitted = True
        else:
            self._fitted = False

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        if not self._fitted:
            # fallback isolado (degenera mas não quebra)
            v = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            m = v.fit_transform([a, b])
            return float(cosine_similarity(m[0], m[1])[0, 0])
        m = self.vectorizer.transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0, 0])


class SemanticScorer:
    """
    Calcula similaridade contextual via sentence-transformers.
    Por padrão usa modelo MULTILÍNGUE — adequado para comparar texto
    em pt-BR (gerado) com referência em inglês (PETv11).
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None):
        self.model = None
        name = model_name or os.getenv("ST_MODEL", self.DEFAULT_MODEL)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name)
            log.info("SemanticScorer carregado: %s", name)
        except Exception as exc:
            log.warning("sentence-transformers indisponível (%s); usando TF-IDF.", exc)

    def score(self, a: str, b: str, fallback: CorpusTfidf | None = None) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        if self.model is None:
            return fallback.similarity(a, b) if fallback else 0.0
        emb = self.model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))


_SHARED_SCORER: SemanticScorer | None = None


def get_scorer() -> SemanticScorer:
    global _SHARED_SCORER
    if _SHARED_SCORER is None:
        _SHARED_SCORER = SemanticScorer()
    return _SHARED_SCORER


# ============================================================
# 7. DIAGNÓSTICO E TAXAS DE ERRO  (B1 + B2)
# ============================================================

@dataclass
class ErrorRates:
    O_OM: float
    O_OT: float
    O_OO: float
    O_OG: float
    MR: float


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def contains_modeling_register(text: str) -> bool:
    lowered = text.lower()
    return any(t in lowered for t in STYLE_CONFIG.forbidden_terms)


def _diagnostic_tfidf(expected_tasks: list[str], sentences: list[str]) -> list[list[float]]:
    """TF-IDF interno só para o diagnóstico (escopo: tarefas + sentenças)."""
    if not expected_tasks or not sentences:
        return []
    corpus = expected_tasks + sentences
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    m = vec.fit_transform(corpus)
    task_vecs = m[: len(expected_tasks)]
    sent_vecs = m[len(expected_tasks):]
    return cosine_similarity(task_vecs, sent_vecs).tolist()


def estimate_error_rates(
    expected_tasks: list[str],
    generated_text: str,
    thresholds: DiagnosticThresholds = THRESHOLDS,
) -> ErrorRates:
    """
    O_OM (omissão): tarefas esperadas com similaridade < weak_match.
    O_OO (má-representação): weak_match ≤ similaridade < strong_match.
    O_OT (Kendall-tau distance norm.): fração de pares (i,j), i<j em
        expected_tasks, cujas melhores correspondências aparecem invertidas
        no texto gerado. Pré-requisito: expected_tasks vem em ordem do processo.
    O_OG (geração espúria): sentenças sem correspondência adequada com
        nenhuma tarefa esperada (similaridade < weak_match).
    MR: indicador binário de vazamento de jargão de modelagem.
    """
    sentences = split_sentences(generated_text)
    match = _diagnostic_tfidf(expected_tasks, sentences)

    if not expected_tasks:
        return ErrorRates(0.0, 0.0, 0.0, 0.0, 0.0)
    if not sentences:
        return ErrorRates(1.0, 1.0, 1.0, 0.0, float(contains_modeling_register(generated_text)))

    task_best_scores = [max(row) if row else 0.0 for row in match]
    omitted = [s < thresholds.weak_match for s in task_best_scores]
    poorly_represented = [
        thresholds.weak_match <= s < thresholds.strong_match for s in task_best_scores
    ]

    o_om = sum(omitted) / len(expected_tasks)
    o_oo = sum(poorly_represented) / len(expected_tasks)

    # Kendall-tau distance normalizada
    best_positions = []
    for row in match:
        if row:
            best_positions.append(max(range(len(row)), key=lambda j: row[j]))
        else:
            best_positions.append(-1)

    inversions = comparable = 0
    for i in range(len(best_positions)):
        for j in range(i + 1, len(best_positions)):
            if best_positions[i] >= 0 and best_positions[j] >= 0:
                comparable += 1
                if best_positions[i] > best_positions[j]:
                    inversions += 1
    o_ot = inversions / comparable if comparable else 0.0

    if match:
        sent_best = [max(row[s] for row in match) for s in range(len(sentences))]
        spurious = [s < thresholds.weak_match for s in sent_best]
        o_og = sum(spurious) / len(sentences)
    else:
        o_og = 0.0

    mr = 1.0 if contains_modeling_register(generated_text) else 0.0

    return ErrorRates(
        O_OM=round(o_om, 4),
        O_OT=round(o_ot, 4),
        O_OO=round(o_oo, 4),
        O_OG=round(o_og, 4),
        MR=round(mr, 4),
    )


def build_local_diagnostic(expected_tasks: list[str], generated_text: str) -> dict:
    rates = estimate_error_rates(expected_tasks, generated_text)
    return {
        "missing_tasks_rate": rates.O_OM,
        "order_topology_error_rate": rates.O_OT,
        "lost_original_tasks_rate": rates.O_OO,
        "spurious_generation_rate": rates.O_OG,
        "modeling_register_leakage": rates.MR,
        "summary": (
            "Diagnóstico heurístico baseado em similaridade TF-IDF, ordem relativa "
            "(Kendall-tau norm.) e presença de jargão de modelagem."
        ),
    }


def parse_json_safely(text: str) -> tuple[dict, bool]:
    """Retorna (diag, ok). ok=False se não conseguiu parsear."""
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), True
            except json.JSONDecodeError:
                pass
    return {"raw_diagnostic": text}, False


# ============================================================
# 8. GERADOR M2T
# ============================================================

@dataclass
class GenerationResult:
    output: str
    intermediate: str | None = None
    diagnostic: dict | None = None
    diagnostic_parse_ok: bool = True
    used_feedback: bool = False


class M2TGenerator:
    def __init__(self, client: UnifiedLLMClient | None = None):
        self.client = client or UnifiedLLMClient()

    def single_shot(self, model: ModelSpec, mermaid_model: str) -> GenerationResult:
        text = self.client.generate(model, single_shot_prompt(mermaid_model))
        return GenerationResult(output=text, used_feedback=False)

    def self_refine(
        self,
        model: ModelSpec,
        mermaid_model: str,
        expected_tasks: list[str],
        use_llm_diagnostic: bool = False,
    ) -> GenerationResult:
        first = self.client.generate(model, single_shot_prompt(mermaid_model))

        parse_ok = True
        if use_llm_diagnostic:
            raw = self.client.generate(
                model,
                diagnostic_prompt(mermaid_model, first, expected_tasks),
            )
            diagnostic, parse_ok = parse_json_safely(raw)
            if not parse_ok:
                log.warning(
                    "Diagnóstico LLM ilegível para modelo=%s; fallback heurístico.",
                    model.display_name,
                )
                diagnostic = build_local_diagnostic(expected_tasks, first)
        else:
            diagnostic = build_local_diagnostic(expected_tasks, first)

        refined = self.client.generate(
            model,
            self_refine_prompt(
                mermaid_model=mermaid_model,
                first_output=first,
                diagnostic_json=json.dumps(diagnostic, ensure_ascii=False, indent=2),
                expected_tasks=expected_tasks,
            ),
        )

        return GenerationResult(
            output=refined,
            intermediate=first,
            diagnostic=diagnostic,
            diagnostic_parse_ok=parse_ok,
            used_feedback=True,
        )


# ============================================================
# 9. AVALIAÇÃO (8 KPIs)
# ============================================================

def bleu_1(reference: str, prediction: str) -> float:
    if not reference.strip() or not prediction.strip():
        return 0.0
    if BLEU is None:
        ref_tokens = reference.lower().split()
        pred_tokens = set(prediction.lower().split())
        if not ref_tokens:
            return 0.0
        return sum(1 for t in ref_tokens if t in pred_tokens) / len(ref_tokens)
    bleu = BLEU(max_ngram_order=1, effective_order=True)
    return float(bleu.sentence_score(prediction, [reference]).score / 100.0)


def evaluate_output(
    expected_text: str,
    expected_tasks: list[str],
    generated_text: str,
    scorer: SemanticScorer,
    corpus_tfidf: CorpusTfidf,
) -> dict:
    ts_c = scorer.score(expected_text, generated_text, fallback=corpus_tfidf)
    ts_nc = corpus_tfidf.similarity(expected_text, generated_text)
    b1 = bleu_1(expected_text, generated_text)
    rates = estimate_error_rates(expected_tasks, generated_text)
    return {
        "TS_C": round(ts_c, 4),
        "TS_NC": round(ts_nc, 4),
        "BLEU_1": round(b1, 4),
        "MR": float(contains_modeling_register(generated_text)),
        **asdict(rates),
    }


def validate_generation(text: str, min_chars: int = 30) -> bool:
    if not text or len(text) < min_chars:
        return False
    if text.lstrip().startswith(("{", "```")):
        return False
    return True


# ============================================================
# 10. ORQUESTRADOR
# ============================================================

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_models(registry: dict[str, ModelSpec]) -> dict[str, str]:
    """Faz uma chamada mínima a cada modelo. Retorna {key: 'OK' | mensagem de erro}."""
    client = UnifiedLLMClient()
    status: dict[str, str] = {}
    for key, spec in registry.items():
        try:
            text = client.generate(spec, "Responda apenas com a palavra 'ok'.")
            status[key] = "OK" if text else "VAZIO"
            log.info("[validate] %s -> %s", key, status[key])
        except Exception as exc:
            status[key] = f"FALHA: {exc}"
            log.error("[validate] %s -> %s", key, status[key])
    return status


def run_experiment(
    use_llm_diagnostic: bool = False,
    sample_size: int | None = None,
    only_models: list[str] | None = None,
    validate_only: bool = False,
) -> pd.DataFrame:
    apply_seed()
    examples = load_petv11(sample_size=sample_size)
    corpus_tfidf = CorpusTfidf([ex.original_text for ex in examples])
    scorer = get_scorer()
    generator = M2TGenerator()

    selected = MODEL_REGISTRY
    if only_models:
        invalid = [k for k in only_models if k not in MODEL_REGISTRY]
        if invalid:
            raise ValueError(
                "Modelos inválidos: "
                + ", ".join(invalid)
                + "\nUse: python m2t_petv11_unico.py list-models"
            )
        selected = {k: MODEL_REGISTRY[k] for k in only_models}
    if not selected:
        raise ValueError("Nenhum modelo selecionado.")

    if validate_only:
        log.info("Modo --validate-only ativo; rodando ping em modelos.")
        status = validate_models(selected)
        for k, v in status.items():
            print(f"{k}: {v}")
        return pd.DataFrame()

    output_csv = OUTPUT_DIR / "results.csv"
    if output_csv.exists():
        try:
            output_csv.unlink()  # reset; cache de chamadas LLM ainda é reaproveitado
        except PermissionError as exc:
            log.error(
                "Não foi possível remover %s — feche o arquivo no Excel/editor "
                "e rode novamente. Detalhe: %s",
                output_csv, exc,
            )
            raise SystemExit(
                f"\n[ERRO] {output_csv} está aberto em outro programa.\n"
                f"Feche-o e rode `python m2t_petv11_unico.py experiment ...` de novo.\n"
            )

    all_rows: list[dict] = []

    for model_key, model_spec in selected.items():
        rows_this: list[dict] = []
        for feedback_condition in ["no_feedback", "with_feedback"]:
            iterator = tqdm(
                examples,
                desc=f"{model_spec.display_name} | {feedback_condition}",
            )
            for example in iterator:
                try:
                    if feedback_condition == "no_feedback":
                        result = generator.single_shot(model_spec, example.mermaid_lr)
                    else:
                        result = generator.self_refine(
                            model_spec,
                            example.mermaid_lr,
                            example.tasks,
                            use_llm_diagnostic=use_llm_diagnostic,
                        )
                except Exception as exc:
                    log.error(
                        "Geração falhou (modelo=%s, condição=%s, processo=%s): %s",
                        model_key, feedback_condition, example.process_id, exc,
                    )
                    continue

                output_valid = validate_generation(result.output)
                if not output_valid:
                    log.warning(
                        "Saída suspeita (modelo=%s, processo=%s): %r",
                        model_key, example.process_id, result.output[:80],
                    )

                metrics = evaluate_output(
                    expected_text=example.original_text,
                    expected_tasks=example.tasks,
                    generated_text=result.output,
                    scorer=scorer,
                    corpus_tfidf=corpus_tfidf,
                )

                gen_path = (
                    GEN_DIR / model_key / feedback_condition
                    / f"{example.process_id}.txt"
                )
                save_text(gen_path, result.output)

                if result.diagnostic:
                    diag_path = (
                        DIAG_DIR / model_key / feedback_condition
                        / f"{example.process_id}.json"
                    )
                    save_text(
                        diag_path,
                        json.dumps(result.diagnostic, ensure_ascii=False, indent=2),
                    )

                rows_this.append({
                    "process_id": example.process_id,
                    "model_key": model_key,
                    "model_name": model_spec.display_name,
                    "model_family": model_spec.family,
                    "provider": model_spec.provider,
                    "feedback": feedback_condition,
                    "used_feedback": result.used_feedback,
                    "output_valid": output_valid,
                    "diagnostic_parse_ok": result.diagnostic_parse_ok,
                    "generation_path": str(gen_path),
                    **metrics,
                })

        df_this = pd.DataFrame(rows_this)
        df_this.to_csv(
            output_csv,
            mode="a",
            header=not output_csv.exists(),
            index=False,
            encoding="utf-8-sig",
        )
        all_rows.extend(rows_this)
        log.info("Checkpoint salvo: %s (%d linhas para %s).",
                 output_csv, len(rows_this), model_key)

    df = pd.DataFrame(all_rows)
    log.info("Resultados finais: %s (%d linhas).", output_csv, len(df))
    return df


# ============================================================
# 11. ANÁLISE ESTATÍSTICA
# ============================================================

def run_factorial_anova(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if smf is None or anova_lm is None:
        raise ImportError("statsmodels não instalado.")
    n_models = df["model_key"].nunique()
    n_feedback = df["feedback"].nunique()
    if n_feedback < 2:
        raise ValueError("ANOVA requer ≥2 condições de feedback.")
    formula = (
        f"{metric} ~ C(model_key) * C(feedback)"
        if n_models >= 2
        else f"{metric} ~ C(feedback)"
    )
    model = smf.ols(formula, data=df).fit()
    return anova_lm(model, typ=2)


def run_lmm(df: pd.DataFrame, metric: str):
    if smf is None:
        raise ImportError("statsmodels não instalado.")
    n_models = df["model_key"].nunique()
    formula = (
        f"{metric} ~ C(model_key) * C(feedback)"
        if n_models >= 2
        else f"{metric} ~ C(feedback)"
    )
    return smf.mixedlm(formula, data=df, groups=df["process_id"]).fit(
        reml=False, method="lbfgs"
    )


def run_mr_glmm(df: pd.DataFrame):
    """GLM logístico para MR (variável Bernoulli)."""
    if smf is None or sm is None:
        raise ImportError("statsmodels não instalado.")
    n_models = df["model_key"].nunique()
    formula = (
        "MR ~ C(model_key) * C(feedback)"
        if n_models >= 2
        else "MR ~ C(feedback)"
    )
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()


def paired_wilcoxon_by_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for model_key, sub in df.groupby("model_key"):
        pivot = sub.pivot_table(
            index="process_id", columns="feedback", values=metric, aggfunc="mean"
        ).dropna()
        if not {"no_feedback", "with_feedback"}.issubset(pivot.columns):
            continue
        if len(pivot) < 2:
            continue
        diffs = pivot["with_feedback"] - pivot["no_feedback"]
        n_nonzero = int((diffs != 0).sum())
        if n_nonzero == 0:
            stat, p = float("nan"), float("nan")
            r_effect = 0.0
        else:
            try:
                stat, p = wilcoxon(pivot["no_feedback"], pivot["with_feedback"],
                                   zero_method="wilcox")
                # tamanho de efeito r = Z / sqrt(N), aproximação
                # Z aproximado a partir do p-valor bicaudal
                from scipy.stats import norm
                if 0 < p < 1:
                    z = abs(norm.isf(p / 2))
                    r_effect = float(z / np.sqrt(n_nonzero))
                else:
                    r_effect = float("nan")
            except ValueError as exc:
                log.warning("Wilcoxon skip (%s/%s): %s", model_key, metric, exc)
                stat, p, r_effect = float("nan"), float("nan"), float("nan")

        rows.append({
            "model_key": model_key,
            "metric": metric,
            "n": len(pivot),
            "n_nonzero": n_nonzero,
            "wilcoxon_stat": stat,
            "p_value": p,
            "effect_r": r_effect,
            "mean_no_feedback": float(pivot["no_feedback"].mean()),
            "mean_with_feedback": float(pivot["with_feedback"].mean()),
            "delta_mean": float(diffs.mean()),
        })
    return pd.DataFrame(rows)


def global_friedman_feedback(df: pd.DataFrame, metric: str) -> dict:
    pivot = df.pivot_table(
        index="process_id",
        columns=["model_key", "feedback"],
        values=metric,
        aggfunc="mean",
    ).dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {
            "metric": metric, "friedman_stat": None, "p_value": None,
            "n": pivot.shape[0], "k_conditions": pivot.shape[1],
        }
    arrays = [pivot[col].values for col in pivot.columns]
    stat, p = friedmanchisquare(*arrays)
    return {
        "metric": metric, "friedman_stat": float(stat), "p_value": float(p),
        "n": int(pivot.shape[0]), "k_conditions": int(pivot.shape[1]),
    }


def mcnemar_mr_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Teste de McNemar pareado para MR (binário) por modelo."""
    if mcnemar is None:
        return pd.DataFrame()
    rows = []
    for model_key, sub in df.groupby("model_key"):
        pivot = sub.pivot_table(
            index="process_id", columns="feedback", values="MR", aggfunc="max"
        ).dropna()
        if not {"no_feedback", "with_feedback"}.issubset(pivot.columns):
            continue
        nf = pivot["no_feedback"].astype(int).values
        wf = pivot["with_feedback"].astype(int).values
        b = int(((nf == 1) & (wf == 0)).sum())  # mudou de 1->0 com feedback
        c = int(((nf == 0) & (wf == 1)).sum())  # mudou de 0->1 com feedback
        n00 = int(((nf == 0) & (wf == 0)).sum())
        n11 = int(((nf == 1) & (wf == 1)).sum())
        if b + c == 0:
            stat, p = float("nan"), 1.0
        else:
            table = [[n11, b], [c, n00]]
            try:
                res = mcnemar(table, exact=(b + c) < 25, correction=True)
                stat, p = float(res.statistic), float(res.pvalue)
            except Exception as exc:
                log.warning("McNemar falhou (%s): %s", model_key, exc)
                stat, p = float("nan"), float("nan")
        rows.append({
            "model_key": model_key, "n": len(pivot),
            "b_improvement": b, "c_regression": c,
            "n00": n00, "n11": n11,
            "mcnemar_stat": stat, "p_value": p,
            "mr_rate_no_feedback": float(pivot["no_feedback"].mean()),
            "mr_rate_with_feedback": float(pivot["with_feedback"].mean()),
        })
    return pd.DataFrame(rows)


def run_analysis(input_path: str | Path) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Não encontrado: {input_path}")

    df = pd.read_csv(input_path)
    anova_tables, wilcoxon_tables, friedman_rows, lmm_blocks = [], [], [], []

    for metric in METRIC_NAMES:
        log.info("=== Métrica: %s ===", metric)

        # ANOVA — pulamos para MR (analisado via GLM logístico)
        if metric != "MR":
            try:
                anova = run_factorial_anova(df, metric)
                anova["metric"] = metric
                anova_tables.append(anova.reset_index().rename(columns={"index": "effect"}))
                log.info("ANOVA %s:\n%s", metric, anova)
            except Exception as exc:
                log.warning("ANOVA falhou (%s): %s", metric, exc)

            try:
                lmm = run_lmm(df, metric)
                lmm_blocks.append(f"\n\n### LMM — {metric}\n\n{lmm.summary()}\n")
            except Exception as exc:
                log.warning("LMM falhou (%s): %s", metric, exc)

        try:
            wilcoxon_tables.append(paired_wilcoxon_by_model(df, metric))
        except Exception as exc:
            log.warning("Wilcoxon falhou (%s): %s", metric, exc)

        try:
            friedman_rows.append(global_friedman_feedback(df, metric))
        except Exception as exc:
            log.warning("Friedman falhou (%s): %s", metric, exc)

    # MR via GLM logístico + McNemar
    try:
        glm_mr = run_mr_glmm(df)
        (OUTPUT_DIR / "glm_mr_results.txt").write_text(
            str(glm_mr.summary()), encoding="utf-8"
        )
        log.info("GLM logístico MR salvo.")
    except Exception as exc:
        log.warning("GLM MR falhou: %s", exc)

    try:
        mcnemar_df = mcnemar_mr_by_model(df)
        if not mcnemar_df.empty:
            mcnemar_df.to_csv(OUTPUT_DIR / "mcnemar_mr.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        log.warning("McNemar falhou: %s", exc)

    if anova_tables:
        pd.concat(anova_tables, ignore_index=True).to_csv(
            OUTPUT_DIR / "anova_results.csv", index=False, encoding="utf-8-sig"
        )
    if wilcoxon_tables:
        pd.concat(wilcoxon_tables, ignore_index=True).to_csv(
            OUTPUT_DIR / "wilcoxon_results.csv", index=False, encoding="utf-8-sig"
        )
    if friedman_rows:
        pd.DataFrame(friedman_rows).to_csv(
            OUTPUT_DIR / "friedman_results.csv", index=False, encoding="utf-8-sig"
        )
    if lmm_blocks:
        (OUTPUT_DIR / "lmm_results.txt").write_text(
            "\n".join(lmm_blocks), encoding="utf-8"
        )
    log.info("Análises salvas em %s", OUTPUT_DIR)


# ============================================================
# 12. PLOTS PARA O ARTIGO  (NOVO)
# ============================================================

def _setup_matplotlib():
    """Estilo IEEE-friendly: serif, sem grid agressivo, paleta colorblind-safe."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
    return plt


def _save_fig(plt, base_name: str) -> tuple[Path, Path]:
    pdf_path = FIGS_DIR / f"{base_name}.pdf"
    png_path = FIGS_DIR / f"{base_name}.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    plt.close()
    return pdf_path, png_path


def _significance_marker(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _short_model_label(key: str) -> str:
    table = {
        "claude_opus_4_7": "Opus 4.7",
        "claude_sonnet_4_6": "Sonnet 4.6",
        "claude_haiku_4_5": "Haiku 4.5",
        "gpt_5_5": "GPT-5.5",
        "gpt_5_4": "GPT-5.4",
        "gpt_4": "GPT-4",
        "deepseek_v4_pro": "DS Pro",
        "deepseek_v4_flash": "DS Flash",
    }
    return table.get(key, key)


def plot_kpi_distributions(df: pd.DataFrame) -> Path:
    """
    Grade 2x4 com boxplots: para cada KPI, distribuição por modelo × condição.
    Foco: visão geral comparativa.
    """
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), sharey=False)
    axes = axes.ravel()
    models = sorted(df["model_key"].unique())
    palette = ["#377eb8", "#e41a1c"]  # azul = no_feedback, vermelho = with_feedback

    for ax, metric in zip(axes, METRIC_NAMES):
        if metric == "MR":
            # MR é binário: barras de proporção (média) por (modelo, feedback)
            agg = (df.groupby(["model_key", "feedback"])["MR"]
                     .mean().reset_index())
            x_pos = []; heights = []; colors_bar = []; ticks_mr = []
            xv = 0
            for mk in models:
                for j, cond in enumerate(["no_feedback", "with_feedback"]):
                    row = agg[(agg.model_key == mk) & (agg.feedback == cond)]
                    if len(row):
                        x_pos.append(xv)
                        heights.append(float(row["MR"].iloc[0]))
                        colors_bar.append(palette[j])
                        if j == 0:
                            ticks_mr.append((xv + 0.5, _short_model_label(mk)))
                        xv += 1
                xv += 0.6
            ax.bar(x_pos, heights, width=0.8, color=colors_bar,
                   alpha=0.7, edgecolor="white", lw=0.4)
            ax.set_title("MR")
            ax.set_xticks([t[0] for t in ticks_mr])
            ax.set_xticklabels([t[1] for t in ticks_mr],
                               rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("taxa de MR (↓ melhor)")
            ax.set_ylim(0, max(0.05, max(heights) * 1.15) if heights else 1)
            continue

        positions, data, colors, ticks = [], [], [], []
        x = 0
        for m in models:
            for j, cond in enumerate(["no_feedback", "with_feedback"]):
                vals = df[(df.model_key == m) & (df.feedback == cond)][metric].dropna().values
                if len(vals):
                    positions.append(x)
                    data.append(vals)
                    colors.append(palette[j])
                    if j == 0:
                        ticks.append((x + 0.5, _short_model_label(m)))
                    x += 1
            x += 0.6  # espaço entre modelos

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data, positions=positions, widths=0.7, patch_artist=True,
            medianprops=dict(color="black", lw=1.0),
            flierprops=dict(marker="o", ms=2, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(metric)
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], rotation=40, ha="right", fontsize=7)
        if METRIC_HIGHER_IS_BETTER[metric]:
            ax.set_ylabel("score (↑ melhor)")
        else:
            ax.set_ylabel("taxa de erro (↓ melhor)")

    # legenda manual
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=palette[0], alpha=0.55, label="no_feedback"),
        Patch(facecolor=palette[1], alpha=0.55, label="with_feedback"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Distribuições por KPI — modelo × condição de feedback", y=1.00)
    fig.tight_layout()
    pdf, _ = _save_fig(plt, "fig_kpi_distributions")
    log.info("Figura salva: %s", pdf)
    return pdf


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                  rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(CONFIG.random_seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        boots[i] = sample.mean()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(diffs.mean()), lo, hi


def plot_paired_deltas(df: pd.DataFrame, wilcoxon_df: pd.DataFrame | None = None) -> Path:
    """
    Forest plot: Δ = (with_feedback − no_feedback) por (modelo, KPI), com IC95% bootstrap.
    Eixo positivo = ganho do feedback (para KPIs higher-is-better) ou
                    aumento do erro (para KPIs lower-is-better).
    """
    plt = _setup_matplotlib()
    models = sorted(df["model_key"].unique())
    # Encurta labels: claude_opus_4_7 -> "Cl Opus 4.7" etc.
    short_label = {
        "claude_opus_4_7": "Opus 4.7",
        "claude_sonnet_4_6": "Sonnet 4.6",
        "claude_haiku_4_5": "Haiku 4.5",
        "gpt_5_5": "GPT-5.5",
        "gpt_5_4": "GPT-5.4",
        "gpt_4": "GPT-4",
        "deepseek_v4_pro": "DS v4 Pro",
        "deepseek_v4_flash": "DS v4 Flash",
    }
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(CONFIG.random_seed)

    for ax, metric in zip(axes, METRIC_NAMES):
        means, los, his, labels, sigs = [], [], [], [], []
        for m in models:
            sub = df[df.model_key == m]
            pivot = sub.pivot_table(
                index="process_id", columns="feedback",
                values=metric, aggfunc="mean"
            ).dropna()
            if not {"no_feedback", "with_feedback"}.issubset(pivot.columns):
                continue
            diffs = (pivot["with_feedback"] - pivot["no_feedback"]).values
            mean, lo, hi = _bootstrap_ci(diffs, rng=rng)
            means.append(mean); los.append(lo); his.append(hi)
            labels.append(short_label.get(m, m))
            if wilcoxon_df is not None and not wilcoxon_df.empty:
                row = wilcoxon_df[(wilcoxon_df.model_key == m) & (wilcoxon_df.metric == metric)]
                sigs.append(_significance_marker(row["p_value"].iloc[0]) if len(row) else "")
            else:
                sigs.append("")

        y = np.arange(len(labels))
        errors = np.array([
            [m - lo for m, lo in zip(means, los)],
            [hi - m for m, hi in zip(means, his)],
        ])
        ax.errorbar(means, y, xerr=errors, fmt="o", color="#1f4e79",
                    ecolor="#1f4e79", capsize=3, ms=4, lw=1.2)
        ax.axvline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(metric, fontsize=10, pad=6)
        if METRIC_HIGHER_IS_BETTER[metric]:
            ax.set_xlabel("Δ (with − no)  →  ganho", fontsize=8)
        else:
            ax.set_xlabel("Δ (with − no)  →  +erro", fontsize=8)
        # marcadores de significância à direita do IC superior
        for yi, sig, hi_val in zip(y, sigs, his):
            if sig and sig != "ns":
                ax.text(hi_val, yi, "  " + sig, va="center", ha="left",
                        fontsize=8, color="darkred", fontweight="bold")

    fig.suptitle("Efeito pareado do feedback (Δ = with − no, IC95% bootstrap)",
                 y=1.00, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    pdf, _ = _save_fig(plt, "fig_paired_deltas")
    log.info("Figura salva: %s", pdf)
    return pdf


def plot_wilcoxon_heatmap(wilcoxon_df: pd.DataFrame) -> Path | None:
    """Heatmap modelo × KPI: cor = effect_r * sinal de Δ; anotação com asteriscos."""
    if wilcoxon_df is None or wilcoxon_df.empty:
        return None
    plt = _setup_matplotlib()
    pivot_eff = wilcoxon_df.pivot(index="model_key", columns="metric", values="effect_r")
    pivot_p = wilcoxon_df.pivot(index="model_key", columns="metric", values="p_value")
    pivot_delta = wilcoxon_df.pivot(index="model_key", columns="metric", values="delta_mean")

    # sinal informativo: para higher-is-better, +Δ é bom; para lower-is-better, +Δ é ruim
    sign = pivot_delta.copy()
    for col in sign.columns:
        if METRIC_HIGHER_IS_BETTER.get(col, True):
            sign[col] = np.sign(sign[col])
        else:
            sign[col] = -np.sign(sign[col])
    signed_eff = pivot_eff * sign

    # alinhar colunas para a ordem METRIC_NAMES
    cols = [c for c in METRIC_NAMES if c in signed_eff.columns]
    signed_eff = signed_eff[cols]
    pivot_p = pivot_p[cols]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(signed_eff))))
    vmax = max(0.6, float(np.nanmax(np.abs(signed_eff.values)) or 0.6))
    im = ax.imshow(signed_eff.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0)
    ax.set_yticks(range(len(signed_eff)))
    ax.set_yticklabels(signed_eff.index)

    for i, model_key in enumerate(signed_eff.index):
        for j, metric in enumerate(cols):
            r = signed_eff.loc[model_key, metric]
            p = pivot_p.loc[model_key, metric]
            if pd.isna(r):
                continue
            txt = f"{r:+.2f}\n{_significance_marker(p)}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color="black" if abs(r) < vmax * 0.6 else "white")
    ax.set_title("Tamanho de efeito (Wilcoxon r assinado pela direção desejável)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("efeito r assinado")
    fig.tight_layout()
    pdf, _ = _save_fig(plt, "fig_wilcoxon_heatmap")
    log.info("Figura salva: %s", pdf)
    return pdf


def _nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    CD = q_alpha * sqrt(k(k+1)/(6n)).
    q_alpha para alpha=0.05 (Demšar 2006), k = nº condições.
    """
    q05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 12: 3.268, 14: 3.354, 16: 3.426,
        18: 3.488, 20: 3.541,
    }
    q = q05.get(k, 3.541 if k > 20 else 1.960)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def plot_critical_difference(df: pd.DataFrame, metric: str = "TS_C") -> Path | None:
    """
    Diagrama de Diferença Crítica (Demšar 2006) para o KPI escolhido,
    considerando as 16 condições (8 modelos × 2 feedbacks) como tratamentos.
    """
    plt = _setup_matplotlib()
    pivot = df.pivot_table(
        index="process_id",
        columns=["model_key", "feedback"],
        values=metric,
        aggfunc="mean",
    ).dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        log.warning("CD diagram: dados insuficientes para %s.", metric)
        return None

    higher_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
    # ranks por linha (processo): 1 = melhor
    if higher_better:
        ranks = pivot.apply(lambda row: rankdata(-row.values), axis=1, result_type="broadcast")
    else:
        ranks = pivot.apply(lambda row: rankdata(row.values), axis=1, result_type="broadcast")

    avg_ranks = ranks.mean(axis=0).sort_values()
    n = pivot.shape[0]
    k = pivot.shape[1]
    cd = _nemenyi_critical_difference(k, n)

    # plot
    labels = ["__".join(map(str, c)) for c in avg_ranks.index]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.25 * k + 2)))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1, max(4, k * 0.2 + 1))
    ax.invert_yaxis()
    ax.axis("off")

    # eixo de ranks no topo
    for r in range(1, k + 1):
        ax.plot([r, r], [-0.1, 0.0], color="black", lw=0.6)
        ax.text(r, -0.25, str(r), ha="center", va="top", fontsize=8)
    ax.plot([1, k], [0, 0], color="black", lw=1)

    # linhas de cada tratamento
    half = (k + 1) / 2
    for i, (label, rank) in enumerate(zip(labels, avg_ranks.values)):
        side = "left" if rank <= half else "right"
        y_text = 0.6 + (i % (k // 2 + 1)) * 0.22
        ax.plot([rank, rank], [0, y_text], color="black", lw=0.6)
        if side == "left":
            ax.plot([rank, 0.6], [y_text, y_text], color="black", lw=0.6)
            ax.text(0.55, y_text, f"{label}  ({rank:.2f})",
                    ha="right", va="center", fontsize=7)
        else:
            ax.plot([rank, k + 0.4], [y_text, y_text], color="black", lw=0.6)
            ax.text(k + 0.45, y_text, f"({rank:.2f})  {label}",
                    ha="left", va="center", fontsize=7)

    # barra CD
    cd_y = -0.6
    ax.plot([1, 1 + cd], [cd_y, cd_y], color="black", lw=2)
    ax.plot([1, 1], [cd_y - 0.1, cd_y + 0.1], color="black", lw=2)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.1, cd_y + 0.1], color="black", lw=2)
    ax.text(1 + cd / 2, cd_y - 0.25, f"CD = {cd:.2f}",
            ha="center", va="top", fontsize=8)

    # cliques de não-significância (post-hoc Nemenyi)
    ranks_sorted = avg_ranks.values
    cliques: list[tuple[int, int]] = []
    i = 0
    while i < len(ranks_sorted):
        j = i
        while j + 1 < len(ranks_sorted) and (ranks_sorted[j + 1] - ranks_sorted[i]) <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i = j + 1 if j > i else i + 1

    clique_y = 0.05
    for idx, (a, b) in enumerate(cliques):
        y = clique_y + idx * 0.08
        ax.plot([ranks_sorted[a], ranks_sorted[b]], [y, y],
                color="darkred", lw=2.5, solid_capstyle="round")

    ax.set_title(
        f"Diagrama CD (Friedman + Nemenyi) — {metric}  |  n={n} processos, k={k} condições",
        fontsize=10,
    )
    fig.tight_layout()
    pdf, _ = _save_fig(plt, f"fig_cd_diagram_{metric}")
    log.info("Figura salva: %s", pdf)
    return pdf


def plot_error_breakdown(df: pd.DataFrame) -> Path:
    """Barras empilhadas: composição média de erros (O-OM, O-OT, O-OO, O-OG) por (modelo, feedback)."""
    plt = _setup_matplotlib()
    error_cols = ["O_OM", "O_OT", "O_OO", "O_OG"]
    agg = (
        df.groupby(["model_key", "feedback"])[error_cols]
        .mean()
        .reset_index()
    )
    models = sorted(agg["model_key"].unique())
    conditions = ["no_feedback", "with_feedback"]
    fig, ax = plt.subplots(figsize=(11, 4.5))

    bar_width = 0.38
    x = np.arange(len(models))
    palette = {
        "O_OM": "#1b9e77",
        "O_OT": "#d95f02",
        "O_OO": "#7570b3",
        "O_OG": "#e7298a",
    }

    for ci, cond in enumerate(conditions):
        bottom = np.zeros(len(models))
        offset = (ci - 0.5) * bar_width
        for col in error_cols:
            heights = []
            for m in models:
                row = agg[(agg.model_key == m) & (agg.feedback == cond)]
                heights.append(float(row[col].iloc[0]) if len(row) else 0.0)
            heights = np.array(heights)
            ax.bar(x + offset, heights, width=bar_width, bottom=bottom,
                   color=palette[col], edgecolor="white", lw=0.4,
                   label=col if ci == 0 else None,
                   alpha=0.85 if cond == "with_feedback" else 0.55)
            bottom += heights
        # rótulo da condição embaixo da barra
        for xi, m in enumerate(models):
            ax.text(xi + offset, -0.02,
                    "wf" if cond == "with_feedback" else "nf",
                    ha="center", va="top", fontsize=6, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_label(m) for m in models],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("taxa média de erro (proporção)")
    ax.set_title("Decomposição de erros por modelo (nf = no_feedback, wf = with_feedback)")
    ax.legend(loc="upper right", ncol=4, fontsize=7, frameon=False)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    fig.tight_layout()
    pdf, _ = _save_fig(plt, "fig_error_breakdown")
    log.info("Figura salva: %s", pdf)
    return pdf


def plot_mr_with_mcnemar(df: pd.DataFrame, mcnemar_df: pd.DataFrame | None) -> Path:
    """Barras: taxa de MR por (modelo, feedback) + asterisco de McNemar."""
    plt = _setup_matplotlib()
    agg = (
        df.groupby(["model_key", "feedback"])["MR"].mean().reset_index()
    )
    models = sorted(agg["model_key"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))

    bar_width = 0.38
    x = np.arange(len(models))
    palette = {"no_feedback": "#377eb8", "with_feedback": "#e41a1c"}

    for ci, cond in enumerate(["no_feedback", "with_feedback"]):
        offset = (ci - 0.5) * bar_width
        heights = [
            float(agg[(agg.model_key == m) & (agg.feedback == cond)]["MR"].iloc[0])
            if len(agg[(agg.model_key == m) & (agg.feedback == cond)]) else 0.0
            for m in models
        ]
        ax.bar(x + offset, heights, width=bar_width, color=palette[cond],
               alpha=0.75, label=cond, edgecolor="white", lw=0.4)
        for xi, h in enumerate(heights):
            if h > 0:
                ax.text(xi + offset, h + 0.01, f"{h:.2f}",
                        ha="center", fontsize=7)

    # marcadores de McNemar
    if mcnemar_df is not None and not mcnemar_df.empty:
        max_h = max(0.05, ax.get_ylim()[1])
        for xi, m in enumerate(models):
            row = mcnemar_df[mcnemar_df.model_key == m]
            if len(row):
                sig = _significance_marker(row["p_value"].iloc[0])
                ax.text(xi, max_h * 0.97, sig, ha="center", fontsize=10,
                        color="darkred")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_label(m) for m in models],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("taxa de MR (vazamento de jargão)")
    ax.set_title("Modeling Register Leakage por modelo × feedback (McNemar)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    pdf, _ = _save_fig(plt, "fig_mr_mcnemar")
    log.info("Figura salva: %s", pdf)
    return pdf


def run_plots(input_path: str | Path) -> list[Path]:
    """
    Subcomando `plots`. Lê results.csv (e wilcoxon_results.csv / mcnemar_mr.csv
    se existirem) e gera o conjunto completo de figuras.
    """
    setup_directories()
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    wilcoxon_path = OUTPUT_DIR / "wilcoxon_results.csv"
    mcnemar_path = OUTPUT_DIR / "mcnemar_mr.csv"

    if not wilcoxon_path.exists() or not mcnemar_path.exists():
        log.info("Resultados estatísticos ausentes; rodando análise primeiro.")
        run_analysis(input_path)

    wilcoxon_df = pd.read_csv(wilcoxon_path) if wilcoxon_path.exists() else None
    mcnemar_df = pd.read_csv(mcnemar_path) if mcnemar_path.exists() else None

    figs = []
    figs.append(plot_kpi_distributions(df))
    figs.append(plot_paired_deltas(df, wilcoxon_df))
    p = plot_wilcoxon_heatmap(wilcoxon_df) if wilcoxon_df is not None else None
    if p:
        figs.append(p)
    figs.append(plot_error_breakdown(df))
    figs.append(plot_mr_with_mcnemar(df, mcnemar_df))
    # CD diagrams para o KPI principal e mais um lower-is-better representativo
    for primary in ["TS_C", "O_OM"]:
        p = plot_critical_difference(df, metric=primary)
        if p:
            figs.append(p)

    log.info("Figuras geradas:\n  - " + "\n  - ".join(str(f) for f in figs))
    return figs


# ============================================================
# 13. INSPEÇÃO DE DATASET
# ============================================================

def inspect_dataset(sample_rows: int = 3) -> None:
    try:
        from datasets import load_dataset
        dataset = load_dataset(CONFIG.dataset_name, split=CONFIG.split)
        print(f"\n=== PETv11: split={CONFIG.split} | linhas={len(dataset)} ===")
        print(f"Colunas: {dataset.column_names}")
        print("\n=== Features ===")
        print(dataset.features)
        print("\n=== Primeiras linhas ===")
        for i in range(min(sample_rows, len(dataset))):
            row = dataset[i]
            print(f"\n--- Linha {i} ---")
            for k, v in row.items():
                s = str(v)
                if len(s) > 800:
                    s = s[:800] + "..."
                print(f"{k}: {s}")
    except Exception as exc:
        print(f"[Erro] inspect_dataset: {exc}")


# ============================================================
# 14. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline M2T PETv11 — versão 2.0 (com correções metodológicas)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("experiment", help="Roda o experimento fatorial 8×2.")
    exp.add_argument("--sample-size", type=int, default=None)
    exp.add_argument("--use-llm-diagnostic", action="store_true",
                     help="Usa o LLM para diagnosticar antes do self-refine.")
    exp.add_argument("--models", nargs="*", default=None,
                     help="Subconjunto de chaves (ex.: gpt_4 deepseek_v4_pro).")
    exp.add_argument("--validate-only", action="store_true",
                     help="Faz apenas ping nos modelos e sai.")

    ana = sub.add_parser("analysis", help="ANOVA, LMM, Wilcoxon, Friedman, GLM-MR, McNemar.")
    ana.add_argument("--input", default=str(OUTPUT_DIR / "results.csv"))

    plt_ = sub.add_parser("plots", help="Gera figuras IEEE-ready a partir de results.csv.")
    plt_.add_argument("--input", default=str(OUTPUT_DIR / "results.csv"))

    sub.add_parser("validate-models", help="Ping em todos os modelos do registry.")
    sub.add_parser("list-models", help="Lista modelos disponíveis.")
    sub.add_parser("inspect-dataset", help="Mostra colunas e amostras do PETv11.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "experiment":
        run_experiment(
            use_llm_diagnostic=args.use_llm_diagnostic,
            sample_size=args.sample_size,
            only_models=args.models,
            validate_only=args.validate_only,
        )
    elif args.command == "analysis":
        run_analysis(args.input)
    elif args.command == "plots":
        run_plots(args.input)
    elif args.command == "validate-models":
        for k, v in validate_models(MODEL_REGISTRY).items():
            print(f"{k}: {v}")
    elif args.command == "list-models":
        print("\nModelos disponíveis:\n")
        for key, spec in MODEL_REGISTRY.items():
            print(f"- {key}: {spec.display_name} | provider={spec.provider} | id={spec.model_id}")
    elif args.command == "inspect-dataset":
        inspect_dataset()


if __name__ == "__main__":
    main()
