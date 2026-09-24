"""
Agente Cartola FC — App Streamlit.

Rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Deploy gratuito: suba este arquivo + requirements.txt num repositório do
GitHub e conecte em https://share.streamlit.io (Streamlit Community Cloud).
"""
import json, os, time, unicodedata
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error

API = "https://api.cartola.globo.com"
CACHE = "cache_cartola"
POS = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
PROVAVEL = 7
FORMACOES = {
    "3-4-3": {1: 1, 2: 0, 3: 3, 4: 4, 5: 3, 6: 1},
    "3-5-2": {1: 1, 2: 0, 3: 3, 4: 5, 5: 2, 6: 1},
    "4-3-3": {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1},
    "4-4-2": {1: 1, 2: 2, 3: 2, 4: 4, 5: 2, 6: 1},
    "4-5-1": {1: 1, 2: 2, 3: 2, 4: 5, 5: 1, 6: 1},
    "5-3-2": {1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1},
    "5-4-1": {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 1},
}
FEATS = ["media_3", "media_5", "media_temp", "jogos", "casa", "cedido", "forca", "posicao_id",
         "rodadas_parado"]

# Grupos de scout (ação detalhada de cada jogador na partida) que viram features extras —
# um proxy do "porquê" da pontuação (parecido em espírito com xG, mas nativo do Cartola).
SCOUT_GRUPOS = {
    "finalizacoes": ("FT", "FD", "FF", "G"),   # volume de finalização/criação de gol
    "gols": ("G",),
    "assistencias": ("A",),
    "desarmes": ("DS",),
    "faltas_sofridas": ("FS",),
    "sg": ("SG",),                              # jogo sem sofrer gols (defesa/goleiro)
    "cartoes": ("CA", "CV"),
}
SCOUT_COLS = list(SCOUT_GRUPOS)
FEATS_SCOUT = [f"{c}_media5" for c in SCOUT_COLS]
FEATS_FORMA = ["time_forma5", "time_saldo5", "time_ppg_mando", "adv_forma5", "adv_saldo5", "adv_ppg_mando",
               "time_posicao_tabela", "time_pontos_temporada", "adv_posicao_tabela", "adv_pontos_temporada"]
# Limiar de "mitada": pontuação no top do percentil informado, por posição (mínimo absoluto de segurança).
MITADA_PERCENTIL = 0.85
MITADA_LIMIAR_MINIMO = 8.0
FEATS_MITADA_CTX = ["cedido_mitada"]
FEATS = FEATS + FEATS_SCOUT + FEATS_FORMA + FEATS_MITADA_CTX



def _agregar_scout(scout):
    scout = scout or {}
    valores = {}
    for nome, codigos in SCOUT_GRUPOS.items():
        if nome == "cartoes":
            valores[nome] = (scout.get("CA") or 0) + 2 * (scout.get("CV") or 0)
        else:
            valores[nome] = sum(scout.get(c) or 0 for c in codigos)
    return valores


# ---------------------------------------------------------------- coleta
def api_get(path, cache=True):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, path.strip("/").replace("/", "_") + ".json")
    if cache and os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    r = requests.get(API + path, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    d = r.json()
    if cache:
        with open(fn, "w") as f:
            json.dump(d, f)
    time.sleep(0.2)
    return d


def mandos(partidas):
    m = {}
    for p in partidas.get("partidas", []):
        if not p.get("valida", True):
            continue
        m[p["clube_casa_id"]] = (1, p["clube_visitante_id"])
        m[p["clube_visitante_id"]] = (0, p["clube_casa_id"])
    return m


# Nomes de campo candidatos pro placar oficial — a API não documenta isso oficialmente,
# então tentamos algumas variações conhecidas antes de desistir de uma partida.
CAMPOS_PLACAR = [
    ("placar_oficial_mandante", "placar_oficial_visitante"),
    ("placar_mandante", "placar_visitante"),
    ("gols_mandante", "gols_visitante"),
]


def _extrair_placar(p):
    for campo_m, campo_v in CAMPOS_PLACAR:
        if p.get(campo_m) is not None and p.get(campo_v) is not None:
            return p[campo_m], p[campo_v]
    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def resultados_clubes(rodada_atual):
    """Resultado (V/E/D) e gols de cada clube, rodada a rodada — usado para calcular a forma recente."""
    linhas = []
    for r in range(1, rodada_atual):
        try:
            partidas = api_get(f"/partidas/{r}")
        except Exception:
            continue
        for p in partidas.get("partidas", []):
            if not p.get("valida", True):
                continue
            gm, gv = _extrair_placar(p)
            if gm is None or gv is None:
                continue
            cm, cv = p["clube_casa_id"], p["clube_visitante_id"]
            pm = 3 if gm > gv else (1 if gm == gv else 0)
            pv = 3 if gv > gm else (1 if gm == gv else 0)
            linhas.append(dict(rodada=r, clube_id=cm, mandante=1, gols_pro=gm, gols_contra=gv, pontos_jogo=pm))
            linhas.append(dict(rodada=r, clube_id=cv, mandante=0, gols_pro=gv, gols_contra=gm, pontos_jogo=pv))
    return pd.DataFrame(linhas)


def form_clubes(resultados):
    """Para cada clube/rodada: forma recente (últimos 5 jogos), saldo de gols recente,
    aproveitamento médio (pontos por jogo) jogando em casa ou fora, e posição na tabela —
    tudo calculado só com jogos ANTERIORES à rodada (sem vazamento de informação)."""
    cols_saida = ["rodada", "clube_id", "forma5", "saldo5", "ppg_mando",
                  "posicao_tabela", "pontos_temporada"]
    if resultados.empty:
        return pd.DataFrame(columns=cols_saida)
    r = resultados.sort_values(["clube_id", "rodada"]).copy()
    r["saldo_jogo"] = r["gols_pro"] - r["gols_contra"]
    r["forma5"] = r.groupby("clube_id")["pontos_jogo"].transform(lambda s: s.shift().rolling(5, 1).mean())
    r["saldo5"] = r.groupby("clube_id")["saldo_jogo"].transform(lambda s: s.shift().rolling(5, 1).mean())
    r["ppg_mando"] = r.groupby(["clube_id", "mandante"])["pontos_jogo"].transform(
        lambda s: s.shift().expanding().mean())

    # Classificação (pontos corridos), critérios de desempate: pontos > saldo > gols pró.
    # Tudo usando só jogos ANTERIORES à rodada (shift), pra não vazar informação do futuro.
    r["pontos_acum"] = r.groupby("clube_id")["pontos_jogo"].transform(lambda s: s.shift().expanding().sum())
    r["saldo_acum"] = r.groupby("clube_id")["saldo_jogo"].transform(lambda s: s.shift().expanding().sum())
    r["gols_acum"] = r.groupby("clube_id")["gols_pro"].transform(lambda s: s.shift().expanding().sum())
    r["pontos_temporada"] = r["pontos_acum"]
    r["posicao_tabela"] = np.nan
    for _, idx in r.groupby("rodada").groups.items():
        sub = r.loc[idx]
        if sub["pontos_acum"].isna().all():
            continue  # primeira rodada: ainda não existe classificação
        sub = sub.sort_values(["pontos_acum", "saldo_acum", "gols_acum"], ascending=False)
        r.loc[sub.index, "posicao_tabela"] = range(1, len(sub) + 1)

    return r[cols_saida]


# ---------------------------------------------------------------- odds (contexto, opcional)
# Isso NÃO entra como feature treinada no modelo — só temos odds da rodada atual, não dá pra
# reconstruir odds de rodadas passadas pra treinar/validar direito. Entra só como painel
# informativo ao lado da escalação.
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "soccer_brazil_campeonato"


def obter_chave_odds():
    try:
        return st.secrets.get("ODDS_API_KEY")
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_odds_rodada(chave):
    if not chave:
        return None, "Chave da API de odds não configurada nos Secrets do Streamlit."
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds",
            params={"apiKey": chave, "regions": "eu,uk", "markets": "h2h", "oddsFormat": "decimal"},
            timeout=20,
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _normalizar_nome_time(nome):
    n = unicodedata.normalize("NFKD", nome or "").encode("ascii", "ignore").decode().lower()
    for lixo in (" fc", " ec", " sc", "-"):
        n = n.replace(lixo, " ")
    return " ".join(n.split())


def _casar_time(nome_odds, nomes_cartola):
    """Acha, entre os clubes do Cartola, qual bate com o nome usado pela API de odds."""
    alvo = _normalizar_nome_time(nome_odds)
    melhor, melhor_pontos = None, 0
    for clube_id, nome_c in nomes_cartola.items():
        n = _normalizar_nome_time(nome_c)
        pontos = 100 if n == alvo else (50 + min(len(n), len(alvo)) if (n in alvo or alvo in n) else 0)
        if pontos > melhor_pontos:
            melhor, melhor_pontos = clube_id, pontos
    return melhor if melhor_pontos >= 50 else None


def odds_por_clube(chave, nomes_cartola):
    """{clube_id: {'vitoria':P,'empate':P,'derrota':P}} pros jogos da rodada + times não identificados."""
    dados, erro = buscar_odds_rodada(chave)
    if erro or not dados:
        return {}, [], erro
    resultado, nao_casados = {}, []
    for evento in dados:
        home, away = evento.get("home_team"), evento.get("away_team")
        somas = {"home": [], "draw": [], "away": []}
        for bk in (evento.get("bookmakers") or []):
            for mercado in bk.get("markets", []):
                if mercado.get("key") != "h2h":
                    continue
                for outcome in mercado.get("outcomes", []):
                    preco = outcome.get("price")
                    if not preco:
                        continue
                    if outcome.get("name") == home:
                        somas["home"].append(preco)
                    elif outcome.get("name") == away:
                        somas["away"].append(preco)
                    else:
                        somas["draw"].append(preco)
        if not somas["home"] or not somas["away"]:
            continue
        probs_brutas = {"home": 1 / np.mean(somas["home"]), "away": 1 / np.mean(somas["away"])}
        if somas["draw"]:
            probs_brutas["draw"] = 1 / np.mean(somas["draw"])
        soma = sum(probs_brutas.values())
        probs = {k: v / soma for k, v in probs_brutas.items()}  # remove o overround da casa de apostas

        id_home, id_away = _casar_time(home, nomes_cartola), _casar_time(away, nomes_cartola)
        if id_home is None:
            nao_casados.append(home)
        if id_away is None:
            nao_casados.append(away)
        if id_home is not None:
            resultado[id_home] = {"vitoria": probs["home"], "empate": probs.get("draw", 0),
                                    "derrota": probs["away"]}
        if id_away is not None:
            resultado[id_away] = {"vitoria": probs["away"], "empate": probs.get("draw", 0),
                                    "derrota": probs["home"]}
    return resultado, nao_casados, None


@st.cache_data(ttl=3600, show_spinner=False)
def historico(rodada_atual):
    linhas = []
    for r in range(1, rodada_atual):
        try:
            pont = api_get(f"/atletas/pontuados/{r}")
            mando = mandos(api_get(f"/partidas/{r}"))
        except Exception:
            continue
        for aid, a in (pont.get("atletas") or {}).items():
            if a["clube_id"] not in mando:
                continue
            casa, adv = mando[a["clube_id"]]
            linha = dict(rodada=r, atleta_id=int(aid), clube_id=a["clube_id"],
                         posicao_id=a["posicao_id"], pontos=a["pontuacao"],
                         casa=casa, adversario=adv)
            linha.update(_agregar_scout(a.get("scout")))
            linhas.append(linha)
    return pd.DataFrame(linhas)


def buscar_times(nome):
    """Busca pública por nome de time no Cartola — não precisa de login/senha."""
    return api_get(f"/times?q={requests.utils.quote(nome)}", cache=False)


def extrair_id_slug(time_sel):
    """Tenta achar o id numérico e o slug do time, cobrindo variações de formato da resposta."""
    aninhado = time_sel.get("time") if isinstance(time_sel.get("time"), dict) else {}

    def _achar(chaves, fonte):
        for k in chaves:
            v = fonte.get(k)
            if v not in (None, ""):
                return v
        return None

    time_id = _achar(["time_id", "id"], time_sel) or _achar(["time_id", "id"], aninhado)
    slug = _achar(["slug"], time_sel) or _achar(["slug"], aninhado)
    try:
        time_id = int(time_id) if time_id is not None else None
    except (TypeError, ValueError):
        time_id = None
    return time_id, slug


@st.cache_data(ttl=3600, show_spinner=False)
def historico_meu_time(time_id, slug, rodada_atual):
    """Busca, rodada a rodada, a pontuação de um time específico.
    Tenta pelo id numérico e depois pelo slug, guardando o que deu errado na primeira
    tentativa para exibir como diagnóstico se nada funcionar."""
    linhas = []
    diagnostico = None
    rotas_base = []
    if time_id:
        rotas_base.append(("id", f"/time/id/{time_id}"))
    if slug:
        rotas_base.append(("slug", f"/time/slug/{slug}"))

    for r in range(1, rodada_atual):
        encontrado = False
        for tipo, base in rotas_base:
            try:
                d = api_get(f"{base}/{r}", cache=True)
            except Exception as e:
                if diagnostico is None:
                    diagnostico = {"tentativa": f"{base}/{r}", "erro": str(e)}
                continue
            if isinstance(d, dict) and d.get("pontos") is not None:
                esquema = d.get("esquema")
                linhas.append({
                    "rodada": r,
                    "pontos": float(d["pontos"]),
                    "patrimonio": d.get("patrimonio"),
                    "esquema": esquema.get("nome") if isinstance(esquema, dict) else esquema,
                })
                encontrado = True
                break
            elif diagnostico is None:
                diagnostico = {"tentativa": f"{base}/{r}", "resposta_recebida": d}
        if not encontrado:
            continue
    return pd.DataFrame(linhas), diagnostico


def rotular_mitada(h):
    """Marca, linha a linha, se aquela pontuação foi uma 'mitada' (top do percentil por
    posição). Usado tanto para treinar o classificador quanto para calcular o quanto cada
    adversário costuma CEDER mitadas."""
    h = h.copy()
    validas = h[h.pontos.notna()]
    limiares = validas.groupby("posicao_id")["pontos"].quantile(MITADA_PERCENTIL)
    limiares = limiares.clip(lower=MITADA_LIMIAR_MINIMO).to_dict()
    h["mitada"] = [
        (int(p >= limiares.get(pos, MITADA_LIMIAR_MINIMO)) if pd.notna(p) else np.nan)
        for p, pos in zip(h.pontos, h.posicao_id)
    ]
    return h


def features(h, forma_tab=None):
    h = h.sort_values(["atleta_id", "rodada"]).copy()
    g = h.groupby("atleta_id")["pontos"]
    h["media_3"] = g.transform(lambda s: s.shift().rolling(3, 1).mean())
    h["media_5"] = g.transform(lambda s: s.shift().rolling(5, 1).mean())
    h["media_temp"] = g.transform(lambda s: s.shift().expanding().mean())
    h["jogos"] = g.transform(lambda s: s.shift().notna().cumsum())

    # Rodadas desde a última vez que o jogador de fato entrou em campo — sem isso, um
    # jogador que sumiu do time (banco/lesão) continua "parecendo" em forma pelas médias
    # antigas. Isso não estraga a média (que já só considera jogos reais), só avisa o
    # modelo quando esses jogos reais ficaram velhos.
    h["ultima_jogada"] = h.groupby("atleta_id")["rodada"].transform(lambda s: s.shift())
    h["rodadas_parado"] = h["rodada"] - h["ultima_jogada"]

    for col in SCOUT_COLS:
        if col not in h.columns:
            h[col] = 0.0
        h[f"{col}_media5"] = h.groupby("atleta_id")[col].transform(
            lambda s: s.shift().rolling(5, 1).mean())

    ced = (h.groupby(["adversario", "posicao_id", "rodada"])["pontos"].mean()
           .reset_index().sort_values("rodada"))
    ced["cedido"] = ced.groupby(["adversario", "posicao_id"])["pontos"].transform(
        lambda s: s.shift().expanding().mean())
    h = h.merge(ced.drop(columns="pontos"), on=["adversario", "posicao_id", "rodada"], how="left")

    h = rotular_mitada(h)
    ced_mit = (h.groupby(["adversario", "posicao_id", "rodada"])["mitada"].mean()
               .reset_index().sort_values("rodada"))
    ced_mit["cedido_mitada"] = ced_mit.groupby(["adversario", "posicao_id"])["mitada"].transform(
        lambda s: s.shift().expanding().mean())
    h = h.merge(ced_mit.drop(columns="mitada"), on=["adversario", "posicao_id", "rodada"], how="left")

    fc = h.groupby(["clube_id", "rodada"])["pontos"].mean().reset_index().sort_values("rodada")
    fc["forca"] = fc.groupby("clube_id")["pontos"].transform(lambda s: s.shift().expanding().mean())
    h = h.merge(fc.drop(columns="pontos"), on=["clube_id", "rodada"], how="left")

    if forma_tab is not None and not forma_tab.empty:
        proprio = forma_tab.rename(columns={
            "forma5": "time_forma5", "saldo5": "time_saldo5", "ppg_mando": "time_ppg_mando",
            "posicao_tabela": "time_posicao_tabela", "pontos_temporada": "time_pontos_temporada"})
        h = h.merge(proprio[["clube_id", "rodada", "time_forma5", "time_saldo5", "time_ppg_mando",
                              "time_posicao_tabela", "time_pontos_temporada"]],
                    on=["clube_id", "rodada"], how="left")
        adversario_tab = forma_tab.rename(columns={
            "clube_id": "adversario", "forma5": "adv_forma5", "saldo5": "adv_saldo5",
            "ppg_mando": "adv_ppg_mando", "posicao_tabela": "adv_posicao_tabela",
            "pontos_temporada": "adv_pontos_temporada"})
        h = h.merge(adversario_tab[["adversario", "rodada", "adv_forma5", "adv_saldo5", "adv_ppg_mando",
                                     "adv_posicao_tabela", "adv_pontos_temporada"]],
                    on=["adversario", "rodada"], how="left")
    else:
        for col in FEATS_FORMA:
            h[col] = np.nan
    return h


def treinar(h):
    treino = h[h.pontos.notna() & (h.jogos >= 1)]
    if len(treino) < 300:
        return None, None
    ult = treino.rodada.max()
    tr, te = treino[treino.rodada <= ult - 3], treino[treino.rodada > ult - 3]
    validacao = None
    if len(tr) > 300 and len(te) > 50:
        m_val = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4)
        m_val.fit(tr[FEATS], tr.pontos)
        mae_m = mean_absolute_error(te.pontos, m_val.predict(te[FEATS]))
        mae_b = mean_absolute_error(te.pontos, te.media_temp.fillna(te.pontos.mean()))
        validacao = (mae_m, mae_b)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4)
    m.fit(treino[FEATS], treino.pontos)
    return m, validacao


def treinar_mitada(h):
    """Classificador de 'índice mitada': probabilidade do jogador ter uma pontuação muito
    acima do normal (top do percentil configurado, por posição) na próxima rodada."""
    dados = h[h.pontos.notna() & (h.jogos >= 1)].copy()
    if len(dados) < 300 or "mitada" not in dados.columns:
        return None
    dados = dados.dropna(subset=["mitada"])
    if dados["mitada"].nunique() < 2:
        return None  # sem exemplos suficientes de ambas as classes ainda
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4)
    m.fit(dados[FEATS], dados["mitada"])
    return m


def otimizar(df, cartoletas, formacao, mult_cap):
    n = len(df)
    pred, preco, pos = df.pred.values, df.preco_num.values, df.posicao_id.values
    obj = -np.concatenate([pred, (mult_cap - 1) * pred])
    A, lb, ub = [], [], []
    A.append(np.concatenate([preco, np.zeros(n)])); lb.append(0); ub.append(cartoletas)
    for p, q in FORMACOES[formacao].items():
        A.append(np.concatenate([(pos == p).astype(float), np.zeros(n)])); lb.append(q); ub.append(q)
    A.append(np.concatenate([np.zeros(n), np.ones(n)])); lb.append(1); ub.append(1)
    lig = np.hstack([-np.eye(n), np.eye(n)])
    A = np.vstack([np.array(A), lig])
    lb = np.concatenate([lb, np.full(n, -np.inf)]); ub = np.concatenate([ub, np.zeros(n)])
    sup = np.concatenate([np.ones(n), (pos != 6).astype(float)])
    res = milp(obj, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(2 * n), bounds=Bounds(0, sup))
    if not res.success:
        return None
    x, c = res.x[:n] > 0.5, res.x[n:] > 0.5
    time_ = df[x].copy()
    time_["capitao"] = c[x]
    return time_, -res.fun


def comparativo_rodadas(hist_time, rodada_atual, cartoletas_padrao, formacao_escolhida, mult_cap, n_rodadas):
    """Para cada uma das últimas `n_rodadas`, recalcula o que o modelo teria escalado
    (treinando só com dados até a rodada anterior) e o time ideal em retrospecto (sabendo
    o resultado real), e compara com a pontuação que o usuário de fato fez."""
    elegiveis = hist_time[hist_time.rodada >= 2].sort_values("rodada").tail(n_rodadas)
    formas = list(FORMACOES) if formacao_escolhida == "auto" else [formacao_escolhida]
    linhas_resultado = []

    for _, linha_usuario in elegiveis.iterrows():
        r = int(linha_usuario.rodada)
        try:
            pont = api_get(f"/atletas/pontuados/{r}")
            mando_r = mandos(api_get(f"/partidas/{r}"))
        except Exception:
            continue

        registros = []
        for aid, a in (pont.get("atletas") or {}).items():
            if a["clube_id"] not in mando_r:
                continue
            casa, adv = mando_r[a["clube_id"]]
            registros.append(dict(atleta_id=int(aid), clube_id=a["clube_id"], posicao_id=a["posicao_id"],
                                   pontos_real=float(a["pontuacao"]), casa=casa, adversario=adv,
                                   preco_num=a.get("preco_num")))
        rodada_df = pd.DataFrame(registros)
        if rodada_df.empty:
            continue
        if rodada_df["preco_num"].isna().all():
            # essa rodada não trouxe preço histórico: usa o preço atual como aproximação
            rodada_df = rodada_df.drop(columns=["preco_num"]).merge(
                atl_disp[["atleta_id", "preco_num"]], on="atleta_id", how="left")
        rodada_df = rodada_df.dropna(subset=["preco_num"]).reset_index(drop=True)
        if rodada_df.empty:
            continue

        h_treino = historico(r)
        forma_tab_r = form_clubes(resultados_clubes(r))
        alvo_rows = rodada_df[["atleta_id", "clube_id", "posicao_id", "casa", "adversario"]].assign(
            rodada=r, pontos=np.nan)
        tudo_r = features(pd.concat([h_treino, alvo_rows], ignore_index=True), forma_tab_r)
        alvo_feats = tudo_r[tudo_r.rodada == r].set_index("atleta_id")
        modelo_r, _ = treinar(tudo_r[tudo_r.rodada < r])

        rodada_df = rodada_df.set_index("atleta_id")
        if modelo_r is None:
            continue  # histórico curto demais nessa rodada pra treinar algo minimamente confiável
        rodada_df["pred"] = modelo_r.predict(alvo_feats.loc[rodada_df.index, FEATS])
        rodada_df = rodada_df.reset_index()

        orcamento = linha_usuario.patrimonio if pd.notna(linha_usuario.get("patrimonio")) else cartoletas_padrao

        def _melhor(df_otim):
            melhor = None
            for f in formas:
                res = otimizar(df_otim, orcamento, f, mult_cap)
                if res and (melhor is None or res[1] > melhor[1]):
                    melhor = res
            return melhor

        melhor_modelo = _melhor(rodada_df)
        rodada_ideal = rodada_df.copy()
        rodada_ideal["pred"] = rodada_ideal["pontos_real"]
        melhor_ideal = _melhor(rodada_ideal)

        def _pontos_reais_do_time(resultado):
            if not resultado:
                return None
            time_, _ = resultado
            extra_capitao = time_.loc[time_.capitao, "pontos_real"].sum() * (mult_cap - 1)
            return float(time_.pontos_real.sum() + extra_capitao)

        linhas_resultado.append({
            "rodada": r,
            "Você": float(linha_usuario.pontos),
            "Modelo": _pontos_reais_do_time(melhor_modelo),
            "Ideal da rodada": _pontos_reais_do_time(melhor_ideal),
        })

    return pd.DataFrame(linhas_resultado)


@st.cache_data(ttl=1800, show_spinner=False)
def montar_base():
    """Baixa mercado, partidas e histórico; devolve dataframe de atletas com previsão pronta."""
    status = api_get("/mercado/status", cache=False)
    rodada = status["rodada_atual"]
    aberto = status.get("status_mercado") == 1

    merc = api_get("/atletas/mercado", cache=False)
    clubes = {int(k): v["abreviacao"] for k, v in merc["clubes"].items()}
    escudos = {int(k): (v.get("escudos") or {}).get("60x60") for k, v in merc["clubes"].items()}
    nomes_clubes = {int(k): v.get("nome", v["abreviacao"]) for k, v in merc["clubes"].items()}
    atl = pd.DataFrame(merc["atletas"])
    atl = atl[atl.status_id == PROVAVEL].copy()
    if "foto" in atl.columns:
        atl["foto_url"] = atl["foto"].apply(
            lambda u: u.replace("FORMATO", "140x140") if isinstance(u, str) else None)
    else:
        atl["foto_url"] = None
    mando = mandos(api_get("/partidas", cache=False))
    atl = atl[atl.clube_id.isin(mando.keys())].copy()
    atl["casa"] = atl.clube_id.map(lambda c: mando[c][0])
    atl["adversario"] = atl.clube_id.map(lambda c: mando[c][1])

    h = historico(rodada)
    forma_tab = form_clubes(resultados_clubes(rodada))
    prox = atl[["atleta_id", "clube_id", "posicao_id", "casa", "adversario"]].assign(
        rodada=rodada, pontos=np.nan)
    tudo = features(pd.concat([h, prox], ignore_index=True), forma_tab)
    alvo = tudo[tudo.rodada == rodada].set_index("atleta_id")

    modelo, validacao = treinar(tudo[tudo.rodada < rodada])
    modelo_mitada = treinar_mitada(tudo[tudo.rodada < rodada])
    atl = atl.set_index("atleta_id")
    if modelo is None:
        atl["pred"] = atl.media_num
    else:
        atl["pred"] = modelo.predict(alvo.loc[atl.index, FEATS])
        sem_jogo = alvo.loc[atl.index, "jogos"].fillna(0).values == 0
        atl.loc[sem_jogo, "pred"] = atl.loc[sem_jogo, "media_num"] * 0.8
    if modelo_mitada is not None:
        atl["prob_mitada"] = modelo_mitada.predict_proba(alvo.loc[atl.index, FEATS])[:, 1]
    else:
        atl["prob_mitada"] = np.nan
    atl = atl.reset_index()
    atl["clube_abrev"] = atl.clube_id.map(clubes)
    atl["clube_escudo"] = atl.clube_id.map(escudos)
    atl["clube_nome"] = atl.clube_id.map(nomes_clubes)
    atl["adversario_abrev"] = atl.adversario.map(clubes)
    atl["adversario_escudo"] = atl.adversario.map(escudos)
    atl["posicao"] = atl.posicao_id.map(POS)

    ultimos_map = {}
    if not h.empty:
        h_sorted = h.sort_values("rodada")
        for atleta_id, grupo in h_sorted.groupby("atleta_id"):
            recentes = grupo.tail(5)
            ultimos_map[int(atleta_id)] = [
                {"rodada": int(r.rodada), "pontos": round(float(r.pontos), 1),
                 "adversario_abrev": clubes.get(int(r.adversario), "?")}
                for r in recentes.itertuples()
            ]
    atl["hist_pontos"] = atl["atleta_id"].map(ultimos_map)
    atl["hist_pontos"] = atl["hist_pontos"].apply(lambda v: v if isinstance(v, list) else [])
    return atl, rodada, aberto, validacao, mando


def cartao_jogador(row, largura=148):
    """Card estilo 'carta de jogador' (dourado), com foto, preço, previsão e histórico recente."""
    cap_badge = ' <span style="color:#8a0000;">🅲</span>' if bool(row.get("capitao", False)) else ""
    foto = row.get("foto_url")
    escudo = row.get("clube_escudo")
    adv_escudo = row.get("adversario_escudo")
    hist = row.get("hist_pontos") or []
    hist = [j for j in hist if isinstance(j, dict) and "pontos" in j]

    prob_mitada = row.get("prob_mitada")
    if prob_mitada is not None and pd.notna(prob_mitada):
        mitada_html = (f'<span title="Chance de pontuação muito acima do normal">'
                        f'🔥 {prob_mitada * 100:.0f}%</span>')
    else:
        mitada_html = ""

    foto_html = (
        f'<img src="{foto}" style="width:64px;height:64px;border-radius:50%;object-fit:cover;'
        f'border:3px solid #4a2f0a;box-shadow:0 2px 6px rgba(0,0,0,.4);">' if foto else
        '<div style="width:64px;height:64px;border-radius:50%;background:#3a2a12;'
        'display:flex;align-items:center;justify-content:center;font-size:26px;'
        'border:3px solid #4a2f0a;">👤</div>'
    )
    escudo_html = f'<img src="{escudo}" width="24" style="vertical-align:middle;">' if escudo else ""
    adv_escudo_html = (f'<img src="{adv_escudo}" width="16" style="vertical-align:middle;margin-right:3px;">'
                        if adv_escudo else "")

    if hist:
        maior = max(3.0, max(abs(j["pontos"]) for j in hist))
        barras = "".join(
            f'<div style="width:8px;height:{max(4, int(abs(j["pontos"]) / maior * 22))}px;'
            f'background:{"#1b8a3a" if j["pontos"] >= 0 else "#b3261e"};border-radius:2px;"></div>'
            for j in hist
        )
        linhas_tooltip = "".join(
            f'<div style="display:flex;justify-content:space-between;gap:10px;padding:3px 0;'
            f'border-bottom:1px solid #3a2a12;"><span>R{j["rodada"]} · x {j["adversario_abrev"]}</span>'
            f'<span style="font-weight:700;color:{"#7ddc8c" if j["pontos"] >= 0 else "#ff8a80"};">'
            f'{j["pontos"]:.1f}</span></div>'
            for j in reversed(hist)
        )
        hist_html = f"""
            <div class="mini-hist">
                <div style="display:flex;gap:3px;justify-content:center;align-items:flex-end;
                            height:24px;margin-top:6px;">{barras}</div>
                <div style="font-size:9px;color:#5a3c0a;margin-top:1px;">últimos {len(hist)} jogos ⓘ</div>
                <div class="mini-tooltip">
                    <div style="font-weight:700;margin-bottom:4px;text-align:center;">
                        Últimos {len(hist)} jogos</div>
                    {linhas_tooltip}
                </div>
            </div>
        """
    else:
        hist_html = '<div style="font-size:10px;color:#5a3c0a;margin-top:6px;">sem histórico ainda</div>'

    cartao_html = f"""
        <div style="
            width:{largura}px;margin:0 auto 6px auto;border-radius:14px;
            background:linear-gradient(160deg,#f7dd8f 0%,#e8b93f 42%,#b8790a 100%);
            border:2px solid #7a4e0a;box-shadow:0 6px 16px rgba(0,0,0,.5);
            padding:8px 6px 8px 6px;text-align:center;color:#2b1900;">
            <div style="font-weight:800;">
                <span style="background:#2b1900;color:#f7dd8f;border-radius:6px;
                             padding:1px 6px;font-size:11px;">{row.posicao}</span>
            </div>
            <div style="margin-top:2px;">{foto_html}</div>
            <div style="font-weight:700;font-size:12px;text-transform:uppercase;
                        margin-top:3px;line-height:1.15;">
                {escudo_html} {row.apelido}{cap_badge}
            </div>
            <div style="font-size:11px;color:#4a2f0a;margin-top:2px;">
                {adv_escudo_html}x {row.adversario_abrev} ({"casa" if row.casa else "fora"})
            </div>
            <div style="display:flex;justify-content:space-around;margin-top:5px;
                        font-size:11px;font-weight:700;border-top:1px solid #7a4e0a;padding-top:5px;">
                <span title="Preço na rodada">💰 C$ {row.preco_num:.1f}</span>
                <span title="Previsão do modelo">📈 {row.pred:.1f} pts</span>
                {mitada_html}
            </div>
            {hist_html}
        </div>
    """
    # Achata tudo numa única linha: o Markdown do Streamlit interpreta HTML indentado
    # em várias linhas como bloco de código, então isso evita esse problema.
    cartao_html = " ".join(l.strip() for l in cartao_html.strip().splitlines() if l.strip())
    st.markdown(cartao_html, unsafe_allow_html=True)


# ============================== INTERFACE ==============================
st.set_page_config(page_title="Agente Cartola", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { gap: 0.3rem !important; }
    .mini-hist { position: relative; display: inline-block; cursor: help; }
    .mini-hist .mini-tooltip {
        visibility: hidden; opacity: 0; transition: opacity .15s ease;
        position: absolute; bottom: 115%; left: 50%; transform: translateX(-50%);
        background: #1b120a; color: #f7dd8f; border: 1px solid #7a4e0a; border-radius: 10px;
        padding: 10px 12px; width: 210px; z-index: 999;
        box-shadow: 0 10px 24px rgba(0,0,0,.6); text-align: left; font-size: 11px;
        pointer-events: none;
    }
    .mini-hist:hover .mini-tooltip { visibility: visible; opacity: 1; }

    /* "Campo de futebol" por trás da escalação montada */
    .st-key-campo_futebol {
        background:
            repeating-linear-gradient(180deg, #1c5c2e 0px, #1c5c2e 48px, #1e6432 48px, #1e6432 96px);
        border: 3px solid rgba(255,255,255,.6);
        border-radius: 16px;
        padding: 16px 14px 20px 14px;
    }
    .st-key-campo_futebol .stCaption, .st-key-campo_futebol p {
        color: #eaf4ec !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="
        background: repeating-linear-gradient(90deg, #1c3d24 0px, #1c3d24 40px,
                    #204826 40px, #204826 80px);
        border-radius: 14px; padding: 18px 20px; margin-bottom: 18px;
        border: 1px solid #3a5c3f;">
        <h1 style="margin:0;color:#f0f2f0;">⚽ Agente Cartola FC</h1>
        <p style="margin:2px 0 0 0;color:#c8d6c9;font-size:13px;">
            Escalação automática e montagem manual com modelo preditivo
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuração")
    cartoletas = st.number_input("Cartoletas disponíveis (C$)", min_value=50.0, max_value=300.0,
                                  value=120.0, step=0.5)
    formacao = st.selectbox("Formação", ["auto"] + list(FORMACOES))
    mult_cap = st.slider("Multiplicador do capitão", 1.0, 2.0, 1.5, 0.1)

with st.spinner("Baixando dados do Cartola e calculando previsões..."):
    atl, rodada, aberto, validacao, mando = montar_base()

with st.sidebar:
    excluir_nomes = st.multiselect("Excluir jogadores específicos (opcional)",
                                    options=sorted(atl.apelido.unique()))

st.caption(f"Rodada {rodada} — mercado {'ABERTO' if aberto else 'FECHADO'}")
if validacao:
    mae_m, mae_b = validacao
    st.caption(f"Validação (últimas 3 rodadas): erro do modelo {mae_m:.2f} pts "
               f"vs. {mae_b:.2f} pts usando só a média da temporada.")

atl_disp = atl[~atl.apelido.isin(excluir_nomes)].copy()

tab_auto, tab_manual, tab_meu = st.tabs(
    ["🤖 Escalação automática", "🛠️ Montar manualmente", "📊 Meu desempenho"])

# ============================== ABA 1 — AUTOMÁTICA ==============================
with tab_auto:
    with st.expander("📊 Contexto de mercado (odds) — opcional"):
        chave_odds = obter_chave_odds()
        if not chave_odds:
            st.caption(
                "Ainda não configurado. No painel do seu app no Streamlit Cloud, vá em "
                "**Settings → Secrets** e cole (trocando pela sua chave real):"
            )
            st.code('ODDS_API_KEY = "sua-chave-aqui"', language="toml")
            st.caption(
                "Isso fica guardado separado do código-fonte, então não aparece no GitHub. "
                "Esse contexto é só informativo — não entra no modelo treinado, porque só "
                "temos odds da rodada atual, não do passado, pra validar direito."
            )
        else:
            nomes_clubes_map = atl_disp.drop_duplicates("clube_id").set_index("clube_id")["clube_nome"].to_dict()
            escudos_map = atl_disp.drop_duplicates("clube_id").set_index("clube_id")["clube_escudo"].to_dict()
            probs_odds, nao_casados, erro_odds = odds_por_clube(chave_odds, nomes_clubes_map)
            if erro_odds:
                st.warning(f"Não consegui buscar as odds agora: {erro_odds}")
            elif not probs_odds:
                st.caption("Sem odds encontradas pra essa rodada no momento.")
            else:
                linhas_odds = [{
                    "Time": escudos_map.get(cid), "Vitória": p["vitoria"] * 100,
                    "Empate": p["empate"] * 100, "Derrota": p["derrota"] * 100,
                } for cid, p in probs_odds.items()]
                st.dataframe(
                    pd.DataFrame(linhas_odds), use_container_width=True, hide_index=True,
                    column_config={
                        "Time": st.column_config.ImageColumn("Time", width="small"),
                        "Vitória": st.column_config.ProgressColumn("Vitória", format="%.0f%%", min_value=0, max_value=100),
                        "Empate": st.column_config.ProgressColumn("Empate", format="%.0f%%", min_value=0, max_value=100),
                        "Derrota": st.column_config.ProgressColumn("Derrota", format="%.0f%%", min_value=0, max_value=100),
                    },
                )
                if nao_casados:
                    st.caption(f"Times que a odds trouxe mas eu não identifiquei: {', '.join(set(nao_casados))}")

    forms = list(FORMACOES) if formacao == "auto" else [formacao]
    melhor = None
    for f in forms:
        r = otimizar(atl_disp, cartoletas, f, mult_cap)
        if r and (melhor is None or r[1] > melhor[1]):
            melhor = (r[0], r[1], f)

    if melhor is None:
        st.error("Nenhuma escalação possível com esse orçamento/formação.")
    else:
        time_, total, f = melhor
        escalados = time_.sort_values(["posicao_id", "pred"], ascending=[True, False])

        st.subheader(f"Formação {f}")
        m1, m2 = st.columns(2)
        m1.metric("Pontuação prevista", f"{total:.1f} pts")
        m2.metric("Custo total", f"C$ {escalados.preco_num.sum():.2f}")

        with st.container(key="campo_futebol"):
            # Ordem visual: ataque no topo (perto do gol adversário) até o goleiro embaixo
            # (perto do próprio gol) — igual um campo de futebol de verdade, de cima pra baixo.
            for pos_id in [5, 4, 3, 2]:  # ATA, MEI, ZAG, LAT
                linha_pos = escalados[escalados.posicao_id == pos_id]
                if linha_pos.empty:
                    continue
                st.caption(f"**{POS[pos_id]}**")
                cols = st.columns(len(linha_pos))
                for col, (_, row) in zip(cols, linha_pos.iterrows()):
                    with col:
                        cartao_jogador(row)

            # Goleiro e técnico juntos na última linha, técnico à direita
            gol_linha = escalados[escalados.posicao_id == 1]
            tec_linha = escalados[escalados.posicao_id == 6]
            if not gol_linha.empty or not tec_linha.empty:
                st.markdown(
                    '<div style="border-top:3px dashed rgba(255,255,255,.5);'
                    'margin:4px 0 10px 0;"></div>',
                    unsafe_allow_html=True,
                )
                st.caption("**GOL / TEC**")
                cols = st.columns(len(gol_linha) + len(tec_linha) or 1)
                i = 0
                for _, row in gol_linha.iterrows():
                    with cols[i]:
                        cartao_jogador(row)
                    i += 1
                for _, row in tec_linha.iterrows():
                    with cols[i]:
                        cartao_jogador(row)
                    i += 1

        csv = escalados.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV da escalação", csv, f"escalacao_rodada_{rodada}.csv")

    st.divider()
    st.subheader("Previsão jogador a jogador")
    st.caption("Todos os jogadores prováveis, com a pontuação que o modelo espera para a rodada.")
    pos_filtro = st.multiselect("Filtrar posição", options=list(POS.values()),
                                 default=list(POS.values()), key="filtro_auto")
    tabela = atl_disp[atl_disp.posicao.isin(pos_filtro)][
        ["clube_escudo", "apelido", "posicao", "adversario_escudo", "casa",
         "preco_num", "media_num", "pred", "prob_mitada"]
    ].rename(columns={
        "clube_escudo": "Time", "apelido": "Jogador", "posicao": "Pos", "adversario_escudo": "Contra",
        "casa": "Mando", "preco_num": "Preço", "media_num": "Média Cartola", "pred": "Previsão (modelo)",
        "prob_mitada": "Chance de mitar",
    }).sort_values("Previsão (modelo)", ascending=False)
    tabela["Mando"] = tabela["Mando"].map({1: "Casa", 0: "Fora"})
    tabela["Chance de mitar"] = tabela["Chance de mitar"] * 100
    st.dataframe(
        tabela, use_container_width=True, hide_index=True,
        column_config={
            "Time": st.column_config.ImageColumn("Time", width="small"),
            "Contra": st.column_config.ImageColumn("Contra", width="small"),
            "Chance de mitar": st.column_config.ProgressColumn(
                "Chance de mitar", format="%.0f%%", min_value=0, max_value=100),
        },
    )

# ============================== ABA 2 — MANUAL ==============================
with tab_manual:
    st.subheader("Monte seu time jogador a jogador")
    formacao_manual = st.selectbox("Formação", list(FORMACOES), key="formacao_manual")
    contagem = FORMACOES[formacao_manual]
    total_slots = sum(contagem.values())
    atl_idx = atl_disp.set_index("atleta_id")

    def _coletar_previa():
        """Lê o que já está selecionado (session_state) antes de redesenhar os seletores,
        só para conseguir mostrar o resumo lá em cima."""
        linhas = []
        for pos_id, qtd in contagem.items():
            for i in range(qtd):
                val = st.session_state.get(f"manual_{formacao_manual}_{pos_id}_{i}")
                if val is not None and val in atl_idx.index:
                    linhas.append(atl_idx.loc[val])
        return linhas

    escolhidos_previa = _coletar_previa()
    cap_id_previa = st.session_state.get(f"capitao_manual_{formacao_manual}")

    # ---- Resumo no topo ----
    custo_previa = sum(e.preco_num for e in escolhidos_previa)
    pontos_previa = sum(e.pred for e in escolhidos_previa)
    if cap_id_previa is not None:
        cap_row = next((e for e in escolhidos_previa
                         if int(e.atleta_id) == cap_id_previa and e.posicao_id != 6), None)
        if cap_row is not None:
            pontos_previa += cap_row.pred * (mult_cap - 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Jogadores escalados", f"{len(escolhidos_previa)}/{total_slots}")
    restante = cartoletas - custo_previa
    c2.metric("Custo total", f"C$ {custo_previa:.2f}",
               delta=f"C$ {restante:.2f} livres" if restante >= 0 else f"estourou C$ {-restante:.2f}",
               delta_color="normal" if restante >= 0 else "inverse")
    c3.metric("Pontuação prevista", f"{pontos_previa:.2f} pts")
    if escolhidos_previa and len(escolhidos_previa) < total_slots:
        st.info(f"Faltam {total_slots - len(escolhidos_previa)} jogador(es) para completar o time.")
    if custo_previa > cartoletas:
        st.warning(f"Esse time estoura seu orçamento de C$ {cartoletas:.2f} em C$ {custo_previa - cartoletas:.2f}.")

    st.divider()

    escolhidos = []  # lista de Series (linhas de atl_disp) já escolhidas, na ordem
    for pos_id, qtd in contagem.items():
        if qtd == 0:
            continue
        st.markdown(f"**{POS[pos_id]}**")
        cols = st.columns(qtd)
        for i in range(qtd):
            with cols[i]:
                ja_escolhidos_ids = [int(e.atleta_id) for e in escolhidos]
                opcoes = (atl_disp[(atl_disp.posicao_id == pos_id) &
                                    (~atl_disp.atleta_id.isin(ja_escolhidos_ids))]
                          .sort_values("pred", ascending=False))
                lookup = {int(r.atleta_id): r for _, r in opcoes.iterrows()}
                ids = [None] + list(lookup.keys())
                escolha_id = st.selectbox(
                    f"{POS[pos_id]} {i + 1}", ids,
                    format_func=lambda x: "— selecione —" if x is None
                    else f"{lookup[x].apelido} ({lookup[x].clube_abrev}) "
                         f"| 💰C$ {lookup[x].preco_num:.2f}  📈{lookup[x].pred:.2f}pts"
                         + (f"  🔥{lookup[x].prob_mitada * 100:.0f}%"
                            if pd.notna(lookup[x].get("prob_mitada")) else ""),
                    key=f"manual_{formacao_manual}_{pos_id}_{i}",
                )

                if escolha_id is not None:
                    linha = lookup[escolha_id]
                    escolhidos.append(linha)
                    cartao_jogador(linha, largura=140)

    st.divider()
    if escolhidos:
        candidatos_cap = [e for e in escolhidos if e.posicao_id != 6]  # técnico não é capitão
        cap_lookup = {int(e.atleta_id): e for e in candidatos_cap}
        cap_ids = [None] + list(cap_lookup.keys())
        st.selectbox(
            "Capitão", cap_ids,
            format_func=lambda x: "— nenhum —" if x is None
            else f"{cap_lookup[x].apelido} ({cap_lookup[x].clube_abrev})",
            key=f"capitao_manual_{formacao_manual}",
        )
        st.caption("O resumo lá em cima já considera o capitão escolhido.")

        tabela_manual = pd.DataFrame([{
            "Time": e.clube_escudo, "Jogador": e.apelido, "Pos": e.posicao,
            "Contra": e.adversario_escudo, "Mando": "Casa" if e.casa else "Fora",
            "Preço": e.preco_num, "Previsão (modelo)": e.pred,
            "Chance de mitar": (e.prob_mitada * 100) if pd.notna(e.get("prob_mitada")) else None,
        } for e in escolhidos])
        st.dataframe(
            tabela_manual, use_container_width=True, hide_index=True,
            column_config={
                "Time": st.column_config.ImageColumn("Time", width="small"),
                "Contra": st.column_config.ImageColumn("Contra", width="small"),
                "Chance de mitar": st.column_config.ProgressColumn(
                    "Chance de mitar", format="%.0f%%", min_value=0, max_value=100),
            },
        )
    else:
        st.caption("Escolha os jogadores acima para ver custo e previsão total.")

# ============================== ABA 3 — MEU DESEMPENHO ==============================
with tab_meu:
    st.subheader("Seu histórico no Cartola")
    st.caption(
        "Busca pública pelo nome do seu time (o mesmo usado no app oficial do Cartola). "
        "Não pede sua senha — só funciona se o seu time estiver com o perfil público, "
        "que é o padrão."
    )

    nome_time = st.text_input("Nome do seu time no Cartola", key="nome_time_busca")
    if st.button("Buscar meu time"):
        if not nome_time.strip():
            st.warning("Digite o nome do time primeiro.")
        else:
            try:
                resultados = buscar_times(nome_time.strip())
            except Exception as e:
                resultados = None
                st.error(f"Não consegui buscar agora: {e}")
            if resultados is not None:
                if not resultados:
                    st.warning("Nenhum time encontrado com esse nome.")
                st.session_state["times_encontrados"] = resultados

    resultados = st.session_state.get("times_encontrados")
    if resultados:
        opcoes = {
            f"{t.get('nome', '?')} — {(t.get('time', {}) or {}).get('nome_cartola', t.get('nome_cartola', ''))}"
            .strip(" —"): t
            for t in resultados
        }
        escolha = st.selectbox("Selecione o seu time", list(opcoes.keys()), key="time_escolhido")
        time_sel = opcoes[escolha]
        time_id, slug = extrair_id_slug(time_sel)

        if not time_id and not slug:
            st.error("Não encontrei o identificador desse time na resposta da API.")
            with st.expander("Ver dados brutos recebidos (para depuração)"):
                st.json(time_sel)
        else:
            if st.button("Carregar meu histórico de pontuação"):
                with st.spinner("Buscando pontuação rodada a rodada..."):
                    hist_time, diagnostico = historico_meu_time(time_id, slug, rodada)
                st.session_state["hist_time"] = hist_time
                st.session_state["hist_time_diag"] = diagnostico

            hist_time = st.session_state.get("hist_time")
            diagnostico = st.session_state.get("hist_time_diag")

            if hist_time is not None and hist_time.empty:
                st.warning(
                    "Não consegui ler a pontuação por rodada — talvez o formato da resposta "
                    "tenha mudado nessa rota. Segue abaixo o que a API respondeu na primeira "
                    "tentativa; me manda um print disso que eu ajusto rapidinho."
                )
                if diagnostico:
                    with st.expander("Ver diagnóstico técnico", expanded=True):
                        st.json(diagnostico)

            elif hist_time is not None:
                media = hist_time.pontos.mean()
                melhor = hist_time.loc[hist_time.pontos.idxmax()]
                pior = hist_time.loc[hist_time.pontos.idxmin()]

                c1, c2, c3 = st.columns(3)
                c1.metric("Média por rodada", f"{media:.1f} pts")
                c2.metric("Melhor rodada", f"R{int(melhor.rodada)} · {melhor.pontos:.1f} pts")
                c3.metric("Pior rodada", f"R{int(pior.rodada)} · {pior.pontos:.1f} pts")

                st.line_chart(hist_time.set_index("rodada")["pontos"])
                st.dataframe(
                    hist_time.rename(columns={
                        "rodada": "Rodada", "pontos": "Pontos", "patrimonio": "Patrimônio",
                        "esquema": "Esquema",
                    }),
                    use_container_width=True, hide_index=True,
                )

                st.divider()
                st.subheader("Comparação: Você vs. Modelo vs. Time ideal")
                st.caption(
                    "Para cada rodada, o modelo é retreinado usando só dados até a rodada "
                    "anterior (sem cola) e aplicado nos resultados reais. O \"time ideal\" é "
                    "o teto matemático — o melhor time possível sabendo o resultado depois, "
                    "dado o mesmo orçamento que você tinha naquela rodada. Isso recalcula o "
                    "modelo várias vezes, então pode levar um tempinho."
                )
                n_rodadas = st.slider("Quantas das últimas rodadas comparar", 3, 10, 5, key="n_rodadas_comp")
                if st.button("Gerar comparação"):
                    with st.spinner("Recalculando o modelo para cada rodada — pode levar 1-2 minutos..."):
                        comp = comparativo_rodadas(hist_time, rodada, cartoletas, formacao, mult_cap, n_rodadas)
                    st.session_state["comparativo"] = comp

                comp = st.session_state.get("comparativo")
                if comp is not None:
                    if comp.empty:
                        st.warning(
                            "Não consegui montar a comparação para nenhuma dessas rodadas "
                            "(pode faltar preço histórico ou dados suficientes). Tenta um "
                            "número menor de rodadas ou rodadas mais recentes."
                        )
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sua média no período", f"{comp['Você'].mean():.1f} pts")
                        if comp["Modelo"].notna().any():
                            c2.metric("Média do modelo", f"{comp['Modelo'].mean():.1f} pts",
                                      delta=f"{comp['Modelo'].mean() - comp['Você'].mean():+.1f}")
                        if comp["Ideal da rodada"].notna().any():
                            c3.metric("Média do time ideal", f"{comp['Ideal da rodada'].mean():.1f} pts")

                        st.line_chart(comp.set_index("rodada")[["Você", "Modelo", "Ideal da rodada"]])
                        st.dataframe(
                            comp.rename(columns={"rodada": "Rodada"}),
                            use_container_width=True, hide_index=True,
                        )


