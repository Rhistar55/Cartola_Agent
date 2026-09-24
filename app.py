"""
Agente Cartola FC — App Streamlit.

Rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Deploy gratuito: suba este arquivo + requirements.txt num repositório do
GitHub e conecte em https://share.streamlit.io (Streamlit Community Cloud).
"""
import json, os, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.ensemble import HistGradientBoostingRegressor
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
FEATS = ["media_3", "media_5", "media_temp", "jogos", "casa", "cedido", "forca", "posicao_id"]


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
            linhas.append(dict(rodada=r, atleta_id=int(aid), clube_id=a["clube_id"],
                               posicao_id=a["posicao_id"], pontos=a["pontuacao"],
                               casa=casa, adversario=adv))
    return pd.DataFrame(linhas)


def features(h):
    h = h.sort_values(["atleta_id", "rodada"]).copy()
    g = h.groupby("atleta_id")["pontos"]
    h["media_3"] = g.transform(lambda s: s.shift().rolling(3, 1).mean())
    h["media_5"] = g.transform(lambda s: s.shift().rolling(5, 1).mean())
    h["media_temp"] = g.transform(lambda s: s.shift().expanding().mean())
    h["jogos"] = g.transform(lambda s: s.shift().notna().cumsum())

    ced = (h.groupby(["adversario", "posicao_id", "rodada"])["pontos"].mean()
           .reset_index().sort_values("rodada"))
    ced["cedido"] = ced.groupby(["adversario", "posicao_id"])["pontos"].transform(
        lambda s: s.shift().expanding().mean())
    h = h.merge(ced.drop(columns="pontos"), on=["adversario", "posicao_id", "rodada"], how="left")

    fc = h.groupby(["clube_id", "rodada"])["pontos"].mean().reset_index().sort_values("rodada")
    fc["forca"] = fc.groupby("clube_id")["pontos"].transform(lambda s: s.shift().expanding().mean())
    return h.merge(fc.drop(columns="pontos"), on=["clube_id", "rodada"], how="left")


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


@st.cache_data(ttl=1800, show_spinner=False)
def montar_base():
    """Baixa mercado, partidas e histórico; devolve dataframe de atletas com previsão pronta."""
    status = api_get("/mercado/status", cache=False)
    rodada = status["rodada_atual"]
    aberto = status.get("status_mercado") == 1

    merc = api_get("/atletas/mercado", cache=False)
    clubes = {int(k): v["abreviacao"] for k, v in merc["clubes"].items()}
    atl = pd.DataFrame(merc["atletas"])
    atl = atl[atl.status_id == PROVAVEL].copy()
    mando = mandos(api_get("/partidas", cache=False))
    atl = atl[atl.clube_id.isin(mando.keys())].copy()
    atl["casa"] = atl.clube_id.map(lambda c: mando[c][0])
    atl["adversario"] = atl.clube_id.map(lambda c: mando[c][1])

    h = historico(rodada)
    prox = atl[["atleta_id", "clube_id", "posicao_id", "casa", "adversario"]].assign(
        rodada=rodada, pontos=np.nan)
    tudo = features(pd.concat([h, prox], ignore_index=True))
    alvo = tudo[tudo.rodada == rodada].set_index("atleta_id")

    modelo, validacao = treinar(tudo[tudo.rodada < rodada])
    atl = atl.set_index("atleta_id")
    if modelo is None:
        atl["pred"] = atl.media_num
    else:
        atl["pred"] = modelo.predict(alvo.loc[atl.index, FEATS])
        sem_jogo = alvo.loc[atl.index, "jogos"].fillna(0).values == 0
        atl.loc[sem_jogo, "pred"] = atl.loc[sem_jogo, "media_num"] * 0.8
    atl = atl.reset_index()
    atl["clube_abrev"] = atl.clube_id.map(clubes)
    atl["adversario_abrev"] = atl.adversario.map(clubes)
    atl["posicao"] = atl.posicao_id.map(POS)
    return atl, rodada, aberto, validacao, mando


# ============================== INTERFACE ==============================
st.set_page_config(page_title="Agente Cartola", page_icon="⚽", layout="wide")
st.title("⚽ Agente Cartola FC")

with st.sidebar:
    st.header("Configuração")
    cartoletas = st.number_input("Cartoletas disponíveis (C$)", min_value=50.0, max_value=300.0,
                                  value=120.0, step=0.5)
    formacao = st.selectbox("Formação", ["auto"] + list(FORMACOES))
    mult_cap = st.slider("Multiplicador do capitão", 1.0, 2.0, 1.5, 0.1)
    gerar = st.button("Gerar escalação", type="primary")

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

col1, col2 = st.columns([1, 2])

if gerar:
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
        with col1:
            st.subheader(f"Formação {f}")
            st.metric("Pontuação prevista", f"{total:.1f} pts")
            st.metric("Custo total", f"C$ {time_.preco_num.sum():.2f}")
            escalados = time_.sort_values(["posicao_id", "pred"], ascending=[True, False])
            for _, j in escalados.iterrows():
                cap = " 🅲" if j.capitao else ""
                st.write(f"**{j.posicao}** — {j.apelido} ({j.clube_abrev} x {j.adversario_abrev}, "
                         f"{'casa' if j.casa else 'fora'}) · C$ {j.preco_num:.2f} · "
                         f"prev {j.pred:.2f}{cap}")
            csv = escalados.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV da escalação", csv, f"escalacao_rodada_{rodada}.csv")

# ---------------------------------------------------------------- tabela geral
with col2 if gerar else st.container():
    st.subheader("Previsão jogador a jogador")
    st.caption("Todos os jogadores prováveis, com a pontuação que o modelo espera para a rodada.")
    pos_filtro = st.multiselect("Filtrar posição", options=list(POS.values()), default=list(POS.values()))
    tabela = atl_disp[atl_disp.posicao.isin(pos_filtro)][
        ["apelido", "posicao", "clube_abrev", "adversario_abrev", "casa", "preco_num", "media_num", "pred"]
    ].rename(columns={
        "apelido": "Jogador", "posicao": "Pos", "clube_abrev": "Clube", "adversario_abrev": "Adversário",
        "casa": "Mando", "preco_num": "Preço", "media_num": "Média Cartola", "pred": "Previsão (modelo)",
    }).sort_values("Previsão (modelo)", ascending=False)
    tabela["Mando"] = tabela["Mando"].map({1: "Casa", 0: "Fora"})
    st.dataframe(tabela, use_container_width=True, hide_index=True)
