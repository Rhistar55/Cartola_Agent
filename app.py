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
    escudos = {int(k): (v.get("escudos") or {}).get("60x60") for k, v in merc["clubes"].items()}
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
    atl["clube_escudo"] = atl.clube_id.map(escudos)
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

    st.markdown(
        f"""
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
            </div>
            {hist_html}
        </div>
        """,
        unsafe_allow_html=True,

    )


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

tab_auto, tab_manual = st.tabs(["🤖 Escalação automática", "🛠️ Montar manualmente"])

# ============================== ABA 1 — AUTOMÁTICA ==============================
with tab_auto:
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

        for pos_id in [1, 2, 3, 4, 5, 6]:
            linha_pos = escalados[escalados.posicao_id == pos_id]
            if linha_pos.empty:
                continue
            st.caption(f"**{POS[pos_id]}**")
            cols = st.columns(len(linha_pos))
            for col, (_, row) in zip(cols, linha_pos.iterrows()):
                with col:
                    cartao_jogador(row)

        csv = escalados.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV da escalação", csv, f"escalacao_rodada_{rodada}.csv")

    st.divider()
    st.subheader("Previsão jogador a jogador")
    st.caption("Todos os jogadores prováveis, com a pontuação que o modelo espera para a rodada.")
    pos_filtro = st.multiselect("Filtrar posição", options=list(POS.values()),
                                 default=list(POS.values()), key="filtro_auto")
    tabela = atl_disp[atl_disp.posicao.isin(pos_filtro)][
        ["clube_escudo", "apelido", "posicao", "adversario_escudo", "casa", "preco_num", "media_num", "pred"]
    ].rename(columns={
        "clube_escudo": "Time", "apelido": "Jogador", "posicao": "Pos", "adversario_escudo": "Contra",
        "casa": "Mando", "preco_num": "Preço", "media_num": "Média Cartola", "pred": "Previsão (modelo)",
    }).sort_values("Previsão (modelo)", ascending=False)
    tabela["Mando"] = tabela["Mando"].map({1: "Casa", 0: "Fora"})
    st.dataframe(
        tabela, use_container_width=True, hide_index=True,
        column_config={
            "Time": st.column_config.ImageColumn("Time", width="small"),
            "Contra": st.column_config.ImageColumn("Contra", width="small"),
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
                         f"| 💰C$ {lookup[x].preco_num:.2f}  📈{lookup[x].pred:.2f}pts",
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
        } for e in escolhidos])
        st.dataframe(
            tabela_manual, use_container_width=True, hide_index=True,
            column_config={
                "Time": st.column_config.ImageColumn("Time", width="small"),
                "Contra": st.column_config.ImageColumn("Contra", width="small"),
            },
        )
    else:
        st.caption("Escolha os jogadores acima para ver custo e previsão total.")
