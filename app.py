import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import google.generativeai as genai

# --- PAGINA CONFIGURATIE ---
st.set_page_config(page_title="Pro Performance & AI Dashboard", layout="wide")
st.title("🚴‍♂️ Pro Performance & AI Dashboard")

# --- ZIJBLAD VOOR INSTELLINGEN ---
with st.sidebar:
    st.header("🔑 Koppelgegevens")
    # Checkt eerst of de sleutels al veilig in de cloud staan, anders kun je ze handmatig invullen
    default_id = st.secrets.get("ATHLETE_ID", "") if "ATHLETE_ID" in st.secrets else ""
    default_int = st.secrets.get("INTERVALS_KEY", "") if "INTERVALS_KEY" in st.secrets else ""
    default_gem = st.secrets.get("GEMINI_KEY", "") if "GEMINI_KEY" in st.secrets else ""

    athlete_id = st.text_input("Athlete ID (bijv. i123456)", value=default_id)
    intervals_key = st.text_input("Intervals.icu API Key", value=default_int, type="password")
    gemini_key = st.text_input("Gemini API Key", value=default_gem, type="password")

# --- DATA OPHALEN & VERWERKEN ---
@st.cache_data(ttl=600)
def haal_data_op(athlete_id, api_key, dagen_terug):
    eind_datum = datetime.today().strftime('%Y-%m-%d')
    start_datum = (datetime.today() - timedelta(days=dagen_terug)).strftime('%Y-%m-%d')
    auth = ('API_KEY', api_key)
    
    act_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest={start_datum}&newest={eind_datum}"
    wel_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={start_datum}&newest={eind_datum}"
    
    act_res = requests.get(act_url, auth=auth)
    wel_res = requests.get(wel_url, auth=auth)
    
    if act_res.status_code != 200 or wel_res.status_code != 200:
        return None, None
        
    df_act = pd.DataFrame(act_res.json())
    df_wel = pd.DataFrame(wel_res.json())
    
    # Data opschonen & verrijken voor grafieken en hover-info
    if not df_act.empty:
        if 'type' in df_act.columns:
            df_act = df_act[df_act['type'].notna()].copy()
        if 'start_date_local' in df_act.columns:
            df_act['datum'] = pd.to_datetime(df_act['start_date_local']).dt.strftime('%Y-%m-%d')
            df_act = df_act.sort_values('start_date_local')
        if 'moving_time' in df_act.columns:
            df_act['duur_min'] = (df_act['moving_time'] / 60).round(0)
            df_act['duur_uur'] = (df_act['moving_time'] / 3600).round(2)
        if 'distance' in df_act.columns:
            df_act['afstand_km'] = (df_act['distance'] / 1000).round(1)
            
    if not df_wel.empty and 'id' in df_wel.columns:
        df_wel['datum'] = pd.to_datetime(df_wel['id']).dt.strftime('%Y-%m-%d')
        df_wel = df_wel.sort_values('datum')
        if 'sleepSecs' in df_wel.columns:
            df_wel['slaap_uren'] = (df_wel['sleepSecs'] / 3600).round(1)
        if 'ctl' in df_wel.columns and 'atl' in df_wel.columns:
            df_wel['ctl'] = df_wel['ctl'].round(1)
            df_wel['atl'] = df_wel['atl'].round(1)
            df_wel['tsb'] = (df_wel['ctl'] - df_wel['atl']).round(1)
            
    return df_act, df_wel

# --- HOOFDSCHERM ---
if athlete_id and intervals_key:
    df_act, df_wel = haal_data_op(athlete_id, intervals_key, dagen)
    
    if df_act is None:
        st.error("❌ Kan geen verbinding maken. Controleer je Athlete ID (inclusief de 'i') en API Key.")
    else:
        # --- 1. TOP KPI'S MET HOVER-UITLEG ---
        st.subheader(f"⚡ Kerncijfers afgelopen {dagen} dagen")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        
        with k1:
            uren = round(df_act['duur_uur'].sum(), 1) if not df_act.empty and 'duur_uur' in df_act.columns else 0
            aantal = len(df_act) if not df_act.empty else 0
            st.metric(
                "Volume", f"{uren} uur", f"{aantal} sessies",
                help="Totale beweegtijd (moving time) en het totaal aantal geregistreerde trainingen in de geselecteerde periode."
            )
        with k2:
            km = round(df_act['afstand_km'].sum(), 0) if not df_act.empty and 'afstand_km' in df_act.columns else 0
            st.metric(
                "Totale Afstand", f"{int(km)} km",
                help="Alle gefietste, gelopen en gezwommen kilometers bij elkaar opgeteld in deze periode."
            )
        with k3:
            load = int(df_act['icu_training_load'].sum()) if not df_act.empty and 'icu_training_load' in df_act.columns else 0
            gem_load = int(load / (dagen / 7)) if dagen > 0 else 0
            st.metric(
                "Totale Load (TSS)", f"{load}", f"~{gem_load} p/week",
                help="Training Stress Score (TSS): De totale fysiologische belasting van je trainingen op basis van duur en intensiteit (1 uur op FTP = 100 punten)."
            )
        with k4:
            if not df_wel.empty and 'ctl' in df_wel.columns and df_wel['ctl'].notna().any():
                huidige_ctl = round(df_wel['ctl'].dropna().iloc[-1], 1)
                start_ctl = round(df_wel['ctl'].dropna().iloc[0], 1)
                verschil_ctl = round(huidige_ctl - start_ctl, 1)
                st.metric(
                    "Fitness (CTL)", huidige_ctl, f"{verschil_ctl:+} in periode",
                    help="Chronic Training Load (CTL): Je langetermijnfitheid. Dit is het gewogen gemiddelde van je dagelijkse trainingsload (TSS) over de afgelopen 42 dagen. Hoe hoger, hoe groter je basisconditie."
                )
            else:
                st.metric("Fitness (CTL)", "-", help="Geen CTL data gevonden.")
        with k5:
            if not df_wel.empty and 'tsb' in df_wel.columns and df_wel['tsb'].notna().any():
                huidige_tsb = round(df_wel['tsb'].dropna().iloc[-1], 1)
                status = "Fris (Race klaar)" if huidige_tsb > 5 else ("Optimale opbouw" if huidige_tsb >= -15 else "Zwaar belast (Let op)")
                st.metric(
                    "Vorm (TSB)", huidige_tsb, status,
                    help="Training Stress Balance (TSB = Fitness CTL min Vermoeidheid ATL). \n• Boven 0: Je bent fris en uitgerust (ideaal voor een wedstrijd/piek). \n• Tussen -10 en -25: Productieve trainingsfase (je bouwt conditie op). \n• Onder -25: Risico op overbelasting, tijd voor herstel."
                )
            else:
                st.metric("Vorm (TSB)", "-", help="Geen TSB data gevonden.")
        with k6:
            if not df_wel.empty and 'hrv' in df_wel.columns and df_wel['hrv'].notna().any():
                gem_hrv = round(df_wel['hrv'].mean(), 1)
                laatste_hrv = round(df_wel['hrv'].dropna().iloc[-1], 1)
                diff_hrv = round(laatste_hrv - gem_hrv, 1)
                st.metric(
                    "Laatste HRV", f"{laatste_hrv} ms", f"{diff_hrv:+} vs gem ({gem_hrv})",
                    help="Heart Rate Variability (variatie in tijd tussen hartslagen tijdens je slaap). Een waarde rond of boven je gemiddelde betekent dat je zenuwstelsel goed hersteld is. Een flinke duik omlaag wijst op vermoeidheid, stress of opkomende ziekte."
                )
            else:
                st.metric("HRV", "-", help="Geen HRV data gevonden.")

        st.divider()

        # --- 2. VISUELE VOORTGANG TABS ---
        tab_pmc, tab_herstel, tab_power, tab_data = st.tabs([
            "📈 Vorm & Belasting (PMC)", 
            "💤 Herstel, HRV & Slaap", 
            "⚡ Vermogen & Efficiëntie", 
            "📋 Ruwe Data Tabellen"
        ])

        # TAB 1: PMC (Fitness, Vermoeidheid, Vorm & Dagelijkse Load)
        with tab_pmc:
            st.caption("💡 **Tip:** Zweef met je muis over de grafiek voor een gedetailleerde uitleg per dag. Je kunt inzoomen door een vak te slepen.")
            if not df_wel.empty and 'ctl' in df_wel.columns:
                fig_pmc = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                    subplot_titles=("Fitness (CTL) vs. Vermoeidheid (ATL) & Dagelijkse TSS", "Vorm / Freshness (TSB = CTL - ATL)")
                )
                
                if not df_act.empty and 'icu_training_load' in df_act.columns:
                    dag_load = df_act.groupby('datum').agg(
                        totale_tss=('icu_training_load', 'sum'),
                        ritten=('name', lambda x: ', '.join(x.dropna()))
                    ).reset_index()
                    
                    fig_pmc.add_trace(go.Bar(
                        x=dag_load['datum'], y=dag_load['totale_tss'], name="Dagelijkse TSS", 
                        marker_color='rgba(150, 150, 150, 0.4)',
                        customdata=dag_load['ritten'],
                        hovertemplate="<b>Dagelijkse Load: %{y:.0f} TSS</b><br>Sessie(s): %{customdata}<br><i>De trainingsprikkel van deze specifieke dag.</i><extra></extra>"
                    ), row=1, col=1)

                fig_pmc.add_trace(go.Scatter(
                    x=df_wel['datum'], y=df_wel['ctl'], name="Fitness (CTL)", 
                    line=dict(color='#00b4d8', width=3),
                    hovertemplate="<b>Fitness (CTL): %{y:.1f}</b><br><i>Gemiddelde belasting afgelopen 42 dagen (Je basisconditie).</i><extra></extra>"
                ), row=1, col=1)
                
                fig_pmc.add_trace(go.Scatter(
                    x=df_wel['datum'], y=df_wel['atl'], name="Vermoeidheid (ATL)", 
                    line=dict(color='#ef476f', width=2, dash='dot'),
                    hovertemplate="<b>Vermoeidheid (ATL): %{y:.1f}</b><br><i>Gemiddelde belasting afgelopen 7 dagen (Kortetermijnvermoeidheid).</i><extra></extra>"
                ), row=1, col=1)
                
                if 'tsb' in df_wel.columns:
                    kleuren_tsb = ['#06d6a0' if val >= 0 else '#ffd166' if val >= -15 else '#ef476f' for val in df_wel['tsb'].fillna(0)]
                    fig_pmc.add_trace(go.Bar(
                        x=df_wel['datum'], y=df_wel['tsb'], name="Vorm (TSB)", 
                        marker_color=kleuren_tsb,
                        hovertemplate="<b>Vorm (TSB): %{y:.1f}</b><br><i>Positief = Fris | -10 tot -25 = Opbouw | Onder -25 = Overbelastingsrisico.</i><extra></extra>"
                    ), row=2, col=1)
                    fig_pmc.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                    fig_pmc.add_hline(y=-15, line_dash="dot", line_color="#ef476f", annotation_text="Zware opbouw (-15)", row=2, col=1)

                fig_pmc.update_layout(height=580, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig_pmc, use_container_width=True)
            else:
                st.info("Geen CTL/ATL data beschikbaar in deze periode.")

        # TAB 2: HERSTEL, HRV & SLAAP
        with tab_herstel:
            st.caption("💡 **Tip:** Zweef over de punten om direct te zien hoe je zenuwstelsel en slaap reageren op zware trainingsdagen.")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                if not df_wel.empty and ('hrv' in df_wel.columns or 'restingHR' in df_wel.columns):
                    fig_hrv = make_subplots(specs=[[{"secondary_y": True}]])
                    if 'hrv' in df_wel.columns:
                        fig_hrv.add_trace(go.Scatter(
                            x=df_wel['datum'], y=df_wel['hrv'], name="HRV (ms)", 
                            mode='lines+markers', line=dict(color='#06d6a0', width=3),
                            hovertemplate="<b>HRV: %{y:.0f} ms</b><br><i>Hoger of stabiel = Goed herstel van je zenuwstelsel.</i><extra></extra>"
                        ), secondary_y=False)
                    if 'restingHR' in df_wel.columns:
                        fig_hrv.add_trace(go.Scatter(
                            x=df_wel['datum'], y=df_wel['restingHR'], name="Rusthartslag (bpm)", 
                            mode='lines+markers', line=dict(color='#ef476f', width=2),
                            hovertemplate="<b>Rusthartslag: %{y:.0f} bpm</b><br><i>Lagere/stabiele hartslag = Goed hersteld. Piek omhoog = Restvermoeidheid.</i><extra></extra>"
                        ), secondary_y=True)
                    fig_hrv.update_layout(title="Autonoom Zenuwstelsel: HRV vs. Rusthartslag", hovermode="x unified", height=420)
                    fig_hrv.update_yaxes(title_text="HRV (ms) - Hoger is beter", secondary_y=False)
                    fig_hrv.update_yaxes(title_text="Rusthartslag (bpm) - Lager is beter", secondary_y=True)
                    st.plotly_chart(fig_hrv, use_container_width=True)
            
            with col_h2:
                if not df_wel.empty and 'slaap_uren' in df_wel.columns:
                    fig_sleep = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_sleep.add_trace(go.Bar(
                        x=df_wel['datum'], y=df_wel['slaap_uren'], name="Slaapduur", 
                        marker_color='#118ab2', opacity=0.7,
                        hovertemplate="<b>Slaapduur: %{y:.1f} uur</b><br><i>Richtlijn voor duursporters: minimaal 7,5 tot 8 uur.</i><extra></extra>"
                    ), secondary_y=False)
                    if 'sleepScore' in df_wel.columns:
                        fig_sleep.add_trace(go.Scatter(
                            x=df_wel['datum'], y=df_wel['sleepScore'], name="Slaap Score", 
                            mode='lines+markers', line=dict(color='#ffd166', width=3),
                            hovertemplate="<b>Garmin Slaap Score: %{y:.0f}/100</b><br><i>Berekend op basis van diepe slaap, REM-slaap en rusteloosheid.</i><extra></extra>"
                        ), secondary_y=True)
                    fig_sleep.add_hline(y=7.5, line_dash="dot", line_color="green", annotation_text="7.5u Richtlijn", secondary_y=False)
                    fig_sleep.update_layout(title="Slaapduur & Slaapkwaliteit", hovermode="x unified", height=420)
                    fig_sleep.update_yaxes(title_text="Uren slaap", secondary_y=False)
                    fig_sleep.update_yaxes(title_text="Slaap Score (0-100)", secondary_y=True, range=[0, 105])
                    st.plotly_chart(fig_sleep, use_container_width=True)

        # TAB 3: VERMOGEN & AEROBE EFFICIËNTIE
        with tab_power:
            st.caption("💡 **Tip:** Beweeg over een balk of punt om direct de ritnaam, duur, hartslag en uitleg van de vermogensmetrics te zien.")
            if not df_act.empty:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    df_power = df_act[df_act['average_watts'].notna()].copy() if 'average_watts' in df_act.columns else pd.DataFrame()
                    if not df_power.empty:
                        # Zorg dat extra kolommen voor de hover altijd bestaan
                        df_power['hr_hover'] = df_power['average_heartrate'].fillna(0).round(0) if 'average_heartrate' in df_power.columns else 0
                        df_power['np_hover'] = df_power['weighted_average_watts'].fillna(0).round(0) if 'weighted_average_watts' in df_power.columns else 0
                        
                        fig_pow = go.Figure()
                        fig_pow.add_trace(go.Bar(
                            x=df_power['datum'], y=df_power['average_watts'], name="Gemiddeld Wattage", 
                            marker_color='#118ab2',
                            customdata=df_power[['name', 'duur_min', 'afstand_km', 'hr_hover', 'np_hover']],
                            hovertemplate="<b>%{customdata[0]}</b> (%{x})<br>⏱️ Duur: %{customdata[1]:.0f} min | 📍 %{customdata[2]:.1f} km<br>⚡ <b>Gem. Vermogen: %{y:.0f} W</b><br>🔥 Normalized Power (NP): %{customdata[4]:.0f} W<br>❤️ Gem. Hartslag: %{customdata[3]:.0f} bpm<br><i>NP corrigeert voor sprintjes, klimmetjes en benen stilhouden.</i><extra></extra>"
                        ))
                        if 'weighted_average_watts' in df_power.columns:
                            fig_pow.add_trace(go.Scatter(
                                x=df_power['datum'], y=df_power['weighted_average_watts'], name="Normalized Power (NP)", 
                                mode='markers+lines', line=dict(color='#ffd166', width=2), marker=dict(size=9),
                                hovertemplate="<b>Normalized Power (NP): %{y:.0f} W</b><br><i>De echte fysiologische zwaarte van het geleverde vermogen.</i><extra></extra>"
                            ))
                        fig_pow.update_layout(title="Geleverde Wattages per Rit (Avg vs. NP)", hovermode="x unified", height=420)
                        st.plotly_chart(fig_pow, use_container_width=True)
                    else:
                        st.info("Geen ritten met vermogensmeter gevonden in deze periode.")

                with col_p2:
                    if 'icu_efficiency_factor' in df_act.columns and df_act['icu_efficiency_factor'].notna().any():
                        df_ef = df_act[df_act['icu_efficiency_factor'].notna()].copy()
                        df_ef['hr_hover'] = df_ef['average_heartrate'].fillna(0).round(0) if 'average_heartrate' in df_ef.columns else 0
                        df_ef['np_hover'] = df_ef['weighted_average_watts'].fillna(0).round(0) if 'weighted_average_watts' in df_ef.columns else 0

                        fig_ef = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_ef.add_trace(go.Scatter(
                            x=df_ef['datum'], y=df_ef['icu_efficiency_factor'], name="Efficiency Factor (EF)", 
                            mode='lines+markers', line=dict(color='#06d6a0', width=3),
                            customdata=df_ef[['name', 'np_hover', 'hr_hover']],
                            hovertemplate="<b>%{customdata[0]}</b><br><b>Efficiency Factor: %{y:.2f}</b> (NP: %{customdata[1]:.0f}W / HR: %{customdata[2]:.0f}bpm)<br><i>Hoeveel watt je trapt per hartslag. Stijgt deze lijn over weken? Dan groeit je aerobe motor!</i><extra></extra>"
                        ), secondary_y=False)
                        if 'decoupling' in df_ef.columns:
                            fig_ef.add_trace(go.Bar(
                                x=df_ef['datum'], y=df_ef['decoupling'], name="Decoupling / Hartslagdrift (%)", 
                                marker_color='rgba(239, 71, 111, 0.4)',
                                hovertemplate="<b>Decoupling (Drift): %{y:.1f}%</b><br><i>Hoeveel je hartslag oploopt in de 2e helft van je rit bij gelijk vermogen.<br>Onder 5% = Goede duurconditie. Boven 5% = Vermoeidheid, hitte of te weinig gedronken/gegeten.</i><extra></extra>"
                            ), secondary_y=True)
                        fig_ef.update_layout(title="Aerobe Motor: Efficiency Factor & Decoupling", hovermode="x unified", height=420)
                        fig_ef.update_yaxes(title_text="EF (Watt per hartslag - Hoger is fitter)", secondary_y=False)
                        fig_ef.update_yaxes(title_text="Decoupling % (<5% is top)", secondary_y=True)
                        st.plotly_chart(fig_ef, use_container_width=True)
                    else:
                        st.info("Geen Efficiency Factor data beschikbaar.")

                # Verdeling per sporttype met uitgebreide hover
                if 'type' in df_act.columns and 'icu_training_load' in df_act.columns:
                    fig_type = px.bar(
                        df_act, x='datum', y='icu_training_load', color='type', 
                        title="Trainingsload (TSS) per Sportdiscipline", 
                        hover_data={'name': True, 'duur_min': True, 'afstand_km': True, 'icu_training_load': True},
                        labels={'icu_training_load': 'Load (TSS)', 'duur_min': 'Duur (min)', 'afstand_km': 'Afstand (km)', 'name': 'Sessie', 'type': 'Sport'}
                    )
                    fig_type.update_layout(height=360)
                    st.plotly_chart(fig_type, use_container_width=True)

        # TAB 4: RUWE DATA TABELLEN MET KOLOM-TOOLTIPS
        with tab_data:
            st.caption("💡 **Tip:** Zweef met je muis over de kolomnamen bovenin de tabel voor uitleg over elke afkorting.")
            toon_alles = st.checkbox(
                "🔍 Toon élke beschikbare ruwe datakolom uit Garmin & Intervals.icu",
                help="Vink aan om alle ~100 technische kolommen (zoals temperatuur, joules, zonetijden en intervallen) te bekijken."
            )
            
            kern_act = ['datum', 'name', 'type', 'duur_min', 'afstand_km', 'average_watts', 'weighted_average_watts', 'average_heartrate', 'max_heartrate', 'icu_training_load', 'icu_intensity', 'icu_efficiency_factor', 'decoupling', 'average_cadence']
            beschikbaar_act = [c for c in kern_act if c in df_act.columns] if not df_act.empty else []
            
            kern_wel = ['datum', 'restingHR', 'hrv', 'slaap_uren', 'sleepScore', 'ctl', 'atl', 'tsb', 'rampRate', 'weight']
            beschikbaar_wel = [c for c in kern_wel if c in df_wel.columns] if not df_wel.empty else []

            st.markdown("**Trainingen**")
            if not df_act.empty:
                st.dataframe(
                    df_act if toon_alles else df_act[beschikbaar_act], 
                    use_container_width=True,
                    column_config={
                        "weighted_average_watts": st.column_config.NumberColumn("NP (Watt)", help="Normalized Power: Het gecorrigeerde vermogen dat de echte fysiologische belasting weergeeft."),
                        "average_watts": st.column_config.NumberColumn("Gem. Watt", help="Het simpele gemiddelde vermogen inclusief nul-waardes (benen stilhouden)."),
                        "icu_training_load": st.column_config.NumberColumn("Load (TSS)", help="Training Stress Score: Totale zwaarte van de rit."),
                        "icu_intensity": st.column_config.NumberColumn("Intensiteit (%)", help="Intensity Factor (IF): Hoe zwaar de rit was ten opzichte van je FTP (100% = 1 uur voluit op FTP)."),
                        "icu_efficiency_factor": st.column_config.NumberColumn("EF (Watt/HR)", help="Efficiency Factor: Normalized Power gedeeld door je gemiddelde hartslag. Graadmeter voor je aerobe conditie."),
                        "decoupling": st.column_config.NumberColumn("Decoupling (%)", help="Aerobe ontkoppeling: Hoeveel procent je hartslag opliep in de tweede helft van de training ten opzichte van je vermogen.")
                    }
                )
            st.markdown("**Wellness & Herstel**")
            if not df_wel.empty:
                st.dataframe(
                    df_wel if toon_alles else df_wel[beschikbaar_wel], 
                    use_container_width=True,
                    column_config={
                        "restingHR": st.column_config.NumberColumn("Rust HR", help="Laagste gemeten hartslag tijdens rust/slaap."),
                        "hrv": st.column_config.NumberColumn("HRV (ms)", help="Hartslagvariabiliteit in milliseconden. Maatstaf voor herstel van het autonome zenuwstelsel."),
                        "ctl": st.column_config.NumberColumn("Fitness (CTL)", help="Chronic Training Load: Langetermijnfitheid (42 dagen)."),
                        "atl": st.column_config.NumberColumn("Vermoeidheid (ATL)", help="Acute Training Load: Kortetermijnvermoeidheid (7 dagen)."),
                        "tsb": st.column_config.NumberColumn("Vorm (TSB)", help="Training Stress Balance (CTL - ATL): Je actuele frisheid."),
                        "rampRate": st.column_config.NumberColumn("Ramp Rate", help="Hoe snel je Fitness (CTL) per week stijgt. Boven de 5 à 7 punten per week vergroot de kans op overbelasting.")
                    }
                )

        # --- 3. INTERACTIEVE AI PERFORMANCE COACH ---
        st.divider()
        st.subheader("🧠 AI Performance Coach & Data-analist")
        
        if not gemini_key:
            st.warning("👉 Vul links in de zijbalk je **Gemini API Key** in om de AI-analyse te gebruiken.")
        else:
            genai.configure(api_key=gemini_key)
            
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": f"Ik heb je volledige historie van de afgelopen {dagen} dagen klaarstaan (actief model: `{ai_model}`). Klik op een van de knoppen hieronder voor een snelle analyse, of stel je eigen specifieke vraag!"}
                ]

            st.caption("Snelle analyses (zweef over een knop voor details):")
            b1, b2, b3 = st.columns(3)
            snelle_vraag = None
            if b1.button("📊 Analyseer mijn afgelopen week", use_container_width=True, help="Laat de AI je wattages, hartslag en trainingsload van de laatste 7 dagen afzetten tegen je slaap en HRV."):
                snelle_vraag = "Geef een scherpe analyse van mijn afgelopen 7 dagen. Hoe verhouden mijn geleverde wattages en trainingsload zich tot mijn HRV, slaap en rusthartslag? Ben ik goed hersteld?"
            if b2.button("🔋 Check op vermoeidheid & overreaching", use_container_width=True, help="Controleert je TSB, Ramp Rate, HRV-trends en hartslagdrift (decoupling) op vroege signalen van overtraining."):
                snelle_vraag = "Analyseer mijn TSB (vorm), HRV-trend, rusthartslag en decoupling over de hele periode. Zie je rode vlaggen van opstapelende vermoeidheid of kan ik de belasting opschroeven?"
            if b3.button("🎯 Advies voor mijn volgende training", use_container_width=True, help="Berekent op basis van je herstelcijfers van vannacht en je recente belasting wat voor training én welk wattage/hartslag vandaag slim is."):
                snelle_vraag = "Op basis van mijn hersteldata van de laatste 3 dagen (HRV, slaap, TSB) en mijn recente trainingen: wat voor type training (duur, tempo, interval of rust) met welke intensiteit adviseer je voor vandaag en morgen?"

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            chat_input = st.chat_input("Stel een specifieke vraag over je cijfers...")
            gebruikers_vraag = snelle_vraag if snelle_vraag else chat_input

            if gebruikers_vraag:
                st.session_state.messages.append({"role": "user", "content": gebruikers_vraag})
                with st.chat_message("user"):
                    st.markdown(gebruikers_vraag)

                act_data_str = df_act[beschikbaar_act].to_string() if not df_act.empty else "Geen trainingen"
                wel_data_str = df_wel[beschikbaar_wel].to_string() if not df_wel.empty else "Geen wellness data"
                gesprek_historie = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])

                prompt = f"""
                Je bent een analytische, no-nonsense performance coach voor wielrennen en triatlon.
                Je spreekt duidelijk en direct Nederlands, zonder wollig of typisch AI-taalgebruik.
                Onderbouw elke conclusie direct met harde cijfers uit de tabellen hieronder (noem specifieke datums, wattages, Normalized Power, Efficiency Factor, hartslag, HRV, slaapuren en CTL/ATL/TSB).

                === TRAINING DATA AFGELOPEN {dagen} DAGEN ===
                {act_data_str}

                === HERSTEL & WELLNESS DATA AFGELOPEN {dagen} DAGEN ===
                {wel_data_str}

                === EERDERE BERICHTEN IN DIT GESPREK ===
                {gesprek_historie}

                Vraag van de atleet: {gebruikers_vraag}
                """

                with st.chat_message("assistant"):
                    with st.spinner("Cijfers worden doorgelicht..."):
                        try:
                            model = genai.GenerativeModel(ai_model)
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        except Exception as e:
                            st.error(f"Fout bij AI-aanvraag met model '{ai_model}': {e}. Pas eventueel links in het menu de modelnaam aan.")

else:
    st.info("👈 Vul links in de zijbalk je **Athlete ID** en **Intervals.icu API Key** in om het dashboard te activeren.")
