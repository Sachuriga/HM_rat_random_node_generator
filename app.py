import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
import io
import itertools
import datetime

from protocol_utils import build_island_sequence, evaluate_protocol_compliance, get_shortest_distance, select_start_node

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Rat Hexmaze Generator", layout="wide")

st.title("🐀 Rat Hexmaze Experiment Setup")
st.markdown("Generates randomized start nodes based on exclusions, block-based island rotation, and distance rules.")

# ==========================================
# 1. GRAPH GENERATION (Cached for Speed)
# ==========================================
@st.cache_data
def load_graph(uploaded_file):
    if uploaded_file is None:
        return None
    
    # Load node list
    df = pd.read_csv(uploaded_file, header=None, names=['id', 'x', 'y'])
    
    G = nx.Graph()

    # Add nodes
    for idx, row in df.iterrows():
        node_id = str(int(row['id']))
        # Group logic: 100s->1, 200s->2, etc.
        G.add_node(node_id, pos=(row['x'], row['y']), group=int(row['id']) // 100)

    # Add internal edges based on distance
    coords = df[['x', 'y']].values
    distances = squareform(pdist(coords))
    threshold = 65
    nodes = list(G.nodes())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if distances[i, j] < threshold:
                G.add_edge(nodes[i], nodes[j])

    # Remove standard dead/unused nodes
    nodes_to_remove = ['501', '502']
    for n in nodes_to_remove:
        if n in G: G.remove_node(n)

    # Add manual bridge connections
    manual_edges = [('121', '302'), ('324', '401'), ('305', '220'), ('404', '223'), ('201', '124')]
    
    # --- VERIFICATION STEP ---
    all_nodes = set(G.nodes())
    for u, v in manual_edges:
        if u in all_nodes and v in all_nodes:
            G.add_edge(u, v)
        else:
            st.warning(f"⚠️ Warning: Manual bridge node missing in CSV: {u}-{v}")

    G.graph['distance_lookup'] = dict(nx.all_pairs_shortest_path_length(G))

    return G

# ==========================================
# 2. LOGIC GENERATOR
# ==========================================
def generate_sequence(G, inputs):
    # --- CONFIGURATION ---
    TOTAL_TRIALS = inputs['num_trials']
    MIN_DISTANCE_FROM_GOAL = 4  
    
    # --- RULE I: Hard Exclusions ---
    hard_exclusions = ['213', '214', '215', '220', '305', '310', '311', '312']
    total_forbidden = set(hard_exclusions + inputs['prev_used_nodes'] + [inputs['goal']])
    
    all_groups = [1, 2, 3, 4]
    available_groups = [g for g in all_groups if g != inputs['goal_group']] 
    
    distance_cache = {}

    # --- HELPER: Get valid node from a specific group ---
    def get_valid_node(target_group, current_selected, graph, forbidden_set, goal_node):
        return select_start_node(
            graph,
            target_group=target_group,
            current_selected=current_selected,
            forbidden_set=forbidden_set,
            goal_node=goal_node,
            prev_goal_node=inputs['prev_goal_node'],
            prev_goal_group=inputs['prev_goal_group'],
            current_goal_group=inputs['goal_group'],
            min_distance_from_goal=MIN_DISTANCE_FROM_GOAL,
            distance_cache=distance_cache,
        )

    # ---------------------------------------------------------
    # STEP 1: Handle Special Start Node (NGL/PT Logic)
    # ---------------------------------------------------------
    forced_t1_node = None
    t1_dist_info = ""

    if inputs['is_ngl_pt'] and inputs['t1_ref_goal']:
        st.info("Applying HIDDEN RULE: NGL/PT Distance Check")
        candidates = []
        for n in G.nodes():
            if n in total_forbidden: continue
            g = G.nodes[n]['group']
            if g == inputs['goal_group']: continue
            if g == inputs['t1_ref_group']: continue

            try:
                dist_to_curr_goal = get_shortest_distance(G, n, inputs['goal'], distance_cache)
                if dist_to_curr_goal < MIN_DISTANCE_FROM_GOAL: continue

                d_old = get_shortest_distance(G, n, inputs['t1_ref_goal'], distance_cache)
                diff = abs(dist_to_curr_goal - d_old)
                candidates.append((n, dist_to_curr_goal, d_old, diff))
            except Exception:
                continue

        # Prefer a start that is EXACTLY equidistant from the current goal and the
        # old goal, then relax to within 1, then within 2 steps if none qualify.
        selected = None
        for max_diff in (0, 1, 2):
            pool = [c for c in candidates if c[3] <= max_diff]
            if pool:
                preferred = [c for c in pool if G.nodes[c[0]]['group'] != inputs['prev_last_group']]
                selected = random.choice(preferred) if preferred else random.choice(pool)
                break

        if selected:
            forced_t1_node = selected[0]
            t1_dist_info = f" (Dist New: {selected[1]}, Dist Old: {selected[2]}, Diff: {selected[3]})"
        else:
            st.caption("Note: No start node was equidistant (within 2 steps) from the current and old goals; trial 1 selected normally.")

    # ---------------------------------------------------------
    # STEP 2: Generate Island Sequence
    # ---------------------------------------------------------
    forced_t1_group = G.nodes[forced_t1_node]['group'] if forced_t1_node else None
    island_sequence = build_island_sequence(
        total_trials=TOTAL_TRIALS,
        available_groups=all_groups,
        goal_group=inputs['goal_group'],
        forced_t1_group=forced_t1_group,
        prev_last_group=inputs['prev_last_group'],
        is_ngl_pt=inputs['is_ngl_pt'],
    )

    if not island_sequence:
        st.error("Error: Could not construct a valid island sequence from the protocol rules.")
        return None, None

    if forced_t1_node and island_sequence[0] != forced_t1_group:
        st.caption("Note: The protocol-aware island builder adjusted the initial island ordering.")

    # ---------------------------------------------------------
    # STEP 3: Select Nodes
    # ---------------------------------------------------------
    final_sequence = []
    session_selected_nodes = set()
    
    start_index = 0
    if forced_t1_node:
        final_sequence.append(forced_t1_node)
        session_selected_nodes.add(forced_t1_node)
        start_index = 1

    for i in range(start_index, len(island_sequence)):
        target_group = island_sequence[i]
        node = get_valid_node(target_group, session_selected_nodes, G, total_forbidden, inputs['goal'])
        
        if not node:
            st.error(f"Error: Could not find a valid node in Island {target_group} that is >= 4 steps from Goal.")
            return None, None
            
        final_sequence.append(node)
        session_selected_nodes.add(node)

    return final_sequence, t1_dist_info

# ==========================================
# 3. PLOTTING (UPDATED)
# ==========================================
def create_plot(G, sequence, inputs, extra_info=""):
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.get_node_attributes(G, 'pos')
    
    # 1. Background Graph
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=80, alpha=0.2, ax=ax)
    
    # 2. Never Used (Hard Exclusions)
    hard_exclusions = ['213', '214', '215', '220', '305', '310', '311', '312']
    existing_exclusions = [n for n in hard_exclusions if n in G.nodes()]
    if existing_exclusions:
        nx.draw_networkx_nodes(G, pos, nodelist=existing_exclusions, 
                               node_color='black', node_size=100, node_shape='x', label='Never Used', ax=ax)

    # 3. Previous Session Nodes (Faint / Transparent)
    # Filter only nodes that actually exist in graph to prevent errors
    valid_prev_used = [n for n in inputs['prev_used_nodes'] if n in G.nodes()]
    if valid_prev_used:
        nx.draw_networkx_nodes(G, pos, nodelist=valid_prev_used,
                               node_color='gray', node_size=200, alpha=0.15, ax=ax)

    # 4. Previous Session START and END (Highlighted)
    if inputs['prev_first_node'] and inputs['prev_first_node'] in G.nodes():
        n = inputs['prev_first_node']
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color='orange', node_size=300, node_shape='^', alpha=0.5, ax=ax)
        ax.text(pos[n][0], pos[n][1]-20, "Prev Start", fontsize=8, color='orange', ha='center')

    if inputs['prev_last_node'] and inputs['prev_last_node'] in G.nodes():
        n = inputs['prev_last_node']
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color='purple', node_size=300, node_shape='v', alpha=0.5, ax=ax)
        ax.text(pos[n][0], pos[n][1]-20, "Prev End", fontsize=8, color='purple', ha='center')

    # 5. Previous Goal
    if inputs['prev_goal_node']:
        nx.draw_networkx_nodes(G, pos, nodelist=[inputs['prev_goal_node']], 
                               node_color='salmon', node_size=300, node_shape='X', alpha=0.6, ax=ax)
        ax.text(pos[inputs['prev_goal_node']][0], pos[inputs['prev_goal_node']][1]-15, "Prev Goal", fontsize=8, color='salmon', ha='center')

    # 5b. PT Old Goal (probe reference for the trial-1 distance match)
    old_goal_disp = inputs.get('old_goal_display')
    if old_goal_disp and old_goal_disp in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[old_goal_disp],
                               node_color='gold', node_size=500, node_shape='X', alpha=0.75, ax=ax)
        ax.text(pos[old_goal_disp][0], pos[old_goal_disp][1]-15, "Old Goal", fontsize=8, color='goldenrod', ha='center')

    # 6. Current Goal
    nx.draw_networkx_nodes(G, pos, nodelist=[inputs['goal']], node_color='red',
                           node_size=800, node_shape='*', label='Current Goal', ax=ax)
    
    # 7. New Sequence (The Result)
    node_color_hex = '#ADD8E6' 
    for i, node in enumerate(sequence):
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color_hex, 
                               node_size=450, edgecolors='black', ax=ax)
        ax.text(pos[node][0], pos[node][1]+15, str(i+1), fontsize=10, 
                 fontweight='bold', color='black', zorder=10, ha='center')

    # Titles and Legends
    title_text = f"Rat: {inputs['rat_id']} | Day: {inputs['day']}\nCurrent Goal: {inputs['goal']}"
    if extra_info:
        title_text += f"\nTrial 1 Analysis: {extra_info}"
        
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.invert_yaxis()

    return fig

# ==========================================
# 4. EXCEL-EXPORT HELPERS
# ==========================================

# Island number -> letter code used in the 'Raw' worksheet.
ISLAND_MAP = {1: 'i', 2: 'j', 3: 'h', 4: 'e'}

# Column order MUST match the 'Raw' worksheet (columns A..U) so the block
# can be pasted straight into the sheet with everything lined up.
RAW_COLUMNS = [
    'Date', 'Time', 'Experimenter', 'Project', 'Training_order', 'Implant',
    'subject', 'day', 'session', 'type', 'repeat', 'trial', 'trial_type',
    'goal_island', 'goal_node', 'goal_island_n', 'goal_node_n',
    'start_island', 'start_node', 'start_island_n', 'start_node_n',
]

def node_codes(node_id, G):
    """Return (island_letter, node_code, island_n, node_n) for a node id.
    e.g. '120' -> ('i', 'i20', 1, 120)."""
    grp = G.nodes[node_id]['group']
    letter = ISLAND_MAP.get(grp, '?')
    code = f"{letter}{node_id[1:]}"
    return letter, code, grp, int(node_id)

def assign_trial_types(is_ephys, sess_key, n):
    """Per-trial 'trial_type' labels following the lab protocol.

    Ephys (30 trials):
        Normal -> 1,16,30 = 4 ; else 1
        PT     -> trial 1 = 6 ; 16,30 = 4 ; else 1
        NGL    -> trial 1 = 4 ; trial 16 = 5 (goal switch) ; trial 30 = 4 ; else 1
    Non-ephys (20 trials):
        Normal -> all 1
        NGL    -> trial 1 = 2 ; else 1
        PT     -> trial 1 = 3 ; else 1
    """
    tt = [1] * n
    if is_ephys:
        mid = n // 2  # 0-indexed position of trial 16 when n == 30
        tt[0] = 4
        tt[mid] = 4
        tt[-1] = 4
        if sess_key == 'PT':
            tt[0] = 6
        elif sess_key == 'NGL':
            tt[mid] = 5
    else:
        if sess_key == 'NGL':
            tt[0] = 2
        elif sess_key == 'PT':
            tt[0] = 3
    return tt

# ==========================================
# MAIN APP UI
# ==========================================

# --- Sidebar: File Upload ---
st.sidebar.header("1. Setup")
uploaded_file = st.sidebar.file_uploader("Upload 'node_list_new.csv'", type=['csv'])

if not uploaded_file:
    st.info("👋 Please upload the 'node_list_new.csv' file in the sidebar to begin.")
    st.stop()

# Load Graph
G = load_graph(uploaded_file)
if G is None:
    st.error("Failed to load graph.")
    st.stop()

# --- Main Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Experiment Details")
    rat_id = st.text_input("Rat ID", value="R01")
    day = st.text_input("Experiment Day (label for files/plot)", value="Day_01")

    # --- Rat type & session type drive trial count and trial_type labels ---
    rat_type = st.radio("Rat type", ["Ephys (implanted)", "Non-ephys"], horizontal=True)
    is_ephys = rat_type.startswith("Ephys")

    session_type = st.radio(
        "Session type",
        ["Normal", "NGL (New Goal Location)", "PT (Probe Trial)"],
        horizontal=True,
    )
    sess_key = "NGL" if session_type.startswith("NGL") else ("PT" if session_type.startswith("PT") else "Normal")

    num_trials = 30 if is_ephys else 20
    st.caption(f"➡️ {num_trials} trials will be generated ({'ephys' if is_ephys else 'non-ephys'}).")

    # Dynamic Goal Selection
    all_nodes = sorted(list(G.nodes()), key=lambda x: int(x))
    goal_index = all_nodes.index('118') if '118' in all_nodes else 0

    if is_ephys and sess_key == "NGL":
        goal = st.selectbox("OLD Goal Node (trials 1–15)", options=all_nodes, index=goal_index)
        new_goal = st.selectbox("NEW Goal Node (trials 16–30)", options=all_nodes, index=goal_index)
        pt_old_goal = None
    elif sess_key == "PT":
        goal = st.selectbox("Current Goal Node ID", options=all_nodes, index=goal_index)
        new_goal = None
        pt_old_goal = st.selectbox(
            "OLD Goal Node (probe reference)",
            options=all_nodes, index=goal_index,
            help="Trial 1 start is chosen to be equidistant from this old goal and the current goal.",
        )
    else:
        goal = st.selectbox("Current Goal Node ID", options=all_nodes, index=goal_index)
        new_goal = None
        pt_old_goal = None

    goal_group = G.nodes[goal]['group']
    st.write(f"📍 *Goal is in Island: {goal_group}*")

    st.subheader("3. Previous History")
    prev_first_node = st.selectbox("Prev Session: First Start Node (Optional)", options=[""] + all_nodes, index=0)
    prev_last_node = st.selectbox("Prev Session: Last Start Node (Optional)", options=[""] + all_nodes, index=0)
    prev_goal_node = st.selectbox("Prev Session: Goal Node (Optional)", options=[""] + all_nodes, index=0)

    prev_used_str = st.text_area("Prev Session: ALL Start Nodes (Copy from Excel)", value="", height=120, help="Paste a column from Excel directly.")

    # NGL/PT sessions trigger the hidden trial-1 distance rule (vs. previous goal).
    is_ngl_pt = sess_key in ("NGL", "PT")

    st.subheader("4. Excel Paste Settings")
    with st.expander("Session metadata for the paste-ready 'Raw' table", expanded=True):
        exp_date = st.text_input("Date (dd.mm.yyyy)", value=datetime.date.today().strftime('%d.%m.%Y'))
        experimenter = st.text_input("Experimenter", value="")
        mc1, mc2 = st.columns(2)
        with mc1:
            m_subject = st.number_input("subject", value=3, step=1)
            m_day = st.number_input("day (number)", value=1, step=1)
            m_session = st.number_input("session", value=1, step=1)
            m_repeat = st.number_input("repeat", value=1, step=1)
        with mc2:
            m_project = st.number_input("Project", value=3, step=1)
            m_training_order = st.number_input("Training_order", value=6 if is_ephys else 1, step=1)
            m_type = st.number_input("type", value=1, step=1)
        m_implant = 1 if is_ephys else 0
        st.caption(f"Implant will be set to **{m_implant}** ({'ephys' if is_ephys else 'non-ephys'}).")
        include_header = st.checkbox("Include header row in paste block", value=False)

with col2:
    st.subheader("5. Generate")
    if st.button("🚀 Generate Sequence", type="primary"):

        # --- PARSE EXCEL PASTE / NEWLINES ---
        cleaned_str = prev_used_str.replace('\n', ',').replace('\r', ',')
        prev_used = [x.strip() for x in cleaned_str.split(',') if x.strip()]

        # Resolve groups safely
        prev_first_grp = G.nodes[prev_first_node]['group'] if prev_first_node else None
        prev_last_grp = G.nodes[prev_last_node]['group'] if prev_last_node else None
        prev_goal_grp = G.nodes[prev_goal_node]['group'] if prev_goal_node else None

        # Trial-1 distance reference ("old goal"): PT uses its dedicated OLD goal
        # input; every other session type falls back to the previous session goal.
        t1_ref_goal = pt_old_goal if (sess_key == "PT" and pt_old_goal) else (prev_goal_node if prev_goal_node else None)
        t1_ref_group = G.nodes[t1_ref_goal]['group'] if t1_ref_goal else None

        def make_inputs(the_goal, n_tr, extra_used, ngl_flag):
            return {
                "rat_id": rat_id,
                "day": day,
                "num_trials": int(n_tr),
                "goal": the_goal,
                "goal_group": G.nodes[the_goal]['group'],
                "prev_first_node": prev_first_node if prev_first_node else None,
                "prev_first_group": prev_first_grp,
                "prev_last_node": prev_last_node if prev_last_node else None,
                "prev_last_group": prev_last_grp,
                "prev_goal_node": prev_goal_node if prev_goal_node else None,
                "prev_goal_group": prev_goal_grp,
                "t1_ref_goal": t1_ref_goal,
                "t1_ref_group": t1_ref_group,
                "old_goal_display": pt_old_goal if (sess_key == "PT" and pt_old_goal) else None,
                "prev_used_nodes": prev_used + extra_used,
                "is_ngl_pt": ngl_flag,
            }

        # --- Run Logic (NGL ephys = two goals split 15/15) ---
        with st.spinner("Calculating optimal paths..."):
            if is_ephys and sess_key == "NGL":
                half = num_trials // 2
                seq1, debug_info = generate_sequence(G, make_inputs(goal, half, [], is_ngl_pt))
                seq2 = None
                if seq1:
                    # Second half: new goal, exclude first-half nodes, no trial-1 hidden rule.
                    seq2, _ = generate_sequence(G, make_inputs(new_goal, num_trials - half, seq1, False))
                if seq1 and seq2:
                    sequence = seq1 + seq2
                    per_trial_goal = [goal] * half + [new_goal] * (num_trials - half)
                else:
                    sequence, per_trial_goal = None, None
            else:
                sequence, debug_info = generate_sequence(G, make_inputs(goal, num_trials, [], is_ngl_pt))
                per_trial_goal = [goal] * num_trials if sequence else None

        if sequence:
            st.success("Sequence generated successfully!")

            # --- Build paste-ready table matching the 'Raw' worksheet (cols A..U) ---
            trial_types = assign_trial_types(is_ephys, sess_key, len(sequence))

            rows = []
            for i, node_id in enumerate(sequence):
                s_letter, s_code, s_grp, s_num = node_codes(node_id, G)
                g_letter, g_code, g_grp, g_num = node_codes(per_trial_goal[i], G)
                rows.append({
                    'Date': exp_date,
                    'Time': '',
                    'Experimenter': experimenter,
                    'Project': m_project,
                    'Training_order': m_training_order,
                    'Implant': m_implant,
                    'subject': m_subject,
                    'day': m_day,
                    'session': m_session,
                    'type': m_type,
                    'repeat': m_repeat,
                    'trial': i + 1,
                    'trial_type': trial_types[i],
                    'goal_island': g_letter,
                    'goal_node': g_code,
                    'goal_island_n': g_grp,
                    'goal_node_n': g_num,
                    'start_island': s_letter,
                    'start_node': s_code,
                    'start_island_n': s_grp,
                    'start_node_n': s_num,
                })

            df_out = pd.DataFrame(rows, columns=RAW_COLUMNS)

            # --- Display Results ---
            st.dataframe(df_out, use_container_width=True, hide_index=True)

            # --- Paste-ready block: tab-separated so it drops straight into Excel ---
            st.markdown("#### 📋 Copy → paste into Excel (Raw sheet)")
            st.caption(
                "Click in the box, press Ctrl/Cmd+A to select all, copy, then paste into "
                "the first empty row of the 'Raw' sheet. Columns line up with A–U."
            )
            tsv = df_out.to_csv(sep='\t', index=False, header=include_header)
            st.text_area("Paste-ready table (tab-separated)", value=tsv, height=280)

            # --- Protocol compliance report ---
            compliance = evaluate_protocol_compliance(
                G,
                sequence,
                goal_node=goal,
                prev_goal_node=prev_goal_node,
                prev_goal_group=prev_goal_grp,
                current_goal_group=goal_group,
                prev_used_nodes=prev_used,
                goal_group=goal_group,
                is_ngl_pt=is_ngl_pt,
                prev_last_group=prev_last_grp,
                min_distance_from_goal=4,
            )
            if compliance:
                st.subheader("Protocol compliance check")
                failed_items = [item for item in compliance if not item['passed']]
                for item in compliance:
                    icon = "✅" if item['passed'] else "❌"
                    color = "green" if item['passed'] else "red"
                    st.markdown(f"<div style='color:{color};font-weight:600'>{icon} {item['name']}: {item['details']}</div>", unsafe_allow_html=True)

                if failed_items:
                    replacement_candidates = []
                    for item in failed_items:
                        if item['name'] == 'distance_to_goal':
                            replacement_candidates.extend(sequence)
                        elif item['name'] == 'avoid_previous_session_nodes':
                            replacement_candidates.extend([node for node in sequence if node in prev_used_nodes or node == prev_goal_node])
                        elif item['name'] == 'start_island_differs_from_prev_last':
                            replacement_candidates.append(sequence[0])
                    replacement_candidates = sorted(set(replacement_candidates))
                    if replacement_candidates:
                        st.warning(f"Manual replacement candidates: {', '.join(replacement_candidates)}")
                    else:
                        st.info("No obvious manual replacement candidates were identified from the failed checks.")

            # --- Plot ---
            plot_inputs = make_inputs(goal, num_trials, [], is_ngl_pt)
            fig = create_plot(G, sequence, plot_inputs, debug_info)
            st.pyplot(fig)

            # --- Download Buttons ---
            c1, c2, c3 = st.columns(3)

            # CSV Download (always includes header)
            csv = df_out.to_csv(index=False).encode('utf-8')
            c1.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"start_nodes_{rat_id}_{day}.csv",
                mime="text/csv",
            )

            # Excel Download (same columns as the Raw sheet); degrade gracefully.
            try:
                xlsx_buffer = io.BytesIO()
                with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
                    df_out.to_excel(writer, index=False, sheet_name='start_nodes')
                xlsx_buffer.seek(0)
                c2.download_button(
                    label="📊 Excel",
                    data=xlsx_buffer,
                    file_name=f"start_nodes_{rat_id}_{day}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                c2.caption("Excel export needs `openpyxl`.")

            # Image Download
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            c3.download_button(
                label="🖼️ Map",
                data=img_buffer,
                file_name=f"hexmaze_map_{rat_id}_{day}.png",
                mime="image/png",
            )

        else:
            # Error message is handled in the logic function
            pass