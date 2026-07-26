import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import math
import datetime

from protocol_utils import (
    HARD_EXCLUDED_NODES,
    MIN_DISTANCE_FROM_GOAL,
    build_island_sequence,
    build_maze_graph,
    choose_first_trial_group,
    evaluate_protocol_compliance,
    island_capacity,
    select_probe_start_node,
    select_session_nodes,
)

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

    df = pd.read_csv(uploaded_file, header=None, names=['id', 'x', 'y'])
    G, missing_edges = build_maze_graph(df.itertuples(index=False, name=None))

    for u, v in missing_edges:
        st.warning(f"⚠️ Warning: Manual bridge node missing in CSV: {u}-{v}")

    return G

# ==========================================
# 2. LOGIC GENERATOR
# ==========================================
ALL_GROUPS = [1, 2, 3, 4]

# How many independent island orders to try before declaring the constraints
# unsatisfiable. A retry re-rolls the island sequence and every random choice.
MAX_SEQUENCE_ATTEMPTS = 40

# How many of the previous session's start nodes may be reused when an island
# cannot supply enough fresh ones (sessions longer than 20 trials). Taken from
# the top of the list. The full 20 are offered because the shortfall lands on
# one island, and only about a quarter of the list sits on any given island -
# the first ten rarely reach the three or four nodes a bridge goal needs.
MAX_REUSED_START_NODES = 20


def generate_sequence(G, inputs):
    """Build one session's start-node sequence.

    Returns ``(sequence, info)``. ``info`` collects the trial-1 analysis text,
    any relaxations that were needed and an error message, so that this function
    stays free of Streamlit calls and can be retried without spamming the UI.
    """
    # An ephys NGL session runs on two goals (15 trials each); every other
    # session is a single segment.
    segments = inputs['goal_segments']
    total_trials = sum(n for _, n in segments)
    per_trial_goals = [g for g, n in segments for _ in range(n)]

    # Trial-1 rules and the capacity diagnostics use the session's first goal.
    goal_node = segments[0][0]
    goal_group = G.nodes[goal_node]['group']
    session_kind = inputs['session_kind']

    info = {'t1_text': '', 'notes': [], 'error': None}
    distance_cache = {}

    # Nodes that may never be picked this session.
    forbidden = set(HARD_EXCLUDED_NODES) | {g for g, _ in segments}

    # Rules the protocol qualifies with "do not use ... " but which it also says
    # to give way on when they clash with the non-adjacency rule.
    soft_avoid = set(inputs['prev_used_nodes'])
    if inputs['prev_goal_node']:
        soft_avoid.add(inputs['prev_goal_node'])

    # Past 20 trials the islands can run out of nodes that satisfy the spacing
    # rules. The lab's rule for that is to reuse the previous session's start
    # nodes, taking them from the top of the list.
    fallback_nodes = list(inputs['prev_used_nodes'])[:MAX_REUSED_START_NODES]

    # ---------------------------------------------------------
    # STEP 1: Trial 1 (NGL / PT rules)
    # ---------------------------------------------------------
    forced_t1_node = None
    forced_t1_group = None

    if session_kind == 'PT' and inputs['t1_ref_goal']:
        forced_t1_node, probe_info = select_probe_start_node(
            G,
            goal_node=goal_node,
            prev_goal_node=inputs['t1_ref_goal'],
            forbidden_set=forbidden,
            soft_avoid=soft_avoid,
            prev_last_group=inputs['prev_last_group'],
            prev_first_group=inputs['prev_first_group'],
            island_demand=math.ceil(total_trials / 3),
            distance_cache=distance_cache,
        )
        if forced_t1_node:
            forced_t1_group = G.nodes[forced_t1_node]['group']
            info['t1_text'] = (
                f" (Dist New: {probe_info['dist_new']}, "
                f"Dist Old: {probe_info['dist_old']}, Diff: {probe_info['diff']})"
            )
            if 'start_island_topology' in probe_info['relaxed']:
                info['notes'].append(
                    "No island is directly connected to both the current and the old goal "
                    "island; trial 1 fell back to any non-goal island."
                )
            if 'equidistance' in probe_info['relaxed']:
                info['notes'].append(
                    "No start node was equidistant (within 2 nodes) from both goals; "
                    "the closest available match was used. Check trial 1 by hand."
                )
            if 'first_trial_island_differs_from_prev_last' in probe_info['relaxed']:
                info['notes'].append(
                    "Every island that is valid for this probe is the island the previous "
                    "session ended on; trial 1 had to reuse it. Check trial 1 by hand."
                )
        else:
            info['notes'].append(
                "No probe start node satisfied the equidistance rule; trial 1 was selected normally."
            )

    elif session_kind == 'NGL':
        forced_t1_group = choose_first_trial_group(
            G,
            goal_group=goal_group,
            available_groups=ALL_GROUPS,
            prev_goal_group=inputs['prev_goal_group'],
            prev_last_group=inputs['prev_last_group'],
            prev_first_group=inputs['prev_first_group'],
        )

    # ---------------------------------------------------------
    # STEP 2 + 3: Island order, then nodes. Retry on a dead end.
    # ---------------------------------------------------------
    for _ in range(MAX_SEQUENCE_ATTEMPTS):
        # Each goal segment gets its own block structure; the second one only
        # has to not repeat the island the first one ended on.
        island_sequence = []
        for index, (segment_goal, segment_trials) in enumerate(segments):
            first_segment = index == 0
            island_sequence.extend(build_island_sequence(
                total_trials=segment_trials,
                available_groups=ALL_GROUPS,
                goal_group=G.nodes[segment_goal]['group'],
                forced_t1_group=forced_t1_group if first_segment else None,
                prev_last_group=(
                    inputs['prev_last_group'] if first_segment else island_sequence[-1]
                ),
                prev_first_group=inputs['prev_first_group'] if first_segment else None,
            ))

        if len(island_sequence) != total_trials:
            info['error'] = "Could not construct a valid island sequence from the protocol rules."
            return None, info

        allocation = select_session_nodes(
            G,
            island_sequence=island_sequence,
            per_trial_goals=per_trial_goals,
            forbidden_set=forbidden,
            soft_avoid=soft_avoid,
            min_distance_from_goal=MIN_DISTANCE_FROM_GOAL,
            forced_first_node=forced_t1_node,
            fallback_nodes=fallback_nodes,
            distance_cache=distance_cache,
        )

        if allocation:
            sequence = allocation.sequence
            info['reused'] = allocation.borrowed
            if info['reused']:
                info['notes'].append(
                    f"{len(info['reused'])} start node(s) were reused from the previous "
                    f"session because the islands cannot supply {total_trials} nodes that "
                    f"are all >= {MIN_DISTANCE_FROM_GOAL} from the goal and non-adjacent: "
                    + ", ".join(info['reused'])
                )
            return sequence, info

    # Explain *why* it failed: compare what each island can supply against what
    # the block structure asks of it.
    needed_per_island = math.ceil(total_trials / 3)
    capacities = {
        island: island_capacity(
            G, island=island, goal_node=goal_node, forbidden_set=forbidden,
            min_distance_from_goal=MIN_DISTANCE_FROM_GOAL, distance_cache=distance_cache,
        )
        for island in ALL_GROUPS if island != goal_group
    }
    short = {i: c for i, c in capacities.items() if c < needed_per_island}

    if short:
        info['error'] = (
            f"Goal {goal_node} cannot support {total_trials} trials. A {total_trials}-trial "
            f"session needs about {needed_per_island} start nodes per island, but island(s) "
            + ", ".join(f"{i} (max {c})" for i, c in sorted(short.items()))
            + f" cannot supply that many nodes that are both >= {MIN_DISTANCE_FROM_GOAL} steps "
            f"from the goal and non-adjacent to each other. Pick a different goal node."
        )
    else:
        info['error'] = (
            f"Could not satisfy the protocol after {MAX_SEQUENCE_ATTEMPTS} attempts. Island "
            f"capacities are {capacities} against {needed_per_island} needed per island, so "
            f"this is very tight. Try generating again, or reduce the list of "
            f"previous-session start nodes."
        )
    return None, info

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
    existing_exclusions = [n for n in HARD_EXCLUDED_NODES if n in G.nodes()]
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

    # Dynamic Goal Selection. The hard-excluded nodes are never used as a start
    # *or* a goal location, so they are not offered here either.
    all_nodes = sorted(list(G.nodes()), key=lambda x: int(x))
    goal_options = [n for n in all_nodes if n not in HARD_EXCLUDED_NODES]
    goal_index = goal_options.index('118') if '118' in goal_options else 0

    if is_ephys and sess_key == "NGL":
        goal = st.selectbox("OLD Goal Node (trials 1–15)", options=goal_options, index=goal_index)
        # Default the new goal to a different island, since a New Goal Location
        # session by definition moves the goal.
        new_default = next(
            (i for i, n in enumerate(goal_options)
             if G.nodes[n]['group'] != G.nodes[goal_options[goal_index]]['group']),
            goal_index,
        )
        new_goal = st.selectbox("NEW Goal Node (trials 16–30)", options=goal_options, index=new_default)
        pt_old_goal = None
    elif sess_key == "PT":
        goal = st.selectbox("Current Goal Node ID", options=goal_options, index=goal_index)
        new_goal = None
        pt_old_goal = st.selectbox(
            "OLD Goal Node (probe reference)",
            options=goal_options, index=goal_index,
            help="Trial 1 start is chosen to be equidistant from this old goal and the current goal.",
        )
    else:
        goal = st.selectbox("Current Goal Node ID", options=goal_options, index=goal_index)
        new_goal = None
        pt_old_goal = None

    goal_group = G.nodes[goal]['group']
    st.write(f"📍 *Goal is in Island: {goal_group}*")

    if new_goal is not None and new_goal == goal:
        st.error(
            "The NEW goal must differ from the OLD goal — a New Goal Location session "
            "moves the goal. Pick a different NEW goal node."
        )
        st.stop()

    st.subheader("3. Previous History")
    prev_first_node = st.selectbox("Prev Session: First Start Node (Optional)", options=[""] + all_nodes, index=0)
    prev_last_node = st.selectbox("Prev Session: Last Start Node (Optional)", options=[""] + all_nodes, index=0)
    prev_goal_node = st.selectbox("Prev Session: Goal Node (Optional)", options=[""] + all_nodes, index=0)

    prev_used_str = st.text_area("Prev Session: ALL Start Nodes (Copy from Excel)", value="", height=120, help="Paste a column from Excel directly.")

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

        # Probe reference ("old goal"): PT uses its dedicated OLD goal input;
        # every other session type falls back to the previous session goal.
        t1_ref_goal = pt_old_goal if (sess_key == "PT" and pt_old_goal) else (prev_goal_node if prev_goal_node else None)

        def make_inputs(segments, kind):
            return {
                "rat_id": rat_id,
                "day": day,
                "goal_segments": segments,
                # The first segment's goal; used for the plot title and the
                # trial-1 rules. Later segments come from goal_segments.
                "goal": segments[0][0],
                "session_kind": kind,
                "prev_first_node": prev_first_node if prev_first_node else None,
                "prev_first_group": prev_first_grp,
                "prev_last_node": prev_last_node if prev_last_node else None,
                "prev_last_group": prev_last_grp,
                "prev_goal_node": prev_goal_node if prev_goal_node else None,
                "prev_goal_group": prev_goal_grp,
                "t1_ref_goal": t1_ref_goal,
                "old_goal_display": pt_old_goal if (sess_key == "PT" and pt_old_goal) else None,
                # Soft: last session's start nodes, which the protocol lets
                # give way when they clash with the non-adjacency rule.
                "prev_used_nodes": prev_used,
            }

        # --- Run Logic (NGL ephys = two goals split 15/15) ---
        with st.spinner("Calculating optimal paths..."):
            # An ephys NGL session splits 15/15 across the old and new goal;
            # it is still one session, so the nodes are allocated in one pass.
            if is_ephys and sess_key == "NGL":
                half = num_trials // 2
                segments = [(goal, half), (new_goal, num_trials - half)]
            else:
                segments = [(goal, num_trials)]

            sequence, gen_info = generate_sequence(G, make_inputs(segments, sess_key))
            per_trial_goal = [g for g, n in segments for _ in range(n)] if sequence else None

        debug_info = gen_info.get('t1_text', '')
        for note in gen_info.get('notes', []):
            st.caption(f"Note: {note}")
        if gen_info.get('error'):
            st.error(gen_info['error'])

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
                per_trial_goals=per_trial_goal,
                session_kind=sess_key,
                prev_goal_node=prev_goal_node if prev_goal_node else None,
                probe_reference_goal=t1_ref_goal if sess_key == "PT" else None,
                prev_used_nodes=prev_used,
                reused_nodes=gen_info.get('reused', []),
                prev_first_group=prev_first_grp,
                prev_last_group=prev_last_grp,
                min_distance_from_goal=MIN_DISTANCE_FROM_GOAL,
            )
            if compliance:
                st.subheader("Protocol compliance check")
                hard_failures = [i for i in compliance if not i['passed'] and i['severity'] == 'hard']
                soft_failures = [i for i in compliance if not i['passed'] and i['severity'] == 'soft']

                for item in compliance:
                    if item['passed']:
                        icon, color = "✅", "green"
                    elif item['severity'] == 'soft':
                        icon, color = "⚠️", "orange"
                    else:
                        icon, color = "❌", "red"
                    st.markdown(
                        f"<div style='color:{color};font-weight:600'>{icon} {item['name']}: {item['details']}</div>",
                        unsafe_allow_html=True,
                    )

                if hard_failures:
                    st.error(
                        "This sequence breaks a mandatory rule and must not be used as-is: "
                        + ", ".join(i['name'] for i in hard_failures)
                    )
                elif soft_failures:
                    st.warning(
                        "All mandatory rules pass. The protocol allows these to give way when "
                        "they cannot all be met at once: "
                        + ", ".join(i['name'] for i in soft_failures)
                    )
                else:
                    st.success("Every protocol rule passes.")

            # --- Plot ---
            plot_inputs = make_inputs(segments, sess_key)
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