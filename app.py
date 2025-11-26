import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
import io
import itertools

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
    
    # --- HELPER: Get valid node from a specific group ---
    def get_valid_node(target_group, current_selected, graph, forbidden_set, goal_node):
        candidates = [n for n, attr in graph.nodes(data=True) if attr.get('group') == target_group]
        
        # Filter: Forbidden & Already Selected
        candidates = [n for n in candidates if n not in forbidden_set]
        candidates = [n for n in candidates if n not in current_selected]
        
        # Filter: Distance Check
        dist_candidates = []
        for n in candidates:
            try:
                dist = nx.shortest_path_length(graph, source=n, target=goal_node)
                if dist >= MIN_DISTANCE_FROM_GOAL:
                    dist_candidates.append(n)
            except nx.NetworkXNoPath:
                pass 
        candidates = dist_candidates

        if not candidates:
            return None

        # Filter: Dispersion
        dispersion_candidates = []
        for cand in candidates:
            neighbors = list(graph.neighbors(cand))
            if not any(n in current_selected for n in neighbors):
                dispersion_candidates.append(cand)
        
        if dispersion_candidates:
            return random.choice(dispersion_candidates)
        return random.choice(candidates)

    # ---------------------------------------------------------
    # STEP 1: Handle Special Start Node (NGL/PT Logic)
    # ---------------------------------------------------------
    forced_t1_node = None
    t1_dist_info = ""

    if inputs['is_ngl_pt'] and inputs['prev_goal_node']:
        st.info("Applying HIDDEN RULE: NGL/PT Distance Check")
        candidates = []
        for n in G.nodes():
            if n in total_forbidden: continue
            g = G.nodes[n]['group']
            if g == inputs['goal_group']: continue 
            if g == inputs['prev_goal_group']: continue 
            
            try:
                dist_to_curr_goal = nx.shortest_path_length(G, n, inputs['goal'])
                if dist_to_curr_goal < MIN_DISTANCE_FROM_GOAL: continue 

                d_old = nx.shortest_path_length(G, n, inputs['prev_goal_node'])
                diff = abs(dist_to_curr_goal - d_old)
                if diff < 3:
                    candidates.append((n, dist_to_curr_goal, d_old, diff))
            except:
                continue
        
        preferred = [c for c in candidates if G.nodes[c[0]]['group'] != inputs['prev_last_group']]
        selected = random.choice(preferred) if preferred else (random.choice(candidates) if candidates else None)

        if selected:
            forced_t1_node = selected[0]
            t1_dist_info = f" (Dist New: {selected[1]}, Dist Old: {selected[2]}, Diff: {selected[3]})"

    # ---------------------------------------------------------
    # STEP 2: Generate Island Sequence
    # ---------------------------------------------------------
    island_sequence = []
    first_block_pool = list(available_groups)
    
    if forced_t1_node:
        t1_group = G.nodes[forced_t1_node]['group']
        island_sequence.append(t1_group)
        if t1_group in first_block_pool:
            first_block_pool.remove(t1_group)
        random.shuffle(first_block_pool)
        island_sequence.extend(first_block_pool)
    else:
        random.shuffle(first_block_pool)
        # Prevent same start island as prev session last island (unless NGL)
        if not inputs['is_ngl_pt'] and inputs['prev_last_group'] is not None:
            retries = 0
            while first_block_pool[0] == inputs['prev_last_group'] and retries < 50:
                random.shuffle(first_block_pool)
                retries += 1
            if retries > 0 and retries < 50:
                st.caption(f"Note: Adjusted starting island to ensure it differs from Previous Session Last Island ({inputs['prev_last_group']}).")
        island_sequence.extend(first_block_pool)

    while len(island_sequence) < TOTAL_TRIALS:
        block = list(available_groups)
        random.shuffle(block)
        retries = 0
        while block[0] == island_sequence[-1] and retries < 10:
            random.shuffle(block)
            retries += 1
        island_sequence.extend(block)
    
    island_sequence = island_sequence[:TOTAL_TRIALS]

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
    day = st.text_input("Experiment Day", value="Day_01")
    num_trials = st.number_input("Number of Trials to Generate", min_value=5, max_value=50, value=20, step=1)

    # Dynamic Goal Selection
    all_nodes = sorted(list(G.nodes()), key=lambda x: int(x))
    goal_index = all_nodes.index('118') if '118' in all_nodes else 0
    goal = st.selectbox("Current Goal Node ID", options=all_nodes, index=goal_index)
    
    goal_group = G.nodes[goal]['group']
    st.write(f"📍 *Goal is in Island: {goal_group}*")

    st.subheader("3. Previous History")
    prev_first_node = st.selectbox("Prev Session: First Start Node (Optional)", options=[""] + all_nodes, index=0)
    prev_last_node = st.selectbox("Prev Session: Last Start Node (Optional)", options=[""] + all_nodes, index=0)
    prev_goal_node = st.selectbox("Prev Session: Goal Node (Optional)", options=[""] + all_nodes, index=0)
    
    # --- UPDATED INPUT HANDLING ---
    prev_used_str = st.text_area("Prev Session: ALL Start Nodes (Copy from Excel)", value="", height=150, help="Paste a column from Excel directly.")
    is_ngl_pt = st.checkbox("Is this a New Goal Location (NGL) or Probe Trial (PT)?", value=False)

with col2:
    st.subheader("4. Generate")
    if st.button("🚀 Generate Sequence", type="primary"):
        
        # --- PARSE EXCEL PASTE / NEWLINES ---
        # 1. Replace newlines with commas
        # 2. Split by comma
        # 3. Strip whitespace
        cleaned_str = prev_used_str.replace('\n', ',').replace('\r', ',')
        prev_used = [x.strip() for x in cleaned_str.split(',') if x.strip()]
        
        # Resolve groups safely
        prev_first_grp = G.nodes[prev_first_node]['group'] if prev_first_node else None
        prev_last_grp = G.nodes[prev_last_node]['group'] if prev_last_node else None
        prev_goal_grp = G.nodes[prev_goal_node]['group'] if prev_goal_node else None

        inputs = {
            "rat_id": rat_id,
            "day": day,
            "num_trials": int(num_trials),
            "goal": goal,
            "goal_group": goal_group,
            "prev_first_node": prev_first_node if prev_first_node else None,
            "prev_first_group": prev_first_grp,
            "prev_last_node": prev_last_node if prev_last_node else None,
            "prev_last_group": prev_last_grp,
            "prev_goal_node": prev_goal_node if prev_goal_node else None,
            "prev_goal_group": prev_goal_grp,
            "prev_used_nodes": prev_used,
            "is_ngl_pt": is_ngl_pt
        }

        # Run Logic
        with st.spinner("Calculating optimal paths..."):
            sequence, debug_info = generate_sequence(G, inputs)

        if sequence:
            st.success("Sequence generated successfully!")
            
            # Create Dataframe
            island_map = {1: 'i', 2: 'j', 3: 'h', 4: 'e'}
            csv_data = []
            for i, node_id in enumerate(sequence):
                grp_n = G.nodes[node_id]['group']
                grp_char = island_map.get(grp_n, '?')
                node_suffix = node_id[1:]
                node_char = f"{grp_char}{node_suffix}"
                
                csv_data.append({
                    'Trial': i+1,
                    'Start_Island': grp_char,
                    'Start_Node': node_char,
                    'Island_ID': grp_n,
                    'Node_ID': node_id
                })
            
            df_out = pd.DataFrame(csv_data)

            # --- Display Results ---
            st.dataframe(df_out, use_container_width=True)

            # --- Plot ---
            fig = create_plot(G, sequence, inputs, debug_info)
            st.pyplot(fig)

            # --- Download Buttons ---
            c1, c2 = st.columns(2)
            
            # CSV Download
            csv = df_out.to_csv(index=False).encode('utf-8')
            c1.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"start_nodes_{rat_id}_{day}.csv",
                mime="text/csv",
            )
            
            # Image Download
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            c2.download_button(
                label="🖼️ Download Map Image",
                data=img_buffer,
                file_name=f"hexmaze_map_{rat_id}_{day}.png",
                mime="image/png",
            )
            
        else:
            # Error message is handled in the logic function
            pass