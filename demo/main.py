import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import umap
import pandas as pd
import plotly.express as px

from memory.episodic import EpisodicMemory
from demo.environment import LoopEnv


@st.cache()
def generate_loop_memories(episodic_memory,
                           n_steps,
                           vector_dim,
                           n_locations,
                           actions):

    env = LoopEnv(state_dim=vector_dim,
                  n_locations=n_locations,
                  actions=actions)
    n_update_history = n_steps // 100

    history = {'step': [],
               'weight': [],
               'stability': [],
               'oldness': [],
               'state_tm1': [],
               'action': [],
               'state': [],
               'id': []}

    for action in actions:
        history['step'].append(0)
        history['id'].append(-1)
        history['state_tm1'].append(-1)
        history['action'].append(action)
        history['state'].append(-1)
        history['weight'].append(0)
        history['stability'].append(stability_start)
        history['oldness'].append(0)

    for i in range(1, n_steps+1):
        state_m1, action, new_state = env.run()
        episodic_memory.update(state_m1, action, new_state)

        # update history
        if i % n_update_history == 0:
            for sequence in episodic_memory.tree_memory.graph.edges(data=True):
                history['step'].append(i+1)
                history['id'].append(f'{sequence[0]}-{sequence[2]["action"]}-{sequence[1]}')

                history['state_tm1'].append(sequence[0])
                history['action'].append(sequence[2]['action'])
                history['state'].append(sequence[1])

                history['weight'].append(sequence[2]['weight'])
                history['stability'].append(sequence[2]['stability'])
                history['oldness'].append(sequence[2]['oldness'])

    return history


def create_state_graph(episodic_memory, n_neighboors):
    states_attributes = episodic_memory.get_states_attributes()

    raw_states = episodic_memory.get_raw_states()
    coords = umap.UMAP(n_neighbors=n_neighboors, n_components=3,
                       metric="l2").fit_transform(list(raw_states.values()))

    data = {'x': [c[0] for c in coords],
            'y': [c[1] for c in coords],
            'z': [c[2] for c in coords],
            'centrality': [max(0.01, s['centrality']) for s in states_attributes.values()],
            'id': list(states_attributes.keys())}
    df = pd.DataFrame.from_dict(data)
    fig = px.scatter_3d(df, x='x', y='y', z='z', size='centrality', hover_data=['centrality', 'id'],
                        width=800, height=700)

    fig.update_layout({
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)'
    })

    return fig


def create_tree_graph(history):
    df = pd.DataFrame.from_dict(history)
    fig = px.scatter(df, x="stability", y="weight", animation_frame="step",
                     size="weight", hover_name="id",
                     color="action",
                     log_x=False, size_max=15,
                     range_x=[min(df['stability'].values)-1, max(df['stability'].values)+1],
                     range_y=[0, 1.2], width=800, height=500)
    return fig


st.title("Pythia Episodic Memory")
st.markdown(
"""
This is the demo of the episodic memory of the self-supervised robot Pythia.
""")


# Memory settings
st.sidebar.header("Memory settings")
st.sidebar.text("""A new memory will be instanciated 
whenever the values below changes
""")
index_percentage_threshold = st.sidebar.slider("index percentage threshold", 0.01, 0.20, step=0.01)
max_size = st.sidebar.slider("memory capacity", 1, int(1e5), step=10)
vector_dim = st.sidebar.slider("state dimension", 3, 300)
stability_start = st.sidebar.slider("stability start", 20, 1000)

@st.cache(allow_output_mutation=True)
def get_memory(index_percentage_threshold, max_size, vector_dim, stability_start):
    return EpisodicMemory(base_path='artifacts',
                          max_size=max_size,
                          index_percentage_threshold=index_percentage_threshold,
                          vector_dim=vector_dim,
                          stability_start=stability_start)

episodic_memory = get_memory(index_percentage_threshold, max_size, vector_dim, stability_start)

st.sidebar.text(f"""The memory as a capacity of {episodic_memory.max_size},
a percentage threshold of {episodic_memory.index_percentage_threshold}
and a sim threshold of {episodic_memory.index_memory.sim_threshold}""")


# Environment settings
st.sidebar.header("Environment settings")

n_steps = st.sidebar.slider("Number of steps to simulate",
                            min_value=1,
                            max_value=10000)
n_locations = st.sidebar.slider("Number of different locations",
                            min_value=1,
                            max_value=100,
                            step=1)

run = st.sidebar.button("Run simulation")
short_memory_descr = st.sidebar.empty()
if run:
    with st.spinner('Running the simulation...'):
        history = generate_loop_memories(episodic_memory,
                                         n_steps=n_steps,
                                         n_locations=n_locations,
                                         vector_dim=vector_dim,
                                         actions=['up', 'down', 'right', 'left'])
    st.success('Done!')
    short_memory_descr.text(f'Number of nodes : {len(episodic_memory.tree_memory.graph.nodes())}\n'
                            f'Number of edges : {len(episodic_memory.tree_memory.graph.edges())}\n'
                            f'Number of forgeted nodes : {episodic_memory.forgeted}')


# main panel
if episodic_memory.index_memory.index is not None and len(episodic_memory) > 4:
    st.subheader("Index memory visualization")
    st.text("""Each point is a state and the more two points are near, 
the more the associated states are similar""")

    n_neighboors = st.slider("Number of neighboors to compute UMAP", 2, 30, step=1)

    state_fig = create_state_graph(episodic_memory, n_neighboors)
    st.plotly_chart(state_fig)


    st.subheader("Tree memory evolution")
    st.text("""Each point is a transition (state_tm1, action, state) the weight (x-axis)
is the strength of the memory element, the stability of a transition defines the speed
at which the element will be forgotten""")

    tree_fig = create_tree_graph(history)
    st.plotly_chart(tree_fig)

