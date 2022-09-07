import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def get_unsatisfied_nodes(graph):
    ratio_similar = {}
    for (n, d) in graph.nodes(data=True):
        if d['type'] == 0:
            continue
        ratio_similar[n] = []
        neighbors = list(graph.neighbors(n))
        for neigh in neighbors:
            ratio_similar[n].append(graph.nodes[n]["type"] == graph.nodes[neigh]["type"])

        ratio_similar[n] = np.mean(ratio_similar[n])

    unsatisfied_nodes = []

    for (k, v) in ratio_similar.items():
        if v < THRESHOLD:
            unsatisfied_nodes.append(k)

    return ratio_similar, unsatisfied_nodes


def label_unsatisfied(graph, unsatisfied_nodes):
    labels = {}
    for n in graph.nodes():
        if n in unsatisfied_nodes:
            labels[n] = 'x'
        else:
            labels[n] = ''

    return labels


def plot_graph(g, p, show_labels, nodes_size):
    empty_nodes = [n for (n,d) in g.nodes(data=True) if d['type'] == 0]
    type1_nodes = [n for (n,d) in g.nodes(data=True) if d['type'] == 1]
    type2_nodes = [n for (n,d) in g.nodes(data=True) if d['type'] == 2]
    if N_CLASSES >= 3:
        type3_nodes = [n for (n, d) in g.nodes(data=True) if d['type'] == 3]
    if N_CLASSES >= 4:
        type4_nodes = [n for (n, d) in g.nodes(data=True) if d['type'] == 4]

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(empty_nodes) != 0:
        nx.draw_networkx_nodes(g, p, node_color='white', nodelist=empty_nodes, node_size=nodes_size)
    nx.draw_networkx_nodes(g, p, node_color='red', nodelist=type2_nodes, node_size=nodes_size)
    nx.draw_networkx_nodes(g, p, node_color='green', nodelist=type1_nodes, node_size=nodes_size)


    if N_CLASSES >= 3:
        nx.draw_networkx_nodes(g, p, node_color='orange', nodelist=type3_nodes, node_size=nodes_size)
    if N_CLASSES >= 4:
        nx.draw_networkx_nodes(g, p, node_color='lightblue', nodelist=type4_nodes, node_size=nodes_size)

    _, unsatisfied_nodes = get_unsatisfied_nodes(graph)


    nx.draw_networkx_edges(g, p, width=0.5, alpha=0.7)
    if show_labels:
        labels = label_unsatisfied(graph, unsatisfied_nodes)
        nx.draw_networkx_labels(g, p, labels=labels, font_color='black')
    return fig



st.set_page_config(
    page_title='PhD Computational Task', layout="centered")
st.title('PhD Computational Task')
st.markdown('Author: Ruslan Klymentiev')

algorithm = st.sidebar.selectbox(
    label='Graph type:',
    options=('Grid World', 'Erdos-Renyi Graph'),
    index=0)

if algorithm == 'Grid World':
    N = st.sidebar.slider(
        label='Grid size (NxN):',
        value=20, min_value=10,
        max_value=50, step=1)

    PERC_EMPTY = st.sidebar.slider(
        label='Ratio of empty nodes:',
        value=0.1, min_value=.1,
        max_value=.9, step=.1)

    n_nodes = N*N
    nodes_size = ((N - 10) / (50-10)) * (50-400) + 400

    N_CLASSES = int(st.sidebar.number_input(
        label='Number of classes:',
        value=2, min_value=2,
        max_value=4, step=1))
    #
    # THRESHOLD = st.sidebar.slider(
    #     label='Threshold:',
    #     value=0.3, min_value=0.,
    #     max_value=1., step=0.1)
    #
    # PERC_EMPTY = st.sidebar.slider(
    #     label='Percentage of empty nodes:',
    #     value=0.1, min_value=.1,
    #     max_value=.9, step=.1)
    #
    # N_STEPS = st.sidebar.slider(
    #     label='Number of simulation steps:',
    #     value=5, min_value=1,
    #     max_value=20, step=1)
    #
    # show_labels = st.sidebar.checkbox('Highlight Unsatisfied Nodes?', value=True)

    graph = nx.grid_2d_graph(N, N)
    pos = dict((i, i) for i in graph.nodes())

    for n in graph.nodes():
        graph.nodes[n]['type'] = np.random.choice(range(N_CLASSES+1), p=[PERC_EMPTY]+[(1-PERC_EMPTY)/N_CLASSES]*N_CLASSES)

    empty_cell_list = [n for (n, d) in graph.nodes(data=True) if d['type'] == 0]

    for ((u, v), d) in graph.nodes(data=True):
        if (u + 1 <= N - 1) and (v + 1 <= N - 1):
            graph.add_edge((u, v), (u + 1, v + 1))

    for ((u, v), d) in graph.nodes(data=True):
        if (u + 1 <= N - 1) and (v - 1 >= 0):
            graph.add_edge((u, v), (u + 1, v - 1))

    all_neighbors = {}
    for n in graph.nodes():
        all_neighbors[n] = list(graph.neighbors(n))

    for (n, d) in graph.nodes(data=True):
        if d['type'] == 0:
            neighbors = list(graph.neighbors(n))
            for neigh in neighbors:
                graph.remove_edge(n, neigh)

elif algorithm == 'Erdos-Renyi Graph':
    n_nodes = st.sidebar.slider(
        label='Number of nodes:',
        value=100, min_value=50,
        max_value=200, step=25)

    prob = st.sidebar.slider(
        label='Probability of edge creation:',
        value=0.5, min_value=0.,
        max_value=1., step=0.1)

    N_CLASSES = int(st.sidebar.number_input(
        label='Number of classes:',
        value=2, min_value=2,
        max_value=4, step=1))

    graph = nx.erdos_renyi_graph(n=n_nodes, p=prob)
    pos = nx.spring_layout(graph, seed=1)

    for n in graph.nodes():
        graph.nodes[n]['type'] = np.random.choice(range(1,N_CLASSES+1))

    nodes_size = 300


THRESHOLD = st.sidebar.slider(
    label='Threshold:',
    value=0.3, min_value=0.,
    max_value=1., step=0.1)

N_STEPS = st.sidebar.slider(
    label='Number of simulation steps:',
    value=5, min_value=1,
    max_value=20, step=1)

show_labels = st.sidebar.checkbox('Highlight unsatisfied nodes?', value=True)
st.sidebar.write('')





tab1, tab2 = st.tabs(["Initial Graph", "Simulations"])

with tab1:
    st.pyplot(plot_graph(graph, pos, show_labels, nodes_size))
    ratio_similar, unsatisfied_nodes = get_unsatisfied_nodes(graph)
    st.write(f"""**Number of unsatisfied nodes**: {len(unsatisfied_nodes)} out of {n_nodes} 
    ({len(unsatisfied_nodes) * 100 / (n_nodes):.2f}%)""")
    st.write(f"""**Average similarity ratio**: {np.nanmean(list(ratio_similar.values())):.2f}""")

with tab2:
    by_step = st.radio(
        "Visualization of results",
        ('Step by step', 'Just the final result'),
        help="Warning: step by step visualization can take a long time to run.")

    run = st.button('Run the simulation')

    f = st.empty()
    if run:

        for _ in range(N_STEPS):

            _, unsatisfied_nodes = get_unsatisfied_nodes(graph)


            for n in unsatisfied_nodes:

                if algorithm == 'Grid World':
                    new_pos = empty_cell_list[np.random.choice(len(empty_cell_list))]
                    empty_cell_list.remove(new_pos)
                    empty_cell_list.append(n)
                    neighbors = list(graph.neighbors(n))
                    for neigh in neighbors:
                        graph.remove_edge(n, neigh)

                    graph.nodes(data=True)[new_pos]['type'] = graph.nodes(data=True)[n]['type']
                    graph.nodes(data=True)[n]['type'] = 0

                    for neigh in all_neighbors[new_pos]:
                        if graph.nodes(data=True)[neigh]['type'] != 0:
                            graph.add_edge(new_pos, neigh)
                elif algorithm == 'Erdos-Renyi Graph':
                    neighbors = list(graph.neighbors(n))
                    for neigh in neighbors:
                        graph.remove_edge(n, neigh)

                    all_nodes = list(graph.nodes())
                    all_nodes.remove(n)
                    for n2 in all_nodes:
                        if np.random.random() < prob:
                            graph.add_edge(n, n2)

            if by_step == 'Step by step':
                f.pyplot(plot_graph(graph, pos, show_labels, nodes_size))

        if by_step == 'Just the final result':
            f.pyplot(plot_graph(graph, pos, show_labels, nodes_size))

        ratio_similar, unsatisfied_nodes = get_unsatisfied_nodes(graph)
        st.write(f"**Number of unsatisfied nodes**: {len(unsatisfied_nodes)} out of {n_nodes} ({len(unsatisfied_nodes)*100/(n_nodes):.2f}%)")
        st.write(f"""**Average similarity ratio**: {np.nanmean(list(ratio_similar.values())):.2f}""")