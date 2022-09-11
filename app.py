# app link: https://rklymentiev-phd-app-app-8qs9b9.streamlitapp.com/
# source of motivation: https://www.geeksforgeeks.org/spatial-segregation-in-social-networks/
# to run the script locally run the command `streamlit run app.py`
# documentation: https://docs.streamlit.io/knowledge-base/using-streamlit/how-do-i-run-my-streamlit-script
# small issue with the app: whenever "Run the simulation" button is pressed, streamlit reruns *everything*,
# so the initial graph is also updating

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def get_unsatisfied_nodes(graph, thresh):
    """
    Returns list of unsitisfied nodes of a given graph.

    Parameters:
    -----------
    graph:
        Graph object
    thresh: float
        threshold value in a [0,1] range

    Returns:
    --------
    ratio_similar: dict
        Average similarity rate for each node
    unsatisfied_nodes: list
        List of unsitisfied nodes
    """

    ratio_similar = {}
    for (n, d) in graph.nodes(data=True):
        if d['type'] == 0:  # skip if an empty node
            continue
        ratio_similar[n] = []
        # iterate over node's neighbors and check if type is the same
        neighbors = list(graph.neighbors(n))
        for neigh in neighbors:
            ratio_similar[n].append(graph.nodes[n]["type"] == graph.nodes[neigh]["type"])

        # calculate average similarity rate
        ratio_similar[n] = np.mean(ratio_similar[n])

    # check which nodes have the similarity rate less than a threshold value
    unsatisfied_nodes = []
    for (k, v) in ratio_similar.items():
        if v < thresh:
            unsatisfied_nodes.append(k)

    return ratio_similar, unsatisfied_nodes


def label_unsatisfied(graph, unsatisfied_nodes):
    """
    Creates a dictionary with the labels for unsatisfied nodes.

    Parameters:
    -----------
    graph:
        Graph object
    unsatisfied_nodes: list
        List of unsatisfied nodes

    Returns:
    --------
    labels: dict
        'x' if unsatisfied, empty string if not
    """

    labels = {}
    for n in graph.nodes():
        if n in unsatisfied_nodes:
            labels[n] = 'x'
        else:
            labels[n] = ''

    return labels


def plot_graph(g, p, n_classes, show_labels, nodes_size, thresh):
    """
    Plot the graph.

    Parameters:
    -----------
    graph:
        Graph object
    p: dict
        Dictionary with nodes' positions
    n_classes: int
        Number of classes in the system
    show_labels: bool
        List of unsatisfied nodes.
    nodes_size: int
        Size of nodes for the plot
    """

    # get node IDs into separate lists by type
    empty_nodes = [n for (n,d) in g.nodes(data=True) if d['type'] == 0]
    type1_nodes = [n for (n,d) in g.nodes(data=True) if d['type'] == 1]
    type2_nodes = [n for (n,d) in g.nodes(data=True) if d['type'] == 2]
    if n_classes >= 3:
        type3_nodes = [n for (n, d) in g.nodes(data=True) if d['type'] == 3]
    if n_classes >= 4:
        type4_nodes = [n for (n, d) in g.nodes(data=True) if d['type'] == 4]

    # draw the nodes
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(empty_nodes) != 0:
        nx.draw_networkx_nodes(g, p, node_color='white', nodelist=empty_nodes, node_size=nodes_size)
    nx.draw_networkx_nodes(g, p, node_color='red', nodelist=type2_nodes, node_size=nodes_size, edgecolors='black')
    nx.draw_networkx_nodes(g, p, node_color='green', nodelist=type1_nodes, node_size=nodes_size, edgecolors='black')
    if n_classes >= 3:
        nx.draw_networkx_nodes(g, p, node_color='orange', nodelist=type3_nodes, node_size=nodes_size, edgecolors='black')
    if n_classes >= 4:
        nx.draw_networkx_nodes(g, p, node_color='lightblue', nodelist=type4_nodes, node_size=nodes_size, edgecolors='black')

    _, unsatisfied_nodes = get_unsatisfied_nodes(graph, thresh)

    # draw the edges
    nx.draw_networkx_edges(g, p, width=0.5, alpha=0.7)
    # draw the labels
    if show_labels:
        labels = label_unsatisfied(graph, unsatisfied_nodes)
        nx.draw_networkx_labels(g, p, labels=labels, font_color='black')

    return fig


st.set_page_config(
    page_title='PhD Computational Task', layout="centered")
st.title('PhD Computational Task')
st.text('Author: Ruslan Klymentiev\nDate: 15.09.2022')

algorithm = st.sidebar.selectbox(
    label='Graph type:',
    options=('Grid World', 'Erdos-Renyi Graph'),
    index=0)  # default to grid world

if algorithm == 'Grid World':
    N = st.sidebar.slider(
        label='Grid size (NxN):',
        value=20, min_value=10,
        max_value=50, step=1)

    perc_empty = st.sidebar.slider(
        label='Ratio of empty nodes:',
        value=0.1, min_value=.1,
        max_value=.9, step=.1)

    # calulate the node size based on the number of nodes. sizes are in a [50, 200] range
    nodes_size = ((N - 10) / (50-10)) * (50-200) + 200

    n_classes = int(st.sidebar.number_input(
        label='Number of classes:',
        value=2, min_value=2,
        max_value=4, step=1))

    graph = nx.grid_2d_graph(N, N)  # create the graph
    pos = dict((i, i) for i in graph.nodes())  # position of each node (the same as ID)

    # randomly assign the type to each node
    # 0 - empty node
    for n in graph.nodes():
        graph.nodes[n]['type'] = np.random.choice(
            range(n_classes+1),
            p=[perc_empty]+[(1-perc_empty)/n_classes]*n_classes)

    # list of empty nodes
    empty_cell_list = [n for (n, d) in graph.nodes(data=True) if d['type'] == 0]
    n_nodes = N * N - len(empty_cell_list)  # number of non-empty nodes in the system

    # create diagonal edges
    for (u, v) in graph.nodes():
        if (u + 1 <= N - 1) and (v + 1 <= N - 1):
            graph.add_edge((u, v), (u + 1, v + 1))

    for (u, v) in graph.nodes():
        if (u + 1 <= N - 1) and (v - 1 >= 0):
            graph.add_edge((u, v), (u + 1, v - 1))

    # dictionary with all neighbors (incl. empty nodes)
    all_neighbors = {}
    for n in graph.nodes():
        all_neighbors[n] = list(graph.neighbors(n))

    # remove edges with the empty nodes
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
        value=0.1, min_value=0.1,
        max_value=.9, step=0.1)

    n_classes = int(st.sidebar.number_input(
        label='Number of classes:',
        value=2, min_value=2,
        max_value=4, step=1))

    graph = nx.erdos_renyi_graph(n=n_nodes, p=prob)  # create the graph
    pos = nx.spring_layout(graph, seed=1)  # constant position of the nodes

    # randombly assign types
    for n in graph.nodes():
        graph.nodes[n]['type'] = np.random.choice(range(1,n_classes+1))

    nodes_size = 300


threshold = st.sidebar.slider(
    label='Threshold:',
    value=0.3, min_value=0.,
    max_value=1., step=0.1)

n_steps = st.sidebar.slider(
    label='Number of simulation steps:',
    value=5, min_value=5,
    max_value=20, step=1)

show_labels = st.sidebar.checkbox('Highlight unsatisfied nodes?', value=True)
st.sidebar.write('')  # just so it looks better

tab1, tab2 = st.tabs(["Initial Graph", "Simulations"])

with tab1:
    st.pyplot(plot_graph(graph, pos, n_classes, show_labels, nodes_size, threshold))
    ratio_similar, unsatisfied_nodes = get_unsatisfied_nodes(graph, threshold)
    st.write(f"""**Number of unsatisfied nodes**: {len(unsatisfied_nodes)} out of {n_nodes}
    ({len(unsatisfied_nodes) * 100 / (n_nodes):.2f}%)""")
    st.write(f"""**Average similarity ratio**: {np.nanmean(list(ratio_similar.values())):.2f}""")

with tab2:
    by_step = st.radio(
        "Visualization of results",
        ('Step by step', 'Just the final result'),
        help="Warning: step by step visualization can take a long time to run.")

    run = st.button('Run')

    f = st.empty()
    if run:
        for _ in range(n_steps):
            _, unsatisfied_nodes = get_unsatisfied_nodes(graph, threshold)
            # exit the loop of there are no more unsatisfied nodes
            if len(unsatisfied_nodes) == 0:
                break

            # iterate over all unsatisfied nodes
            for n in unsatisfied_nodes:
                if algorithm == 'Grid World':
                    # randomly select a new position for a nore
                    new_pos = empty_cell_list[np.random.choice(len(empty_cell_list))]
                    # remove the new position from the list and add the old one
                    empty_cell_list.remove(new_pos)
                    empty_cell_list.append(n)
                    # remove edges with the old neighbors
                    neighbors = list(graph.neighbors(n))
                    for neigh in neighbors:
                        graph.remove_edge(n, neigh)

                    # swap the types of nodes
                    graph.nodes(data=True)[new_pos]['type'] = graph.nodes(data=True)[n]['type']
                    graph.nodes(data=True)[n]['type'] = 0

                    # create edges with the new neighbors
                    for neigh in all_neighbors[new_pos]:
                        if graph.nodes(data=True)[neigh]['type'] != 0:
                            graph.add_edge(new_pos, neigh)

                elif algorithm == 'Erdos-Renyi Graph':
                    neighbors = list(graph.neighbors(n))
                    # remove edges with the old neighbors
                    for neigh in neighbors:
                        graph.remove_edge(n, neigh)

                    all_nodes = list(graph.nodes())
                    all_nodes.remove(n)
                    # randomly create edges with the new neighbors
                    for n2 in all_nodes:
                        if np.random.random() < prob:
                            graph.add_edge(n, n2)

            if by_step == 'Step by step':
                f.pyplot(plot_graph(graph, pos, n_classes, show_labels, nodes_size, threshold))

        if by_step == 'Just the final result':
            f.pyplot(plot_graph(graph, pos, n_classes, show_labels, nodes_size, threshold))

        ratio_similar, unsatisfied_nodes = get_unsatisfied_nodes(graph, threshold)
        st.write(f"**Number of unsatisfied nodes**: {len(unsatisfied_nodes)} out of {n_nodes} ({len(unsatisfied_nodes)*100/(n_nodes):.2f}%)")
        st.write(f"""**Average similarity ratio**: {np.nanmean(list(ratio_similar.values())):.2f}""")
