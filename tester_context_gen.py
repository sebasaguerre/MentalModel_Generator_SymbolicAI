import os
import tempfile
import textwrap
import random
from graphviz import Digraph
from tester_model import TesterModel
from context_gen import Extractor, Generator


def visualize(model, title=None):
    """Render a labelled (multi-edge) Kripke model to an SVG and open it."""

    # constant-size nodes for visual consistency
    dot = Digraph(node_attr={
        'fontname': 'Helvetica,Arial,sans-serif',
        'fontsize': '10',
        'style': 'filled',
        'fillcolor': '#fcfcfc',
        'fixedsize': 'shape',
        'width': '1.2',
        'height': '1.2',
    })

    if title:
        dot.attr(label=title, labelloc="t", fontsize="14", fontname="Helvetica-Bold")

    # compact layered layout; merge parallel edges; cap size to fit whole model
    dot.attr(rankdir="LR", nodesep="0.3", ranksep="0.3")
    # dot.attr(concentrate="true")
    dot.attr(size="13,10")  # no '!' -> only scales down to fit, never pads

    # nodes: circles, terminal (dead-end) states as red boxes
    for s, props in model.labels.items():
        node_id = str(s)
        prop_str = f"{{{textwrap.fill(', '.join(props), width=10)}}}"
        label = f"{node_id}\n{prop_str}"

        is_terminal = s not in model.relations or not model.relations[s]
        if is_terminal:
            dot.node(node_id, label=label, shape='box', color='#d9534f', penwidth='2')
        else:
            dot.node(node_id, label=label, shape='circle', color='#4a4a4a')

    # edges (labelled by action for multi-edge models)
    if not model.multi_edges:
        dot.edges(
            (str(s), str(next_s))
            for s, next_states in model.relations.items()
            for next_s in next_states
        )
    else:
        for s, actions in model.relations.items():
            for action, next_states in actions.items():
                for next_s in next_states:
                    dot.edge(str(s), str(next_s), label=str(action))

    dot.edge_attr.update(color="#4a4a4a", arrowhead="vee", arrowsize="0.8")

    # render to a temp file, open it, then clean up the source
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gv") as temp_gv:
        temp_base = temp_gv.name
    dot.render(temp_base, view=True, format="svg")
    try:
        os.remove(temp_base)
    except OSError:
        pass


def main():
    model = TesterModel()
    extractor = Extractor(model)

    # visualize model 
    # visualize(TesterModel(), title="Tester Kripke Model")

    # choose random state 
    # state = random.choice(tuple(model.states))
    # print(f"State: {state}")

    # DAG of depth k
    k = 2
    prop_tree = extractor.extract_labels("s7", k)
    extractor.print_label_tree(prop_tree)

    # generate formula bottom-up and print
    generator = Generator(button_up=True)
    formula = generator.generate_formula(prop_tree, k)
    print("\nGenerated formula:")
    print(repr(formula))
    print(type(formula))
    action_formulas = generator.action_formulas(formula)
    print("\nAction formulas:")
    print
    for a_formula in action_formulas:
        print(f"{a_formula.action}: {repr(a_formula)}")

if __name__ == "__main__":

    main()
