# Connects all agents in order using LangGraph.

from langgraph.graph import StateGraph, END
from graph.state import AgentState

from graph.nodes.parser_node import parser_node
from graph.nodes.router_node import router_node
from graph.nodes.solver_node import solver_node
from graph.nodes.verifier_node import verifier_node
from graph.nodes.explainer_node import explainer_node

def build_graph():
    graph = StateGraph(AgentState)

    # Add all agents as nodes
    graph.add_node("parser", parser_node)
    graph.add_node("router", router_node)
    graph.add_node("solver", solver_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("explainer", explainer_node)

    # Connect them in order
    graph.set_entry_point("parser")
    graph.add_edge("parser", "router")
    graph.add_edge("router", "solver")
    graph.add_edge("solver", "verifier")
    graph.add_edge("verifier", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile()