from __future__ import annotations

import os
from typing import Optional

from graphviz import Digraph
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import patheffects as pe


def _fallback_matplotlib(output_path: str) -> str:
    # Minimal, detailed flowchart with straight arrows and rectangular boxes
    bg = "#FFFFFF"
    stroke = "#111111"
    fill = "#F9FAFB"
    text_c = "#111111"

    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.set_axis_off()

    # Node positions with standard flowchart spacing (normalized figure coords)
    pos = {
        "Start": (0.08, 0.5),
        "UserInput": (0.22, 0.5),
        "Agent": (0.36, 0.5),
        "Memory": (0.50, 0.5),
        "Gate": (0.64, 0.5),
        "Judge": (0.78, 0.5),
        "End": (0.92, 0.5),
    }

    def draw_box(center, label, width=0.10, height=0.2, shape="rect"):
        x, y = center
        if shape == "diamond":
            # Diamond shape for decision nodes
            diamond_points = [
                [x, y + height/2], [x + width/2, y], 
                [x, y - height/2], [x - width/2, y]
            ]
            diamond = plt.Polygon(diamond_points, linewidth=2.5, edgecolor=stroke, 
                                facecolor=fill, zorder=2)
            ax.add_patch(diamond)
        elif shape == "oval":
            # Oval shape for start/end nodes
            from matplotlib.patches import Ellipse
            oval = Ellipse((x, y), width, height, linewidth=2.5, edgecolor=stroke, 
                          facecolor=fill, zorder=2)
            ax.add_patch(oval)
        else:
            # Rectangle for process nodes
            rect = Rectangle((x - width/2, y - height/2), width, height,
                           linewidth=2.5, edgecolor=stroke, facecolor=fill, zorder=2)
            ax.add_patch(rect)
        
        txt = ax.text(x, y, label, color=text_c, ha="center", va="center", 
                     fontsize=11, weight='bold', wrap=True)
        txt.set_path_effects([pe.withStroke(linewidth=1, foreground="#FFFFFF")])

    def draw_arrow(p1, p2, label=None, offset=0.05):
        (x1, y1), (x2, y2) = p1, p2
        # Draw arrow with standard flowchart styling
        arr = FancyArrowPatch((x1 + offset, y1), (x2 - offset, y2),
                              arrowstyle="-|>", mutation_scale=25, linewidth=3.0,
                              color=stroke, zorder=3)
        ax.add_patch(arr)
        if label:
            lx = (x1 + x2) / 2
            ly = y1 + 0.08
            ax.text(lx, ly, label, color=text_c, ha="center", va="center", fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor=stroke, linewidth=2))

    # Draw nodes using standard flowchart shapes
    draw_box(pos["Start"], "START", shape="oval")
    draw_box(pos["UserInput"], "USER\nINPUT", shape="rect")
    draw_box(pos["Agent"], "AGENT\nTURN", shape="rect")
    draw_box(pos["Memory"], "MEMORY\nUPDATE", shape="rect")
    draw_box(pos["Gate"], "ROUND\n>= 8?", shape="diamond", width=0.12, height=0.25)
    draw_box(pos["Judge"], "JUDGE", shape="rect")
    draw_box(pos["End"], "END", shape="oval")

    # Draw arrows with standard flowchart styling
    draw_arrow(pos["Start"], pos["UserInput"])
    draw_arrow(pos["UserInput"], pos["Agent"])
    draw_arrow(pos["Agent"], pos["Memory"])
    draw_arrow(pos["Memory"], pos["Gate"])
    
    # Standard flowchart loop back arrow
    gate_x, gate_y = pos["Gate"]
    agent_x, agent_y = pos["Agent"]
    loop_arrow = FancyArrowPatch((gate_x - 0.06, gate_y - 0.12), (agent_x + 0.06, agent_y - 0.1),
                                 arrowstyle="-|>", mutation_scale=25, linewidth=3.0,
                                 color=stroke, zorder=3,
                                 connectionstyle="arc3,rad=-0.5")
    ax.add_patch(loop_arrow)
    ax.text((gate_x + agent_x) / 2, gate_y - 0.2, "NO", color=text_c, ha="center", va="center", 
           fontsize=10, weight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor=stroke, linewidth=2))
    
    draw_arrow(pos["Gate"], pos["Judge"], label="YES")
    draw_arrow(pos["Judge"], pos["End"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_png = output_path if output_path.lower().endswith(".png") else output_path + ".png"
    plt.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor(), 
                pad_inches=0.5, dpi=300, format='png')
    plt.close(fig)
    return out_png


def save_dag_png(output_path: str) -> Optional[str]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot = Digraph(comment="Debate DAG", format="png")
    # Minimal flowchart aesthetics
    dot.attr(rankdir="LR", bgcolor="#FFFFFF", labelloc="t", labeljust="c",
             label="Multi-Agent Debate Flow", fontname="Helvetica", fontsize="18")
    dot.attr(margin="0.2", pad="0.2", dpi="200", splines="ortho")
    dot.attr("node", fontname="Helvetica", fontsize="12", shape="box",
             style="filled", color="#111111", fillcolor="#F9FAFB", penwidth="1.6")
    dot.attr("edge", fontname="Helvetica", fontsize="10", color="#111111",
             arrowsize="0.8", penwidth="1.6")

    # Nodes (detailed)
    dot.node("Start", "Start")
    dot.node("UserInput", "UserInput (Topic)")
    dot.node("Agent", "Agent Turn (Scientist or Philosopher)")
    dot.node("Memory", "Memory Update (Summary + Opponent Notes)")
    dot.node("Gate", "Round >= 8? (Loop Control)")
    dot.node("Judge", "Judge (Summary + Winner)")
    dot.node("End", "End")

    # Edges (straight/orthogonal)
    dot.edge("Start", "UserInput")
    dot.edge("UserInput", "Agent")
    dot.edge("Agent", "Memory")
    dot.edge("Memory", "Gate")
    dot.edge("Gate", "Agent", label="No")
    dot.edge("Gate", "Judge", label="Yes")
    dot.edge("Judge", "End")

    try:
        out = dot.render(filename=output_path, cleanup=True)
        return out
    except Exception:
        # Fallback: always save a DOT file even if PNG fails
        dot_path = output_path + ".dot"
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot.source)
        # Clean up any extension-less temp file left by graphviz
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(output_path + ".gv"):
                os.remove(output_path + ".gv")
        except Exception:
            pass
        # Additionally, render a PNG using matplotlib so users can view it without Graphviz
        try:
            png_path = _fallback_matplotlib(output_path + "_fallback.png")
            return png_path
        except Exception:
            # If even fallback fails, at least return the DOT path
            return dot_path
