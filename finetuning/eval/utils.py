import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from typing import Dict

def display_projections_3d(
    labels: np.ndarray,
    projections: np.ndarray,
    image_paths: np.ndarray | None,
    image_data_uris: Dict[str, str] | None,
    show_legend: bool = False,
    show_markers_with_text: bool = True,
    title="CLIP Image Embeddings (3D)",
    file_name="embeddings_3d"
) -> None:
    # Create a separate trace for each unique label
    unique_labels = np.unique(labels)
    traces = []
    for unique_label in unique_labels:
        mask = labels == unique_label
        customdata_masked = image_paths[mask] if image_paths else None
        trace = go.Scatter3d(
            x=projections[mask][:, 0],
            y=projections[mask][:, 1],
            z=projections[mask][:, 2],
            mode='markers+text' if show_markers_with_text else 'markers',
            text=labels[mask],
            customdata=customdata_masked,
            name=str(unique_label),
            marker=dict(size=8),
            hovertemplate="<b>class: %{text}</b><br>path: %{customdata}<extra></extra>"
        )
        traces.append(trace)

    # Create the 3D scatter plot
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=1000,
        height=1000,
        showlegend=show_legend,
        title=title
    )
    fig.write_html(f"{file_name}.html", include_plotlyjs="cdn")
    # fig.write_image(f"{file_name}.png", width=1200, height=900)
    fig.show()
    print("Saved 3D visualization!")

def display_projections_2d(
    labels: np.ndarray,
    projections: np.ndarray,
    image_paths: np.ndarray | None = None,
    image_data_uris: dict[str, str] | None = None,   # kept for API parity (unused here)
    show_legend: bool = True,
    show_markers_with_text: bool = False,
    title: str = "CLIP Image Embeddings (2D)",
    file_name: str | None = "embeddings_2d",         # None = don’t save
    marker_size: int = 7,
    marker_opacity: float = 0.9,
):
    projections = np.asarray(projections)
    labels = np.asarray(labels)

    traces = []
    unique_labels = np.unique(labels)
    for u in unique_labels:
        m = (labels == u)
        hover = "<b>class:</b> %{text}<extra></extra>"
        kwargs = dict(
            x=projections[m, 0],
            y=projections[m, 1],
            mode="markers+text" if show_markers_with_text else "markers",
            name=str(u),
            text=labels[m],
            marker=dict(size=marker_size, opacity=marker_opacity),
            hovertemplate=hover,
        )
        if image_paths is not None:
            ip = np.asarray(image_paths)
            kwargs["customdata"] = ip[m]
            kwargs["hovertemplate"] = "<b>class:</b> %{text}<br><b>path:</b> %{customdata}<extra></extra>"

        traces.append(go.Scattergl(**kwargs))

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        title=title,
        width=900,
        height=700,
        showlegend=show_legend,
        template="simple_white",
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.write_html(f"{file_name}.html", include_plotlyjs="cdn")
    # fig.write_image(f"{file_name}.png", width=1200, height=900)
    fig.show()
    print(f"Saved 2D visualization!!")
