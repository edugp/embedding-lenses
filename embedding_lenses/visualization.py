from typing import Dict

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Cividis256, Palette
from bokeh.plotting import Figure, figure
from bokeh.transform import factor_cmap


def draw_interactive_scatter_plot(
    hover_fields: Dict[str, np.ndarray], xs: np.ndarray, ys: np.ndarray, values: np.ndarray, palette: Palette = Cividis256
) -> Figure:
    # Normalize values to range between 0-255, to assign a color for each value
    max_value = values.max()
    min_value = values.min()
    if max_value - min_value == 0:
        values_color = np.ones(len(values)).astype(int).astype(str)
    else:
        values_color = ((values - min_value) / (max_value - min_value) * 255).round()
    values_color_set = np.sort(values_color).astype(int).astype(str)

    values_list = values.astype(str).tolist()
    values_set = np.sort(values).astype(str).tolist()

    data = {"_x_": xs, "_y_": ys, "_label_color_": values_list}
    hover_data = {field: values.astype(str).tolist() for field, values in hover_fields.items()}
    data.update(hover_data)
    source = ColumnDataSource(data=data)
    hover = HoverTool(tooltips=[(field, "@%s{safe}" % field) for field in hover_data.keys()])
    p = figure(plot_width=800, plot_height=800, tools=[hover])
    p.circle(
        "_x_",
        "_y_",
        size=10,
        source=source,
        fill_color=factor_cmap("_label_color_", palette=[palette[int(id_)] for id_ in values_color_set], factors=values_set),
    )

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.toolbar.logo = None
    return p
