// Vite requires explicit default export handling for react-plotly.js (CJS module)
import { forwardRef } from "react";
import _Plot from "react-plotly.js";
import type { PlotParams } from "react-plotly.js";

type PlotModule = typeof _Plot & { default?: typeof _Plot };
const PlotBase = (_Plot as PlotModule).default ?? _Plot;

const Plot = forwardRef<InstanceType<typeof _Plot>, PlotParams>((props, ref) => (
  <PlotBase
    {...props}
    ref={ref}
    config={{ ...props.config, setBackground: "opaque" }}
  />
));

Plot.displayName = "Plot";

export default Plot;
