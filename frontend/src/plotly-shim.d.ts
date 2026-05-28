// Ambient module declaration for plotly.js's UMD subpath bundle. The
// package types only describe the root entrypoint, so importing the
// dist build (which we do at runtime to dodge the production
// downloadImage/toImage tree-shaking bug) needs a stub here.
declare module "plotly.js/dist/plotly" {
  // The dist bundle's default export is the full Plotly runtime — same
  // surface as the typed root entrypoint, so we re-use it.
  import Plotly from "plotly.js";
  export default Plotly;
}
