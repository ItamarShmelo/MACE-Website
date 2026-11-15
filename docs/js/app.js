import { renderLinePlot, renderHeatmap } from './plot.js';

/* ── State ─────────────────────────────────────────────── */

// One manifest per weight function; `manifest` is the reference copy that
// defines the (shared) grids and temperature list.
const WEIGHTS = ["uniform", "wien", "planck"];
const WEIGHT_LABELS = { uniform: "Uniform", wien: "Wien", planck: "Planck" };
const manifests = {};
let manifest = null;
let pyodide = null;
let collapseCode = null;
let fineBoundaries = null;
let nFineGroups = 0;
let nFineAngleBins = 0;
let jszipLoaded = false;


/* ── DOM refs ──────────────────────────────────────────── */

const $ = (id) => document.getElementById(id);
const loadingMsg = $("loading-message");
const pyodideLoading = $("pyodide-loading");
const downloadForm = $("download-form");
const gridSummary = $("grid-summary");
const tempSelect = $("temperature-select");
const customTempInput = $("custom-temperature-input");
const customTempError = $("custom-temperature-error");
const tempHint = $("temperature-hint");
const nGroupsInput = $("n-groups");
const nGroupsHint = $("n-groups-hint");
const boundariesDisplay = $("boundaries-display");
const boundariesAuto = $("boundaries-auto");
const customBoundariesInput = $("custom-boundaries-input");
const customBoundariesError = $("custom-boundaries-error");
const nAnglesInput = $("n-angles");
const nAnglesHint = $("n-angles-hint");
const angleInfo = $("angle-info");
const customAngleInput = $("custom-angle-input");
const customAngleError = $("custom-angle-error");
const summaryShape = $("summary-shape");
const summaryAxes = $("summary-axes");
const summarySize = $("summary-size");
const btnDownload = $("btn-download");
const btnDownloadAll = $("btn-download-all");
const btnDownloadMetadata = $("btn-download-metadata");
const progressContainer = $("progress-container");
const progressBar = $("progress-bar");
const progressLabel = $("progress-label");

const plotForm = $("plot-form");
const plotPlaceholder = $("plot-placeholder");
const plotModeSelect = $("plot-mode-select");
const plotTempSelect = $("plot-temperature-select");
const plotWeightSelect = $("plot-weight-select");
const plotQuantitySelect = $("plot-quantity-select");
const plotButton = $("plot-button");
const weightHint = $("weight-hint");

const plotSection = $("plot-section");
const plotChart = $("plot-chart");
const plotDescription = $("plot-description");
const plotSelLeft = $("plot-sel-left");
const plotSelRight = $("plot-sel-right");
const plotSelLeftBox = $("plot-sel-left-box");
const plotSelRightBox = $("plot-sel-right-box");
const plotSelLeftLabel = $("plot-sel-left-label");
const plotSelRightLabel = $("plot-sel-right-label");
const plotFormError = $("plot-form-error");

/* ── Mode helpers ──────────────────────────────────────── */

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function isTempCustom() { return getRadioValue("temp-mode") === "custom"; }
function isEnergyCustom() { return getRadioValue("energy-mode") === "custom"; }
function isAngleCustom() { return getRadioValue("angle-mode") === "custom"; }
function currentWeight() { return getRadioValue("weight-fn") || "uniform"; }
function currentQuantity() { return getRadioValue("quantity") || "sigma"; }

/* ── Data access ───────────────────────────────────────── */

function dataUrl(weight, file) { return `data/${weight}/${file}`; }

async function fetchBuffer(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

// Writes the stored table(s) needed for `quantity` at manifest index `idx`
// into the Pyodide FS under `prefix` and returns the Python tuple expression
// (sigma_bytes, dsigma_bytes | None) that reads them back.
async function stageTables(weight, quantity, idx, prefix) {
  const entry = manifests[weight].temperatures[idx];
  const sigmaPath = `/tmp/${prefix}_sigma.npz`;
  pyodide.FS.writeFile(sigmaPath, await fetchBuffer(dataUrl(weight, entry.file)));
  if (quantity === "sigma") return `(open("${sigmaPath}","rb").read(), None)`;
  const dsigmaPath = `/tmp/${prefix}_dsigma.npz`;
  pyodide.FS.writeFile(dsigmaPath, await fetchBuffer(dataUrl(weight, entry.dsigma_dT_file)));
  return `(open("${sigmaPath}","rb").read(), open("${dsigmaPath}","rb").read())`;
}

function updateWeightHint() {
  const w = currentWeight();
  const notes = {
    uniform: "w = 1 across each incoming group.",
    wien: "w = x³ e⁻ˣ with x = E/kT.",
    planck: "w = x³/(eˣ − 1) with x = E/kT (Wien form above x = 25).",
  };
  const q = currentQuantity() === "sigma"
    ? "Cross section σ in cm²."
    : "Full temperature derivative dσ/dT in cm²/K, including the weight-function terms.";
  weightHint.textContent = manifests[w] ? `${notes[w]} ${q}` : `${WEIGHT_LABELS[w]} tables are not available.`;
}

/* ── Helpers ───────────────────────────────────────────── */

function compactExp(v) {
  return v.toExponential(2).replace(/e\+?0*(-?)0*(\d+)/, "e$1$2");
}

function formatTemp(t) {
  return compactExp(t.temperature_K * 8.617333262e-5) + " eV";
}

function pickLogSpacedBoundaries(fineBounds, nCoarse) {
  const G = fineBounds.length - 1;
  if (nCoarse >= G) return Array.from(fineBounds);
  return Array.from({ length: nCoarse + 1 }, (_, i) => {
    const frac = i / nCoarse;
    const logMin = Math.log(fineBounds[0]);
    const logMax = Math.log(fineBounds[G]);
    return Math.exp(logMin + frac * (logMax - logMin));
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* ── Temperature helpers ──────────────────────────────── */

function getTemperatureRange() {
  if (!manifest) return { min: 1000, max: 1e9 };
  const temps = manifest.temperatures;
  return { min: temps[0].temperature_K, max: temps[temps.length - 1].temperature_K };
}

function findBracketingTemps(T_target) {
  const temps = manifest.temperatures;
  if (T_target <= temps[0].temperature_K) return { lo: 0, hi: 0, exact: true };
  if (T_target >= temps[temps.length - 1].temperature_K)
    return { lo: temps.length - 1, hi: temps.length - 1, exact: true };
  for (let i = 0; i < temps.length - 1; i++) {
    const tLo = temps[i].temperature_K;
    const tHi = temps[i + 1].temperature_K;
    if (Math.abs(T_target - tLo) / tLo < 1e-9) return { lo: i, hi: i, exact: true };
    if (Math.abs(T_target - tHi) / tHi < 1e-9) return { lo: i + 1, hi: i + 1, exact: true };
    if (T_target > tLo && T_target < tHi) return { lo: i, hi: i + 1, exact: false };
  }
  return { lo: temps.length - 1, hi: temps.length - 1, exact: true };
}

function getSelectedTemperatures() {
  if (isTempCustom()) return parseCustomTemperatures();
  const selected = Array.from(tempSelect.selectedOptions)
    .map((opt) => parseInt(opt.value))
    .filter((v) => !isNaN(v));
  if (selected.length === 0) return null;
  const temps = selected.map((idx) => {
    const entry = manifest.temperatures.find((t) => t.index === idx);
    return entry ? entry.temperature_K : null;
  }).filter((v) => v !== null);
  return deduplicateAndSort(temps);
}

function deduplicateAndSort(temps) {
  const sorted = [...temps].sort((a, b) => a - b);
  const deduped = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    if (Math.abs(sorted[i] - deduped[deduped.length - 1]) / sorted[i] > 1e-9)
      deduped.push(sorted[i]);
  }
  return deduped;
}

function parseCustomTemperatures() {
  const text = customTempInput.value.trim();
  if (!text) { customTempError.textContent = ""; return null; }
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customTempError.textContent = "Non-numeric value found.";
    customTempError.classList.add("error");
    return null;
  }
  if (values.length < 1) {
    customTempError.textContent = "Enter at least one temperature.";
    customTempError.classList.add("error");
    return null;
  }
  if (values.some((v) => v <= 0)) {
    customTempError.textContent = "Temperatures must be positive.";
    customTempError.classList.add("error");
    return null;
  }
  const range = getTemperatureRange();
  if (values.some((v) => v < range.min * (1 - 1e-6) || v > range.max * (1 + 1e-6))) {
    customTempError.textContent =
      `Temperatures must lie within [${range.min.toExponential(3)}, ${range.max.toExponential(3)}] K.`;
    customTempError.classList.add("error");
    return null;
  }
  const sorted = deduplicateAndSort(values);
  customTempError.textContent = `\u2713 ${sorted.length} temperature(s) specified.`;
  customTempError.classList.remove("error");
  return sorted;
}

/* ── UI update functions ───────────────────────────────── */

function getCurrentEnergyBoundaries() {
  if (isEnergyCustom()) return parseCustomBoundaries();
  const n = Math.max(1, Math.min(nFineGroups, parseInt(nGroupsInput.value) || nFineGroups));
  return pickLogSpacedBoundaries(fineBoundaries, n);
}

function getCurrentAngleBins() {
  if (isAngleCustom()) {
    const bounds = parseCustomAngleBoundaries();
    if (bounds) return { boundaries: bounds };
    return null;
  }
  const v = parseInt(nAnglesInput.value);
  if (isNaN(v) || v < 1) return { count: 1 };
  return { count: v };
}

function getAngleBinCount() {
  const a = getCurrentAngleBins();
  if (!a) return 0;
  return a.count || (a.boundaries.length - 1);
}

function getAngleBoundariesArray() {
  const a = getCurrentAngleBins();
  if (!a) return null;
  if (a.boundaries) return a.boundaries;
  const nA = a.count;
  const bounds = [];
  for (let i = 0; i <= nA; i++) bounds.push(-1 + (2 * i) / nA);
  return bounds;
}

function updateBoundariesDisplay() {
  if (!fineBoundaries || isEnergyCustom()) return;
  const bounds = getCurrentEnergyBoundaries();
  if (!bounds) return;
  boundariesDisplay.innerHTML = bounds.map((v) => compactExp(v * 1e3) + " eV").join("<br>");
}

function updateSummary() {
  const bounds = getCurrentEnergyBoundaries();
  if (!bounds) return;
  const angleCfg = getCurrentAngleBins();
  if (!angleCfg) return;
  const nG = bounds.length - 1;
  const nA = getAngleBinCount();
  const temps = getSelectedTemperatures();
  const nT = temps ? temps.length : 0;

  if (nT > 1) {
    summaryShape.innerHTML = `${nT} &times; ${nG} &times; ${nG} &times; ${nA}`;
    summaryAxes.textContent = "[temperature] \u00D7 [incoming group] \u00D7 [outgoing group] \u00D7 [angle bin]";
    summarySize.textContent = formatSize(nT * nG * nG * nA * 8);
  } else if (nG === 1 && nA === 1) {
    summaryShape.textContent = "scalar";
    summaryAxes.textContent = "\u2014";
    summarySize.textContent = "8 B";
  } else if (nG === 1) {
    summaryShape.textContent = `${nA}`;
    summaryAxes.textContent = "[angle bin]";
    summarySize.textContent = formatSize(nA * 8);
  } else if (nA === 1) {
    summaryShape.innerHTML = `${nG} &times; ${nG}`;
    summaryAxes.textContent = "[incoming group] \u00D7 [outgoing group]";
    summarySize.textContent = formatSize(nG * nG * 8);
  } else {
    summaryShape.innerHTML = `${nG} &times; ${nG} &times; ${nA}`;
    summaryAxes.textContent = "[incoming group] \u00D7 [outgoing group] \u00D7 [angle bin]";
    summarySize.textContent = formatSize(nG * nG * nA * 8);
  }
}

function updateAngleHints() {
  if (isAngleCustom()) return;
  const aInfo = getCurrentAngleBins();
  if (!aInfo) return;
  const nA = aInfo.count || 1;
  nAnglesHint.textContent = `Any integer from 1 to ${nFineAngleBins}. Values above ${nFineAngleBins} exceed stored resolution.`;
  angleInfo.textContent = `\u0394\u03BE = ${compactExp(2 / nA)}`;
}

function parseCustomBoundaries() {
  const text = customBoundariesInput.value.trim();
  if (!text) { customBoundariesError.textContent = ""; return null; }
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customBoundariesError.textContent = "Non-numeric value found.";
    customBoundariesError.classList.add("error");
    return null;
  }
  if (values.length < 2) {
    customBoundariesError.textContent = "Need at least 2 boundary values.";
    customBoundariesError.classList.add("error");
    return null;
  }
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      customBoundariesError.textContent = "Boundaries must be strictly increasing.";
      customBoundariesError.classList.add("error");
      return null;
    }
  }
  if (values[0] <= 0) {
    customBoundariesError.textContent = "Boundaries must be positive.";
    customBoundariesError.classList.add("error");
    return null;
  }
  const eMin = fineBoundaries[0];
  const eMax = fineBoundaries[fineBoundaries.length - 1];
  if (values[0] < eMin * (1 - 1e-6) || values[values.length - 1] > eMax * (1 + 1e-6)) {
    customBoundariesError.textContent = `Boundaries must lie within [${compactExp(eMin * 1e3)}, ${compactExp(eMax * 1e3)}] eV.`;
    customBoundariesError.classList.add("error");
    return null;
  }
  customBoundariesError.textContent = `\u2713 ${values.length - 1} groups defined.`;
  customBoundariesError.classList.remove("error");
  return values;
}

function parseCustomAngleBoundaries() {
  const text = customAngleInput.value.trim();
  if (!text) { customAngleError.textContent = ""; return null; }
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customAngleError.textContent = "Non-numeric value found.";
    customAngleError.classList.add("error");
    return null;
  }
  if (values.length < 2) {
    customAngleError.textContent = "Need at least 2 boundary values.";
    customAngleError.classList.add("error");
    return null;
  }
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      customAngleError.textContent = "Boundaries must be strictly increasing.";
      customAngleError.classList.add("error");
      return null;
    }
  }
  if (values[0] < -1 - 1e-9 || values[values.length - 1] > 1 + 1e-9) {
    customAngleError.textContent = "Boundaries must lie within [\u22121, 1].";
    customAngleError.classList.add("error");
    return null;
  }
  customAngleError.textContent = `\u2713 ${values.length - 1} angle bins defined.`;
  customAngleError.classList.remove("error");
  return values;
}

/* ── Mode panel switching ──────────────────────────────── */

function setupModeRadios(name, panels) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((radio) => {
    radio.addEventListener("change", () => {
      const val = radio.value;
      for (const [key, panelId] of Object.entries(panels)) {
        const panel = $(panelId);
        if (panel) panel.hidden = key !== val;
      }
      updateBoundariesDisplay();
      updateSummary();
      updateButtonStates();
    });
  });
}

/* ── Event wiring ──────────────────────────────────────── */

function wireDownloadEvents() {
  document.querySelectorAll('input[name="weight-fn"], input[name="quantity"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      updateWeightHint();
      updateButtonStates();
    });
  });
  setupModeRadios("temp-mode", { stored: "temp-stored-panel", custom: "temp-custom-panel" });
  setupModeRadios("energy-mode", { standard: "energy-standard-panel", custom: "energy-custom-panel" });
  setupModeRadios("angle-mode", { standard: "angle-standard-panel", custom: "angle-custom-panel" });

  nGroupsInput.addEventListener("input", () => {
    const v = parseInt(nGroupsInput.value);
    if (v < 1 || v > nFineGroups) {
      nGroupsHint.textContent = `Must be between 1 and ${nFineGroups}.`;
      nGroupsHint.classList.add("error");
    } else {
      nGroupsHint.textContent = `1 \u2013 ${nFineGroups}`;
      nGroupsHint.classList.remove("error");
    }
    updateBoundariesDisplay();
    updateSummary();
  });

  customBoundariesInput.addEventListener("input", () => {
    parseCustomBoundaries();
    updateSummary();
  });

  nAnglesInput.addEventListener("input", () => {
    updateAngleHints();
    updateSummary();
  });

  customAngleInput.addEventListener("input", () => {
    parseCustomAngleBoundaries();
    updateSummary();
  });

  customTempInput.addEventListener("input", () => {
    parseCustomTemperatures();
    updateSummary();
    updateButtonStates();
  });

  tempSelect.addEventListener("change", () => {
    updateSummary();
    updateButtonStates();
  });

  btnDownload.addEventListener("click", handleDownload);
  btnDownloadAll.addEventListener("click", handleDownloadAll);
  btnDownloadMetadata.addEventListener("click", handleDownloadMetadata);
}

function updateButtonStates() {
  const hasTemp = isTempCustom()
    ? parseCustomTemperatures() !== null
    : tempSelect.selectedOptions.length > 0;
  const hasWeight = !!manifests[currentWeight()];
  btnDownload.disabled = !hasTemp || !pyodide || !hasWeight;
  btnDownloadAll.disabled = !jszipLoaded || !pyodide || !manifest || !hasWeight;
  btnDownloadMetadata.disabled = !hasTemp || !pyodide;
  plotButton.disabled = !pyodide || !manifest;
}

/* ── Progress helpers ──────────────────────────────────── */

function showProgress(pct, label) {
  progressContainer.hidden = false;
  progressBar.style.width = pct + "%";
  progressLabel.textContent = label;
}

function hideProgress() {
  setTimeout(() => {
    progressContainer.hidden = true;
    progressBar.style.width = "0%";
  }, 2000);
}

/* ── Collapse call builders ────────────────────────────── */

function isIdentityGrid(energyBounds, angleInfo) {
  if (angleInfo.boundaries) return false;
  if (angleInfo.count !== nFineAngleBins) return false;
  if (energyBounds.length !== fineBoundaries.length) return false;
  for (let i = 0; i < energyBounds.length; i++) {
    if (Math.abs(energyBounds[i] - fineBoundaries[i]) > fineBoundaries[i] * 1e-9) return false;
  }
  return true;
}

/* ── Single/multi-temperature download ─────────────────── */

async function handleDownload() {
  const temps = getSelectedTemperatures();
  if (!temps || temps.length === 0 || !pyodide) return;

  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleInfo = getCurrentAngleBins();
  if (!angleInfo) return;
  const nCoarseGroups = energyBounds.length - 1;
  const nA = getAngleBinCount();
  const nT = temps.length;
  const identity = isIdentityGrid(energyBounds, angleInfo);

  btnDownload.disabled = true;
  showProgress(5, "Processing\u2026");

  try {
    const anglePyArg = angleInfo.boundaries
      ? `angle_boundaries=[${angleInfo.boundaries.join(",")}]`
      : `n_angle_bins=${angleInfo.count}`;
    const energyPyList = `[${energyBounds.join(",")}]`;
    const weight = currentWeight();
    const quantity = currentQuantity();
    const temperatures = manifests[weight].temperatures;

    await pyodide.runPythonAsync("_dl_results = []");

    for (let i = 0; i < nT; i++) {
      showProgress(5 + Math.round((i / nT) * 80), `Processing temperature ${i + 1}/${nT}\u2026`);
      const T = temps[i];
      const bracket = findBracketingTemps(T);
      const entryLo = temperatures[bracket.lo];
      const entryHi = temperatures[bracket.hi];

      const loFiles = await stageTables(weight, quantity, bracket.lo, "lo");
      const hiFiles = bracket.exact ? "None" : await stageTables(weight, quantity, bracket.hi, "hi");
      await pyodide.runPythonAsync(
        `_dl_results.append(matrix_at_temperature(\n    "${quantity}", ${loFiles}, ${hiFiles},\n` +
        `    ${entryLo.temperature_K}, ${entryHi.temperature_K}, ${T},\n    ${energyPyList},\n` +
        `    ${anglePyArg},\n    identity=${identity ? "True" : "False"},\n))`
      );
    }

    showProgress(88, "Building output file\u2026");

    const npyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
if len(_dl_results) == 1:
    _arr = _dl_results[0]
    N, _, K = _arr.shape
    if N == 1 and K == 1:
        _arr = _arr.ravel()[0]
    elif N == 1:
        _arr = _arr[0, 0, :]
    elif K == 1:
        _arr = _arr[:, :, 0]
else:
    _arr = np.stack(_dl_results, axis=0)
_buf = io.BytesIO()
np.save(_buf, _arr)
del _dl_results
_buf.getvalue()
`);

    showProgress(95, "Preparing download\u2026");
    const blob = new Blob([npyBytes.toJs()], { type: "application/octet-stream" });
    triggerDownload(blob, buildFilename(weight, quantity, temps, nCoarseGroups, nA));
    showProgress(100, "Download complete.");
    hideProgress();
  } catch (err) {
    progressLabel.textContent = "Error: " + err.message;
    progressBar.style.width = "0%";
    console.error(err);
  } finally {
    updateButtonStates();
  }
}

function buildFilename(weight, quantity, temps, nG, nA) {
  const stem = `compton_${quantity}_${weight}`;
  const nT = temps.length;
  if (nT === 1) {
    const bracket = findBracketingTemps(temps[0]);
    if (bracket.exact) return `${stem}_T${String(bracket.lo).padStart(3, "0")}_${nG}g_${nA}a.npy`;
    return `${stem}_T${temps[0].toExponential(3)}K_${nG}g_${nA}a.npy`;
  }
  return `${stem}_${nT}T_${nG}g_${nA}a.npy`;
}

/* ── Bulk download ─────────────────────────────────────── */

async function handleDownloadAll() {
  if (!manifest || !jszipLoaded || !pyodide) return;

  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleInfo = getCurrentAngleBins();
  if (!angleInfo) return;
  const nCoarseGroups = energyBounds.length - 1;
  const nA = getAngleBinCount();
  const identity = isIdentityGrid(energyBounds, angleInfo);
  const weight = currentWeight();
  const quantity = currentQuantity();
  const temperatures = manifests[weight].temperatures;
  const total = temperatures.length;
  const totalEstimate = nCoarseGroups * nCoarseGroups * nA * 8 * total;

  if (totalEstimate > 200 * 1024 * 1024) {
    const sizeMB = (totalEstimate / (1024 * 1024)).toFixed(0);
    if (!confirm(`Estimated total size: ~${sizeMB} MB.\nThis may use significant browser memory. Proceed?`)) return;
  }

  btnDownloadAll.disabled = true;

  if (identity) {
    showProgress(0, `Fetching 0/${total}\u2026`);
    try {
      // Raw stored tables; dsigma/dT files hold T_dsigma_dT = T * dsigma/dT
      // (see the bundled manifest.json).
      const zip = new JSZip();
      zip.file("manifest.json", JSON.stringify(manifests[weight], null, 2));
      for (let i = 0; i < total; i++) {
        const t = temperatures[i];
        const file = quantity === "sigma" ? t.file : t.dsigma_dT_file;
        zip.file(file, await fetchBuffer(dataUrl(weight, file)));
        showProgress(Math.round(((i + 1) / total) * 90), `Fetching ${i + 1}/${total}\u2026`);
      }
      showProgress(92, "Compressing ZIP\u2026");
      const blob = await zip.generateAsync({ type: "blob" });
      triggerDownload(blob, `compton_${quantity}_${weight}_all_temperatures_raw.zip`);
      showProgress(100, "Download complete.");
      hideProgress();
    } catch (err) {
      progressLabel.textContent = "Error: " + err.message;
      console.error(err);
    } finally {
      updateButtonStates();
    }
    return;
  }

  showProgress(0, `Processing 0/${total}\u2026`);
  try {
    await pyodide.runPythonAsync("_dl_all_results = []");
    for (let i = 0; i < total; i++) {
      const t = temperatures[i];
      const files = await stageTables(weight, quantity, i, "all");
      await pyodide.runPythonAsync(
        `_dl_all_results.append(matrix_at_temperature(\n    "${quantity}", ${files}, None,\n` +
        `    ${t.temperature_K}, ${t.temperature_K}, ${t.temperature_K},\n    [${energyBounds.join(",")}],\n` +
        `    ${angleInfo.boundaries ? `angle_boundaries=[${angleInfo.boundaries.join(",")}]` : `n_angle_bins=${angleInfo.count}`},\n))`
      );
      showProgress(Math.round(((i + 1) / total) * 85), `Processing ${i + 1}/${total}\u2026`);
    }
    showProgress(87, "Building 4D array\u2026");
    const npyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_stacked = np.stack(_dl_all_results, axis=0)
_buf = io.BytesIO()
np.save(_buf, _stacked)
del _dl_all_results
_buf.getvalue()
`);
    const blob = new Blob([npyBytes.toJs()], { type: "application/octet-stream" });
    triggerDownload(blob, `compton_${quantity}_${weight}_all_${total}T_${nCoarseGroups}g_${nA}a.npy`);
    showProgress(100, "Download complete.");
    hideProgress();
  } catch (err) {
    progressLabel.textContent = "Error: " + err.message;
    console.error(err);
  } finally {
    updateButtonStates();
  }
}

/* ── Metadata download ─────────────────────────────────── */

async function handleDownloadMetadata() {
  const temps = getSelectedTemperatures();
  if (!temps || !pyodide) return;
  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleBounds = getAngleBoundariesArray();
  if (!angleBounds) return;

  btnDownloadMetadata.disabled = true;
  try {
    const tempBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_buf = io.BytesIO()
np.save(_buf, np.array([${temps.join(",")}], dtype=np.float64))
_buf.getvalue()
`);
    triggerDownload(new Blob([tempBytes.toJs()], { type: "application/octet-stream" }), "temperatures.npy");

    const energyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_buf = io.BytesIO()
np.save(_buf, np.array([${energyBounds.join(",")}], dtype=np.float64))
_buf.getvalue()
`);
    triggerDownload(new Blob([energyBytes.toJs()], { type: "application/octet-stream" }), "energy_boundaries_keV.npy");

    const angleBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_buf = io.BytesIO()
np.save(_buf, np.array([${angleBounds.join(",")}], dtype=np.float64))
_buf.getvalue()
`);
    triggerDownload(new Blob([angleBytes.toJs()], { type: "application/octet-stream" }), "angle_boundaries_xi.npy");
  } catch (err) {
    console.error("Metadata download error:", err);
  } finally {
    updateButtonStates();
  }
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ── Plot Instrument ───────────────────────────────────── */

function wirePlotEvents() {
  plotModeSelect.addEventListener("change", updatePlotPanels);
  plotButton.addEventListener("click", handlePlot);

  [$("plot-e-group"), $("plot-xi-bin"), $("plot-e-group-xi"),
   $("plot-ep-group-xi"), $("plot-xi-bin-matrix"), plotTempSelect,
   plotModeSelect].forEach((el) => {
    if (el) el.addEventListener("change", updatePlotSelectionInfo);
  });
  [$("plot-e-group"), $("plot-xi-bin"), $("plot-e-group-xi"),
   $("plot-ep-group-xi"), $("plot-xi-bin-matrix")].forEach((el) => {
    if (el) el.addEventListener("input", updatePlotSelectionInfo);
  });
}

function updatePlotPanels() {
  const mode = plotModeSelect.value;
  $("plot-params-ep").hidden = mode !== "ep-profile";
  $("plot-params-xi").hidden = mode !== "xi-profile";
  $("plot-params-matrix").hidden = mode !== "matrix";
  $("plot-scale-switches").hidden = mode === "matrix";
  $("plot-log-x").parentElement.hidden = mode === "xi-profile";
  updatePlotSelectionInfo();
}

function groupMidpoint(idx) {
  if (!fineBoundaries || idx < 0 || idx >= nFineGroups) return NaN;
  return Math.sqrt(fineBoundaries[idx] * fineBoundaries[idx + 1]);
}

function groupBoundsEV(idx) {
  if (!fineBoundaries || idx < 0 || idx >= nFineGroups) return "?";
  return `[${compactExp(fineBoundaries[idx] * 1e3)}, ${compactExp(fineBoundaries[idx + 1] * 1e3)}] eV`;
}

function angleBinMidpoint(idx) {
  if (idx < 0 || idx >= nFineAngleBins) return NaN;
  const w = 2 / nFineAngleBins;
  return -1 + (idx + 0.5) * w;
}

function angleBinBounds(idx) {
  if (idx < 0 || idx >= nFineAngleBins) return "?";
  const w = 2 / nFineAngleBins;
  const lo = -1 + idx * w;
  const hi = -1 + (idx + 1) * w;
  return `[${compactExp(lo)}, ${compactExp(hi)}]`;
}

function updatePlotSelectionInfo() {
  if (!manifest) return;
  const mode = plotModeSelect.value;
  if (mode === "ep-profile") {
    const eIdx = parseInt($("plot-e-group").value);
    const xiIdx = parseInt($("plot-xi-bin").value);
    plotSelLeftLabel.textContent = "Selected E";
    plotSelLeft.textContent = groupBoundsEV(eIdx);
    plotSelRightBox.hidden = false;
    plotSelRightLabel.textContent = "Selected \u03BE";
    plotSelRight.textContent = angleBinBounds(xiIdx);
  } else if (mode === "xi-profile") {
    const eIdx = parseInt($("plot-e-group-xi").value);
    const epIdx = parseInt($("plot-ep-group-xi").value);
    plotSelLeftLabel.textContent = "Selected E";
    plotSelLeft.textContent = groupBoundsEV(eIdx);
    plotSelRightBox.hidden = false;
    plotSelRightLabel.textContent = "Selected E\u2032";
    plotSelRight.textContent = groupBoundsEV(epIdx);
  } else {
    const xiIdx = parseInt($("plot-xi-bin-matrix").value);
    plotSelLeftLabel.textContent = "Selected \u03BE";
    plotSelLeft.textContent = angleBinBounds(xiIdx);
    plotSelRightBox.hidden = true;
  }
}

const npzCache = new Map();

async function fetchNpzMatrix(weight, quantity, tempIndex) {
  const key = `${weight}|${quantity}|${tempIndex}`;
  if (npzCache.has(key)) return npzCache.get(key);
  const entry = manifests[weight].temperatures[tempIndex];
  const file = quantity === "sigma" ? entry.file : entry.dsigma_dT_file;
  pyodide.FS.writeFile("/tmp/plot_input.npz", await fetchBuffer(dataUrl(weight, file)));
  const matProxy = await pyodide.runPythonAsync(`
import numpy as np
_bytes = open("/tmp/plot_input.npz", "rb").read()
_m = (_load_table(_bytes)["sigma"] if "${quantity}" == "sigma" else _load_dsigma_dT(_bytes)[0]).ravel()
_m
`);
  const flat = matProxy.toJs();
  matProxy.destroy();
  const result = { data: flat, nG: nFineGroups, nA: nFineAngleBins };
  npzCache.set(key, result);
  return result;
}

function getMatrixValue(flat, nG, nA, gIn, gOut, aBin) {
  return flat[gIn * nG * nA + gOut * nA + aBin];
}

async function handlePlot() {
  if (!pyodide || !manifest) return;
  const mode = plotModeSelect.value;
  const tempIdx = parseInt(plotTempSelect.value);
  if (isNaN(tempIdx)) return;

  plotButton.disabled = true;
  plotFormError.hidden = true;

  try {
    const weight = plotWeightSelect.value;
    const quantity = plotQuantitySelect.value;
    if (!manifests[weight]) throw new Error(`${WEIGHT_LABELS[weight]} tables are not available.`);
    const { data: flat, nG, nA } = await fetchNpzMatrix(weight, quantity, tempIdx);
    const tempEntry = manifests[weight].temperatures[tempIdx];
    const tempK = tempEntry.temperature_K;
    const signed = quantity !== "sigma";
    const qSymbol = signed ? "dσ/dT" : "σ";
    const qUnits = signed ? "[cm²/K]" : "[cm²]";
    const tag = `${WEIGHT_LABELS[weight]} ${qSymbol}`;

    if (mode === "ep-profile") {
      const eIdx = parseInt($("plot-e-group").value);
      const xiIdx = parseInt($("plot-xi-bin").value);
      if (eIdx < 0 || eIdx >= nG || xiIdx < 0 || xiIdx >= nA) {
        plotFormError.textContent = `Group/bin index out of range (0\u2013${nG - 1} for groups, 0\u2013${nA - 1} for bins).`;
        plotFormError.hidden = false;
        return;
      }
      const xVals = [], yVals = [];
      for (let g = 0; g < nG; g++) {
        xVals.push(groupMidpoint(g));
        yVals.push(getMatrixValue(flat, nG, nA, eIdx, g, xiIdx));
      }
      const logX = $("plot-log-x").checked;
      const logY = $("plot-log-y").checked;
      const xLabel = "Outgoing energy E\u2032 [keV]";
      const yLabel = `${qSymbol}(E \u2192 E\u2032, \u03BE) ${qUnits}`;
      const title = `${tag} \u2014 E\u2032 profile: E = ${compactExp(groupMidpoint(eIdx) * 1e3)} eV, \u03BE\u2248${angleBinMidpoint(xiIdx).toFixed(3)}, T = ${compactExp(tempK * 8.617333262e-5)} eV`;
      renderLinePlot(plotChart, xVals, yVals, { logX, logY, xLabel, yLabel, title, signed });
      plotDescription.textContent = title;
    } else if (mode === "xi-profile") {
      const eIdx = parseInt($("plot-e-group-xi").value);
      const epIdx = parseInt($("plot-ep-group-xi").value);
      if (eIdx < 0 || eIdx >= nG || epIdx < 0 || epIdx >= nG) {
        plotFormError.textContent = `Group index out of range (0\u2013${nG - 1}).`;
        plotFormError.hidden = false;
        return;
      }
      const xVals = [], yVals = [];
      for (let a = 0; a < nA; a++) {
        xVals.push(angleBinMidpoint(a));
        yVals.push(getMatrixValue(flat, nG, nA, eIdx, epIdx, a));
      }
      const logX = false;
      const logY = $("plot-log-y").checked;
      const xLabel = "\u03BE = cos \u03B8";
      const yLabel = `${qSymbol}(E \u2192 E\u2032, \u03BE) ${qUnits}`;
      const title = `${tag} \u2014 \u03BE profile: E = ${compactExp(groupMidpoint(eIdx) * 1e3)} eV, E\u2032 = ${compactExp(groupMidpoint(epIdx) * 1e3)} eV, T = ${compactExp(tempK * 8.617333262e-5)} eV`;
      renderLinePlot(plotChart, xVals, yVals, { logX, logY, xLabel, yLabel, title, signed });
      plotDescription.textContent = title;
    } else {
      const xiIdx = parseInt($("plot-xi-bin-matrix").value);
      if (xiIdx < 0 || xiIdx >= nA) {
        plotFormError.textContent = `Bin index out of range (0\u2013${nA - 1}).`;
        plotFormError.hidden = false;
        return;
      }
      const matrixData = [];
      for (let gIn = 0; gIn < nG; gIn++) {
        const row = [];
        for (let gOut = 0; gOut < nG; gOut++) {
          row.push(getMatrixValue(flat, nG, nA, gIn, gOut, xiIdx));
        }
        matrixData.push(row);
      }
      const xEdges = Array.from(fineBoundaries);
      const yEdges = Array.from(fineBoundaries);
      const title = `${tag} \u2014 E/E\u2032 matrix: \u03BE\u2248${angleBinMidpoint(xiIdx).toFixed(3)} (bin ${xiIdx}), T = ${compactExp(tempK * 8.617333262e-5)} eV`;
      renderHeatmap(plotChart, matrixData, xEdges, yEdges, {
        xLabel: "Outgoing energy E\u2032 [keV]",
        yLabel: "Incoming energy E [keV]",
        zLabel: `${qSymbol} ${qUnits}`,
        signed,
        title
      });
      plotDescription.textContent = title;
    }

    plotSection.hidden = false;
    plotPlaceholder.hidden = true;
    plotChart.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    plotFormError.textContent = "Plot error: " + err.message;
    plotFormError.hidden = false;
    console.error(err);
  } finally {
    plotButton.disabled = false;
  }
}


/* ── Initialization ────────────────────────────────────── */

async function loadManifest() {
  try {
    const results = await Promise.all(WEIGHTS.map(async (w) => {
      const resp = await fetch(`data/${w}/manifest.json`);
      if (!resp.ok) throw new Error(`${w}: HTTP ${resp.status}: ${resp.statusText}`);
      return [w, await resp.json()];
    }).map((p) => p.catch((err) => { console.error(err); return null; })));
    for (const r of results) if (r) manifests[r[0]] = r[1];
    manifest = manifests.uniform || Object.values(manifests)[0];
    if (!manifest) throw new Error("no weight-function manifests found");
    for (const [w, m] of Object.entries(manifests)) {
      const sameGrid = m.n_groups === manifest.n_groups && m.n_angle_bins === manifest.n_angle_bins &&
        m.temperatures.length === manifest.temperatures.length;
      if (!sameGrid) throw new Error(`${w} manifest grid differs from ${manifest.weight_function}`);
    }
    for (const w of WEIGHTS) {
      const available = !!manifests[w];
      document.querySelectorAll(`input[name="weight-fn"][value="${w}"]`).forEach((r) => { r.disabled = !available; });
      const opt = plotWeightSelect.querySelector(`option[value="${w}"]`);
      if (opt) opt.disabled = !available;
    }
    if (!manifests[currentWeight()]) {
      const first = WEIGHTS.find((w) => manifests[w]);
      document.querySelector(`input[name="weight-fn"][value="${first}"]`).checked = true;
      plotWeightSelect.value = first;
    }
    updateWeightHint();
  } catch (err) {
    loadingMsg.textContent = "Failed to load dataset metadata: " + err.message;
    loadingMsg.classList.remove("status");
    loadingMsg.classList.add("error");
    console.error(err);
    return;
  }

  fineBoundaries = new Float64Array(manifest.boundaries_keV);
  nFineGroups = manifest.n_groups;
  nFineAngleBins = manifest.n_angle_bins;

  nGroupsInput.max = nFineGroups;
  nGroupsInput.value = nFineGroups;
  nGroupsHint.textContent = `1 \u2013 ${nFineGroups}`;
  nAnglesInput.max = nFineAngleBins;
  nAnglesInput.value = nFineAngleBins;

  const tMin = manifest.temperatures[0];
  const tMax = manifest.temperatures[manifest.temperatures.length - 1];

  function texExp(v) {
    const s = v.toExponential(1);
    const [m, e] = s.split("e");
    return `${m}\\times10^{${parseInt(e, 10)}}`;
  }
  gridSummary.innerHTML =
    `${nFineGroups} energy groups (\\(${texExp(fineBoundaries[0] * 1e3)}\\) \u2013 \\(${texExp(fineBoundaries[fineBoundaries.length - 1] * 1e3)}\\) eV, log-spaced)<br>` +
    `${nFineAngleBins} angle bins (\\(\\xi = \\cos\\theta \\in [-1,\\,1]\\))<br>` +
    `${manifest.temperatures.length} temperatures (\\(${texExp(tMin.temperature_K * 8.617333262e-5)}\\) \u2013 \\(${texExp(tMax.temperature_K * 8.617333262e-5)}\\) eV, log-spaced)`;
  if (window.renderMathInElement) renderMathInElement(gridSummary, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }] });

  tempSelect.innerHTML = "";
  for (const t of manifest.temperatures) {
    t._index = t.index;
    const opt = document.createElement("option");
    opt.value = t.index;
    opt.textContent = formatTemp(t);
    tempSelect.appendChild(opt);
  }

  plotTempSelect.innerHTML = "";
  for (const t of manifest.temperatures) {
    const opt = document.createElement("option");
    opt.value = t.index;
    opt.textContent = formatTemp(t);
    plotTempSelect.appendChild(opt);
  }

  [$("plot-e-group"), $("plot-e-group-xi"), $("plot-ep-group-xi")].forEach((el) => {
    if (el) { el.max = nFineGroups - 1; }
  });
  [$("plot-xi-bin"), $("plot-xi-bin-matrix")].forEach((el) => {
    if (el) { el.max = nFineAngleBins - 1; }
  });

  loadingMsg.hidden = true;
  downloadForm.hidden = false;
  plotForm.hidden = false;

  updateBoundariesDisplay();
  updateAngleHints();
  updateSummary();
  updateButtonStates();
  updatePlotPanels();
  updatePlotSelectionInfo();
}

async function loadPyodide_() {
  pyodideLoading.hidden = false;
  const pyodideModule = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/",
  });
  await pyodideModule.loadPackage("numpy");

  const collapseResp = await fetch("py/collapse.py");
  collapseCode = await collapseResp.text();
  await pyodideModule.runPythonAsync(collapseCode);

  pyodide = pyodideModule;
  pyodideLoading.hidden = true;
  loadingMsg.hidden = true;
  downloadForm.hidden = false;
  updateButtonStates();
}

async function init() {
  wireDownloadEvents();
  wirePlotEvents();

  const manifestPromise = loadManifest();

  const pyodideScript = document.createElement("script");
  pyodideScript.src = "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js";
  pyodideScript.addEventListener("load", () => {
    loadPyodide_().catch((err) => {
      pyodideLoading.hidden = true;
      loadingMsg.textContent = "Failed to load Python runtime: " + err.message;
      loadingMsg.hidden = false;
      loadingMsg.classList.remove("status");
      loadingMsg.classList.add("error");
      console.error(err);
    });
  });
  document.head.appendChild(pyodideScript);

  const jszipScript = document.createElement("script");
  jszipScript.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
  jszipScript.addEventListener("load", () => {
    jszipLoaded = true;
    updateButtonStates();
  });
  document.head.appendChild(jszipScript);

  await manifestPromise;
}

init();
