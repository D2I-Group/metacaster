/* MetaCaster project page — data extracted from the camera-ready paper. */

/* ------------------------------------------------------------ main table */
/* Table 1: MSE on IND / OOD corpora, K in {10, 30, 50}. Lower is better.
   Best / second-best per row are computed in main.js over METHODS only;
   the two reference columns are excluded from ranking, matching the paper. */

const METHODS = [
  { key: "ours",      label: "MetaCaster", group: "Ours" },
  { key: "timedp",    label: "TimeDP",     group: "Generation Models" },
  { key: "verbalts",  label: "VerbalTS",   group: "Generation Models" },
  { key: "t2s",       label: "T2S",        group: "Generation Models" },
  { key: "diffts",    label: "DiffTS",     group: "Generation Models" },
  { key: "timevae",   label: "TimeVAE",    group: "Generation Models" },
  { key: "repeat",    label: "Repeat",     group: "Augmentation" },
  { key: "bootstrap", label: "Bootstrap",  group: "Augmentation" },
  { key: "jitter",    label: "Jitter",     group: "Augmentation" },
  { key: "magwarp",   label: "MagWarp",    group: "Augmentation" },
];

const REFERENCES = [
  { key: "dsup", label: "D<sub>sup</sub>" },
  { key: "dtr",  label: "D<sub>tr</sub>" },
];

/* Each row: [dataset, ...10 method MSEs, D_sup, D_tr] */
const MAIN_RESULTS = {
  10: {
    IND: [
      ["ETTm1",       0.376, 1.049, 1.004, 0.922, 1.154, 0.749, 0.781, 0.748, 0.746, 1.919, 0.687, 0.316],
      ["Electricity", 0.300, 0.288, 1.061, 1.150, 0.365, 0.328, 0.380, 0.379, 0.375, 7.057, 0.721, 0.121],
      ["Seattle",     1.522, 1.420, 1.800, 1.792, 1.643, 1.759, 1.955, 1.849, 1.925, 43.24, 2.341, 1.024],
      ["SZTaxi",      0.104, 0.124, 0.116, 0.118, 0.111, 0.114, 0.134, 0.118, 0.123, 1.122, 0.187, 0.096],
      ["Sales",       2.616, 2.587, 2.419, 2.758, 4.323, 2.795, 4.175, 2.949, 2.946, 4.321, 3.004, 2.927],
      ["Bitbrains",   0.068, 0.079, 0.098, 0.094, 0.121, 0.109, 0.197, 0.149, 0.159, 0.826, 0.152, 0.150],
      ["Solar",       0.158, 0.208, 0.513, 0.552, 0.238, 0.237, 0.278, 0.272, 0.272, 0.446, 0.440, 0.157],
    ],
    OOD: [
      ["Saugeen",     1.960, 1.492, 1.474, 1.578, 2.281, 2.705, 3.119, 2.887, 2.908, 3.893, 1.859, 1.183],
      ["USbirths",    0.702, 2.690, 1.501, 1.540, 1.775, 0.607, 0.752, 0.715, 0.714, 12.08, 1.204, 0.250],
      ["M4",         2.093, 2.178, 2.331, 2.444, 2.688, 2.993, 3.250, 3.101, 3.057, 8.654, 2.628, 2.330],
    ],
  },
  30: {
    IND: [
      ["ETTm1",       0.345, 1.049, 1.027, 0.910, 1.211, 0.684, 0.714, 0.664, 0.662, 0.943, 0.602, 0.316],
      ["Electricity", 0.226, 0.287, 1.055, 1.140, 0.363, 0.268, 0.305, 0.287, 0.290, 1.644, 0.657, 0.121],
      ["Seattle",     1.177, 1.414, 1.806, 1.777, 1.673, 2.158, 1.738, 1.656, 1.622, 21.15, 1.785, 1.024],
      ["SZTaxi",      0.114, 0.121, 0.116, 0.115, 0.112, 0.156, 0.162, 0.141, 0.135, 1.626, 0.150, 0.096],
      ["Sales",       2.362, 2.479, 2.455, 2.942, 4.439, 2.598, 3.603, 3.131, 3.158, 3.422, 2.611, 2.927],
      ["Bitbrains",   0.125, 0.084, 0.097, 0.095, 0.130, 0.114, 0.463, 0.469, 0.489, 0.777, 0.137, 0.150],
      ["Solar",       0.152, 0.220, 0.515, 0.576, 0.240, 0.234, 0.254, 0.252, 0.248, 0.356, 0.379, 0.157],
    ],
    OOD: [
      ["Saugeen",     1.464, 1.495, 1.480, 1.593, 2.286, 2.019, 2.419, 2.266, 2.277, 2.620, 1.547, 1.183],
      ["USbirths",    0.533, 1.980, 1.509, 1.536, 1.732, 0.474, 0.546, 0.524, 0.525, 5.107, 0.803, 0.250],
      ["M4",         2.112, 2.248, 2.394, 2.522, 2.663, 3.402, 3.126, 3.109, 3.169, 10.321, 2.950, 2.330],
    ],
  },
  50: {
    IND: [
      ["ETTm1",       0.267, 1.063, 1.035, 0.931, 1.284, 0.548, 0.582, 0.556, 0.546, 0.592, 0.456, 0.316],
      ["Electricity", 0.191, 0.279, 1.053, 1.197, 0.352, 0.227, 0.244, 0.234, 0.233, 1.287, 0.472, 0.121],
      ["Seattle",     1.125, 1.423, 1.819, 1.779, 1.732, 2.810, 1.537, 1.462, 1.497, 12.78, 1.533, 1.024],
      ["SZTaxi",      0.110, 0.120, 0.114, 0.113, 0.113, 0.318, 0.230, 0.222, 0.216, 0.352, 0.143, 0.096],
      ["Sales",       2.835, 2.605, 2.474, 2.852, 4.459, 2.683, 2.778, 2.590, 2.560, 3.039, 2.501, 2.927],
      ["Bitbrains",   0.119, 0.097, 0.099, 0.095, 0.124, 0.155, 0.929, 0.976, 0.987, 1.214, 0.119, 0.150],
      ["Solar",       0.149, 0.234, 0.520, 0.553, 0.250, 0.547, 0.234, 0.231, 0.229, 0.333, 0.296, 0.157],
    ],
    OOD: [
      ["Saugeen",     1.415, 1.464, 1.482, 1.591, 2.291, 1.767, 2.462, 2.380, 2.371, 2.528, 1.515, 1.183],
      ["USbirths",    0.424, 2.366, 1.509, 1.531, 1.687, 0.367, 0.396, 0.380, 0.376, 5.787, 0.574, 0.250],
      ["M4",         1.916, 2.157, 2.455, 2.444, 2.896, 3.279, 3.330, 3.300, 3.282, 12.210, 2.560, 2.330],
    ],
  },
};

/* Wins over all 30 (dataset, K) cells — from the last row of Table 1. */
const WINS = [19, 3, 3, 1, 1, 3, 0, 0, 0, 0];

/* --------------------------------------------------------- ablation table */
/* Table 2: MSE at K = 30. "Overall" is the normalised MSE of Eq. (3). */
const ABLATION_DATASETS = [
  "ETTm1", "Electricity", "Seattle", "SZTaxi", "Sales", "Bitbrains", "Solar",
  "Saugeen", "USbirths", "M4",
];

const ABLATION = [
  { label: "MetaCaster", group: null, ours: true,
    v: [0.345, 0.226, 1.177, 0.114, 2.362, 0.125, 0.152, 1.464, 0.533, 2.112], overall: 0.267 },

  { label: "Loss → MMD", group: "Generation objective",
    v: [0.708, 0.157, 1.204, 0.150, 3.581, 0.575, 0.254, 2.339, 0.403, 2.556], overall: 0.764 },
  { label: "Loss → Wasserstein", group: "Generation objective",
    v: [0.702, 0.284, 1.734, 0.152, 3.553, 0.483, 0.254, 2.406, 0.535, 3.105], overall: 0.940 },

  { label: "Remove context 𝖢", group: "Textual context",
    v: [0.386, 0.293, 1.425, 0.144, 2.595, 0.158, 0.281, 2.004, 0.534, 2.312], overall: 0.521 },

  { label: "Gemini-3.1-Pro", group: "LLM backbone",
    v: [0.430, 0.226, 1.177, 0.114, 2.362, 0.120, 0.151, 1.396, 0.533, 2.129], overall: 0.288 },
  { label: "Claude-Opus-4.7", group: "LLM backbone",
    v: [0.494, 0.226, 1.189, 0.114, 2.362, 0.151, 0.150, 1.491, 0.541, 2.129], overall: 0.321 },
  { label: "Qwen3.5-122B-A10B", group: "LLM backbone",
    v: [0.494, 0.223, 1.177, 0.114, 2.362, 0.199, 0.152, 1.729, 0.533, 2.129], overall: 0.366 },
  { label: "GPT-5.3-Codex", group: "LLM backbone",
    v: [0.557, 0.209, 1.115, 0.087, 2.229, 0.098, 0.148, 1.465, 1.488, 2.269], overall: 0.677 },
];

/* ------------------------------------------------------------- LT-Lib pool */
/* Table 8: the 23 lightweight forecasters. params is the raw count for
   sorting; the label is what the paper prints. */
const LTLIB = [
  { name: "Vanilla Linear", family: "Linear",      params: 65e3,  plabel: "65K",  macs: 0.45,  lat: 0.03, vram: 8.4,  ref: "Zeng et al., 2023", url: "https://arxiv.org/abs/2205.13504" },
  { name: "DLinear",        family: "Linear",      params: 129e3, plabel: "129K", macs: 0.90,  lat: 0.10, vram: 8.7,  ref: "Zeng et al., 2023", url: "https://arxiv.org/abs/2205.13504" },
  { name: "NLinear",        family: "Linear",      params: 65e3,  plabel: "65K",  macs: 0.45,  lat: 0.04, vram: 8.4,  ref: "Zeng et al., 2023", url: "https://arxiv.org/abs/2205.13504" },
  { name: "RLinear",        family: "Linear",      params: 65e3,  plabel: "65K",  macs: 0.45,  lat: 0.17, vram: 8.4,  ref: "Li et al., 2023", url: "https://arxiv.org/abs/2305.10721" },
  { name: "CrossLinear",    family: "Linear",      params: 2.5e6, plabel: "2.5M", macs: 28.45, lat: 0.72, vram: 23.1, ref: "Zhou et al., 2025", url: "https://arxiv.org/abs/2505.23116" },
  { name: "MixLinear",      family: "Linear",      params: 243,   plabel: "243",  macs: 0.08,  lat: 0.43, vram: 8.3,  ref: "Ma et al., 2024", url: "https://arxiv.org/abs/2410.02081" },

  { name: "TSMixer",        family: "MLP",         params: 153e3, plabel: "153K", macs: 1.66,  lat: 0.42, vram: 10.1, ref: "Chen et al., 2023", url: "https://arxiv.org/abs/2303.06053" },
  { name: "LightTS",        family: "MLP",         params: 74e3,  plabel: "74K",  macs: 0.70,  lat: 0.75, vram: 10.1, ref: "Zhang et al., 2022", url: "https://arxiv.org/abs/2207.01186" },
  { name: "PatchMLP",       family: "MLP",         params: 2.5e6, plabel: "2.5M", macs: 18.12, lat: 1.21, vram: 20.1, ref: "Tang and Zhang, 2025", url: "https://arxiv.org/abs/2405.13575" },
  { name: "xPatch",         family: "MLP",         params: 770e3, plabel: "770K", macs: 8.16,  lat: 1.44, vram: 13.6, ref: "Stitsyuk and Choi, 2025", url: "https://arxiv.org/abs/2412.17323" },
  { name: "CMoS",           family: "MLP",         params: 13e3,  plabel: "13K",  macs: 0.36,  lat: 1.09, vram: 9.3,  ref: "Si et al., 2025", url: "https://arxiv.org/abs/2505.19090" },
  { name: "PatchTSMixer",   family: "MLP",         params: 553e3, plabel: "553K", macs: 20.83, lat: 1.12, vram: 14.4, ref: "Ekambaram et al., 2023", url: "https://arxiv.org/abs/2306.09364" },

  { name: "FITS",           family: "Freq/Filter", params: 925,   plabel: "925",  macs: 0.01,  lat: 0.32, vram: 8.3,  ref: "Xu et al., 2024", url: "https://arxiv.org/abs/2307.03756" },
  { name: "CycleNet",       family: "Freq/Filter", params: 65e3,  plabel: "65K",  macs: 0.45,  lat: 0.32, vram: 8.5,  ref: "Lin et al., 2024a", url: "https://arxiv.org/abs/2409.18479" },
  { name: "PaiFilter",      family: "Freq/Filter", params: 136e3, plabel: "136K", macs: 0.95,  lat: 0.35, vram: 9.8,  ref: "Yi et al., 2024", url: "https://arxiv.org/abs/2411.01623" },
  { name: "TexFilter",      family: "Freq/Filter", params: 179e3, plabel: "179K", macs: 0.94,  lat: 1.62, vram: 9.9,  ref: "Yi et al., 2024", url: "https://arxiv.org/abs/2411.01623" },
  { name: "FreqCycle",      family: "Freq/Filter", params: 64e3,  plabel: "64K",  macs: 0.43,  lat: 0.78, vram: 9.5,  ref: "Zhang et al., 2026", url: "https://arxiv.org/abs/2603.09661" },

  { name: "TimeMixer",      family: "Mixing",      params: 377e3, plabel: "377K", macs: 50.08, lat: 0.80, vram: 65.0, ref: "Wang et al., 2024", url: "https://arxiv.org/abs/2405.14616" },
  { name: "TimeBase",       family: "Mixing",      params: 146,   plabel: "146",  macs: 0.02,  lat: 0.29, vram: 9.2,  ref: "Huang et al., 2025a" },
  { name: "TimeBridge",     family: "Mixing",      params: 36e3,  plabel: "36K",  macs: 3.01,  lat: 2.09, vram: 10.4, ref: "Liu et al., 2025b", url: "https://arxiv.org/abs/2410.04442" },
  { name: "TimeEmb",        family: "Mixing",      params: 308e3, plabel: "308K", macs: 1.89,  lat: 0.47, vram: 10.5, ref: "Xia et al., 2025", url: "https://arxiv.org/abs/2510.00461" },
  { name: "Amplifier",      family: "Mixing",      params: 327e3, plabel: "327K", macs: 1.06,  lat: 0.73, vram: 10.6, ref: "Fei et al., 2025", url: "https://arxiv.org/abs/2501.17216" },
  { name: "SparseTSF",      family: "Mixing",      params: 137,   plabel: "137",  macs: 0.08,  lat: 0.20, vram: 8.2,  ref: "Lin et al., 2024b", url: "https://arxiv.org/abs/2405.00946" },
];
