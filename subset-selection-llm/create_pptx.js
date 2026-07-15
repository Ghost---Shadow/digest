const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

pres.layout = "LAYOUT_16x9";
pres.author = "Subset Selection Review";
pres.title = "A Review on the Recent Advances in Subset Selection Problems in Large Language Models";

// Color palette - Midnight Executive
const C = {
  navy: "1E2761",
  darkNavy: "141B3D",
  ice: "CADCFC",
  white: "FFFFFF",
  offWhite: "F4F6FC",
  accent: "4A6CF7",
  accent2: "6C83E8",
  gray: "6B7280",
  darkGray: "374151",
  lightGray: "E5E7EB",
  cardBg: "FFFFFF",
  pretrain: "3B82F6",
  finetune: "8B5CF6",
  coreset: "10B981",
};

const FONTS = { heading: "Georgia", body: "Calibri" };
const mkShadow = () => ({ type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.1 });

// Helper: add a dark slide with title
function addDarkSlide() {
  const s = pres.addSlide();
  s.background = { color: C.darkNavy };
  return s;
}

// Helper: add a light content slide with header bar
function addContentSlide(title, subtitle) {
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  // Header bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.9, fill: { color: C.navy } });
  s.addText(title, { x: 0.5, y: 0.1, w: 9, h: 0.7, fontSize: 22, fontFace: FONTS.heading, color: C.white, bold: true, margin: 0 });
  if (subtitle) {
    s.addText(subtitle, { x: 0.5, y: 0.55, w: 9, h: 0.3, fontSize: 11, fontFace: FONTS.body, color: C.ice, margin: 0 });
  }
  return s;
}

// Helper: add a card shape
function addCard(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.cardBg },
    shadow: mkShadow(),
    line: opts.border ? { color: opts.border, width: 1 } : undefined,
  });
}

// Helper: bullet list in a region
function addBullets(slide, items, x, y, w, h, opts = {}) {
  const textItems = items.map((item, i) => ({
    text: item,
    options: {
      bullet: true,
      breakLine: i < items.length - 1,
      fontSize: opts.fontSize || 13,
      color: opts.color || C.darkGray,
      fontFace: FONTS.body,
    }
  }));
  slide.addText(textItems, { x, y, w, h, valign: "top", paraSpaceAfter: 4 });
}

// ═══════════════════════════════════════════════════
// SLIDE 1: TITLE
// ═══════════════════════════════════════════════════
{
  const s = addDarkSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.darkNavy } });
  // Accent line
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 1.6, w: 2, h: 0.05, fill: { color: C.accent } });
  s.addText("A Review on the Recent Advances in", {
    x: 1.5, y: 1.8, w: 7, h: 0.6, fontSize: 20, fontFace: FONTS.body, color: C.ice, margin: 0
  });
  s.addText("Subset Selection Problems\nin Large Language Models", {
    x: 1.5, y: 2.3, w: 7, h: 1.4, fontSize: 32, fontFace: FONTS.heading, color: C.white, bold: true, margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 3.9, w: 2, h: 0.05, fill: { color: C.accent } });
  s.addText("A Survey of 10 Papers from ICLR 2025 and Related Venues", {
    x: 1.5, y: 4.1, w: 7, h: 0.4, fontSize: 14, fontFace: FONTS.body, color: C.ice, margin: 0
  });
}

// ═══════════════════════════════════════════════════
// SLIDE 2: OUTLINE
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("Outline");
  const sections = [
    ["01", "Introduction & Motivation", "Why subset selection matters for LLMs"],
    ["02", "Taxonomy of Approaches", "Categorizing 10 recent papers"],
    ["03", "Pre-training Data Selection", "CRISP, PDS, QUAD, TrackStar"],
    ["04", "Instruction Fine-tuning Selection", "DELIFT, ADAPT-INF, KMQ, ROSE"],
    ["05", "Coreset & Task-Specific Selection", "STAFF, LENS"],
    ["06", "Comparative Analysis & Themes", "Cross-cutting patterns and insights"],
    ["07", "Open Challenges & Conclusion", "Future directions"],
  ];
  sections.forEach((sec, i) => {
    const yy = 1.15 + i * 0.6;
    addCard(s, 0.5, yy, 9, 0.5);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: yy, w: 0.07, h: 0.5, fill: { color: C.accent } });
    s.addText(sec[0], { x: 0.7, y: yy, w: 0.5, h: 0.5, fontSize: 16, fontFace: FONTS.heading, color: C.accent, bold: true, valign: "middle", margin: 0 });
    s.addText(sec[1], { x: 1.3, y: yy, w: 4, h: 0.5, fontSize: 14, fontFace: FONTS.heading, color: C.darkGray, bold: true, valign: "middle", margin: 0 });
    s.addText(sec[2], { x: 5.5, y: yy, w: 3.8, h: 0.5, fontSize: 11, fontFace: FONTS.body, color: C.gray, valign: "middle", margin: 0 });
  });
}

// ═══════════════════════════════════════════════════
// SLIDE 3: INTRODUCTION
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("Introduction: Why Subset Selection?");
  // Left column - motivation
  addCard(s, 0.5, 1.1, 4.3, 4.2);
  s.addText("The Challenge", { x: 0.7, y: 1.2, w: 4, h: 0.4, fontSize: 16, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  addBullets(s, [
    "LLM training requires massive datasets (trillions of tokens)",
    "Not all data contributes equally to model performance",
    "Redundant or low-quality data wastes compute resources",
    "Training costs scale linearly with dataset size",
    "Careful selection can match full-data performance with a fraction of samples",
  ], 0.7, 1.7, 3.9, 3.4, { fontSize: 12 });

  // Right column - goals
  addCard(s, 5.2, 1.1, 4.3, 4.2);
  s.addText("Goals of Subset Selection", { x: 5.4, y: 1.2, w: 4, h: 0.4, fontSize: 16, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  addBullets(s, [
    "Quality: Select samples most useful for learning",
    "Diversity: Cover the full data distribution",
    "Efficiency: Minimize computational overhead of selection itself",
    "Scalability: Handle billions of training examples",
    "Task Alignment: Match selection to downstream objectives",
  ], 5.4, 1.7, 3.9, 3.4, { fontSize: 12 });
}

// ═══════════════════════════════════════════════════
// SLIDE 4: TAXONOMY
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("Taxonomy of Approaches");
  const cats = [
    { label: "Pre-training Data Selection", color: C.pretrain, papers: ["CRISP (Apple)", "PDS (Tsinghua/Microsoft)", "QUAD (Beijing IT/SenseTime)", "TrackStar (Google DeepMind)"], desc: "Select subsets from large corpora for pre-training" },
    { label: "Instruction Fine-tuning Selection", color: C.finetune, papers: ["DELIFT (UIUC/IBM)", "ADAPT-INF (UNC Chapel Hill)", "KMQ (Anonymous)", "ROSE (Anonymous)"], desc: "Curate instruction data for fine-tuning" },
    { label: "Coreset & Task-Specific", color: C.coreset, papers: ["STAFF (Xi'an Jiaotong/NTU)", "LENS (Fudan University)"], desc: "Select task-specific coresets or ICL examples" },
  ];
  cats.forEach((cat, i) => {
    const yy = 1.15 + i * 1.4;
    addCard(s, 0.5, yy, 9, 1.25);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: yy, w: 0.08, h: 1.25, fill: { color: cat.color } });
    s.addText(cat.label, { x: 0.75, y: yy + 0.05, w: 3.5, h: 0.35, fontSize: 15, fontFace: FONTS.heading, color: cat.color, bold: true, margin: 0 });
    s.addText(cat.desc, { x: 0.75, y: yy + 0.35, w: 3.5, h: 0.3, fontSize: 10, fontFace: FONTS.body, color: C.gray, margin: 0 });
    cat.papers.forEach((p, j) => {
      const px = 4.5 + (j % 2) * 2.5;
      const py = yy + 0.15 + Math.floor(j / 2) * 0.5;
      s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: 2.3, h: 0.42, fill: { color: cat.color, transparency: 90 }, line: { color: cat.color, width: 1 } });
      s.addText(p, { x: px, y: py, w: 2.3, h: 0.42, fontSize: 10, fontFace: FONTS.body, color: cat.color, align: "center", valign: "middle", margin: 0 });
    });
  });
}

// ═══════════════════════════════════════════════════
// PAPER SLIDES HELPER
// ═══════════════════════════════════════════════════
function addPaperMethodSlide(title, venue, oneLiner, method, keyIdea, models, datasets) {
  const s = addContentSlide(title, venue);
  // One-liner box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 9, h: 0.5, fill: { color: C.accent, transparency: 90 }, line: { color: C.accent, width: 1 } });
  s.addText(oneLiner, { x: 0.7, y: 1.05, w: 8.6, h: 0.5, fontSize: 12, fontFace: FONTS.body, color: C.accent, italic: true, valign: "middle", margin: 0 });

  // Method card
  addCard(s, 0.5, 1.7, 5.8, 3.6);
  s.addText("Method", { x: 0.7, y: 1.8, w: 2, h: 0.35, fontSize: 15, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  addBullets(s, method, 0.7, 2.2, 5.4, 2.9, { fontSize: 12 });

  // Right sidebar
  addCard(s, 6.5, 1.7, 3, 1.7);
  s.addText("Key Idea", { x: 6.7, y: 1.8, w: 2.6, h: 0.3, fontSize: 13, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  s.addText(keyIdea, { x: 6.7, y: 2.15, w: 2.6, h: 1.1, fontSize: 11, fontFace: FONTS.body, color: C.darkGray, valign: "top", margin: 0 });

  addCard(s, 6.5, 3.55, 3, 1.75);
  s.addText("Models", { x: 6.7, y: 3.6, w: 2.6, h: 0.25, fontSize: 12, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  s.addText(models, { x: 6.7, y: 3.85, w: 2.6, h: 0.55, fontSize: 10, fontFace: FONTS.body, color: C.gray, valign: "top", margin: 0 });
  s.addText("Datasets", { x: 6.7, y: 4.4, w: 2.6, h: 0.25, fontSize: 12, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  s.addText(datasets, { x: 6.7, y: 4.65, w: 2.6, h: 0.55, fontSize: 10, fontFace: FONTS.body, color: C.gray, valign: "top", margin: 0 });
  return s;
}

function addPaperResultsSlide(title, baselines, results, takeaway) {
  const s = addContentSlide(title + " - Results & Analysis");

  // Baselines
  addCard(s, 0.5, 1.1, 4.3, 2.0);
  s.addText("Baselines", { x: 0.7, y: 1.2, w: 3, h: 0.3, fontSize: 14, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  addBullets(s, baselines, 0.7, 1.55, 3.9, 1.4, { fontSize: 11 });

  // Key results
  addCard(s, 5.2, 1.1, 4.3, 2.0);
  s.addText("Key Results", { x: 5.4, y: 1.2, w: 3, h: 0.3, fontSize: 14, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  addBullets(s, results, 5.4, 1.55, 3.9, 1.4, { fontSize: 11 });

  // Takeaway
  addCard(s, 0.5, 3.3, 9, 2.0);
  s.addText("Takeaway", { x: 0.7, y: 3.4, w: 2, h: 0.35, fontSize: 14, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
  addBullets(s, takeaway, 0.7, 3.8, 8.6, 1.3, { fontSize: 12 });
  return s;
}

// ═══════════════════════════════════════════════════
// PAPER 1: ADAPT-INF
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "ADAPT-INF: Scalable Continual Multimodal Instruction Tuning",
  "ICLR 2025 | UNC Chapel Hill",
  "Compute gradients, run K-means clustering, then select the most informative samples (entropy) from each cluster",
  [
    "Extract gradient vectors from model layers for each sample",
    "Perform pseudo-task clustering using K-means on gradients",
    "Use scoring function experts to evaluate sample importance per cluster",
    "Select balanced subset using entropy-based CCS sampling strategy",
    "Periodically prune semantically redundant samples via cosine similarity",
    "Supports lifelong instruction tuning across sequential datasets",
  ],
  "Entropy maximization within pseudo-task clusters ensures diverse, informative sample selection under budget constraints",
  "LLaVA 1.5, TinyLLaVA",
  "LLaVA-1.5, M3IT, MiniGPT4, MANTIS, LAMM, VisionFLAN"
);
addPaperResultsSlide("ADAPT-INF", [
  "Multi-task training",
  "Sequential training",
  "Random Experience Replay",
  "Score-based (Perplexity, EL2N)",
  "SemDeDup, Density-based Pruning, COINCIDE"
], [
  "Outperforms baselines on continual learning benchmarks",
  "Maintains skill retention across sequential datasets",
  "Efficient with LITE variant for reduced pool sizes",
  "Balanced performance across diverse visual tasks"
], [
  "Gradient-based clustering effectively identifies pseudo-tasks in multimodal data",
  "Entropy-based selection within clusters prevents collapse to easy or hard examples",
  "Pruning mechanism controls data pool growth for scalable lifelong learning",
]);

// ═══════════════════════════════════════════════════
// PAPER 2: CRISP
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "CRISP: Clustered Importance Sampling for Pre-training",
  "ICLR 2025 | Apple",
  "Cluster data, then sample weighted by ratio of specialist to generalist dataset distributions",
  [
    "Compute SBERT embeddings for each token window from generalist data",
    "Cluster generalist dataset using hierarchical clustering",
    "Compute cluster histogram for specialist dataset",
    "Estimate importance weights: w(c) = P(c|D_s) / P(c|D_g)",
    "Sample clusters from generalist data proportional to importance weights",
    "Continue pre-training the language model on sampled data",
  ],
  "Importance sampling reweights clusters so training distribution matches the specialist domain",
  "350M to 7B LLMs",
  "RedPj2, PubMed, StackExchange, Wikipedia, ARC, MMLU"
);
addPaperResultsSlide("CRISP", [
  "Fine-tuning generalist models",
  "Task-specific pretraining",
  "DoGE method",
  "Cross-entropy difference (CED)"
], [
  "Consistent improvements over baselines across model scales",
  "Effective for both language modeling and MCQ tasks",
  "Scales well from 350M to 7B parameters",
  "Handles multi-task transfer learning scenarios"
], [
  "Importance sampling via clustering is a lightweight alternative to per-sample scoring",
  "SBERT-based clustering captures topical similarity effectively",
  "The specialist-to-generalist ratio provides an intuitive and effective weighting scheme",
]);

// ═══════════════════════════════════════════════════
// PAPER 3: DELIFT
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "DELIFT: Data Efficient Language Model Instruction Fine-Tuning",
  "ICLR 2025 | UIUC + IBM Research",
  "Compute pairwise utility of 1-shot examples, then use submodular subset selection to pick the best ones",
  [
    "Calculate pairwise utility metric U_F_ij for all data pairs",
    "Utility measures advantage of in-context example (x_j, y_j) for predicting y_i",
    "Construct kernel matrix from non-negative utilities",
    "Apply submodular optimization: Facility Location (FL), FLMI, or FLCG",
    "Greedy algorithm selects subset A maximizing the submodular objective",
    "Fine-tune model on selected subset A",
  ],
  "Submodular functions provide near-optimal subset selection with theoretical guarantees (1-1/e approximation)",
  "Llama-3.2-3B, Mistral-7B, Qwen2-72B",
  "Mix-Instruct, P3, HotpotQA, MMLU, MT-Bench, GSM-8k"
);
addPaperResultsSlide("DELIFT", [
  "Full Data training",
  "Random selection",
  "SelectIT, LESS",
  "DEFT-UCS"
], [
  "Achieves near full-data performance with 30% of data",
  "Outperforms random and other selection baselines",
  "Effective across instruction tuning and task-specific stages",
  "Works with both QLoRA and full fine-tuning"
], [
  "Pairwise utility from ICL naturally captures data complementarity and redundancy",
  "Different submodular objectives suit different fine-tuning stages (general vs task-specific)",
  "The connection between utility metric and pointwise mutual information provides theoretical grounding",
]);

// ═══════════════════════════════════════════════════
// PAPER 4: KMQ
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "KMQ: Diversify and Conquer",
  "ICLR 2025 | Under Review",
  "K-means for diversity, perplexity ratio of generated to gold responses as quality score",
  [
    "Initialize budget b and divide into N iterations",
    "Cluster data using K-means for diversity coverage",
    "Sample from clusters weighted by quality scores",
    "Fine-tune model on selected subset for one epoch",
    "Compute quality: S = -log(PPL(x+y_gen) / PPL(x+y_gold))",
    "Update cluster weights based on quality scores and iterate",
  ],
  "Iterative refinement adjusts cluster weights to progressively focus on difficult/valuable clusters",
  "Llama-2-7B, Llama-3-8B, Mistral-7B",
  "Alpaca, WizardLM"
);
addPaperResultsSlide("KMQ", [
  "Random selection",
  "Deita, QDIT",
  "k-Center, kM-Closest, kM-Random"
], [
  "Outperforms static selection methods on benchmarks",
  "Iterative refinement improves over single-pass selection",
  "Transfers well across different base models",
  "Strong results on both Alpaca and WizardLM"
], [
  "Combining diversity (K-means) with quality (perplexity ratio) is more effective than either alone",
  "Iterative resampling adapts to model's evolving capabilities during training",
  "Perplexity ratio provides a training-free quality signal that correlates with sample difficulty",
]);

// ═══════════════════════════════════════════════════
// PAPER 5: LENS
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "LENS: Finding Support Examples for In-Context Learning",
  "Fudan University",
  "Use perplexity advantage for quality and cosine similarity for diversity in ICL example selection",
  [
    "Compute InfoScore for each candidate example: measures prediction improvement",
    "InfoScore I(e, D) = sum of contribution gaps c(e, e') across dataset",
    "Progressive filtering: iteratively remove uninformative examples",
    "Initialize diverse permutations of filtered candidates",
    "Diversity-guided search balances informativeness and cosine similarity",
    "Select top-performing permutations as final support examples",
  ],
  "Two-stage filter-then-search balances computational efficiency with selection quality for ICL",
  "GPT-2 (GPT2-L)",
  "SST-2, SST-5, Amazon, MR, Subj, TREC, AGNews, DBPedia"
);
addPaperResultsSlide("LENS", [
  "Zero-shot, Random",
  "Herding, K-Center Greedy",
  "CRAIG, GradMatch",
  "Facility Location, Graph Cut, Glister"
], [
  "Consistent gains across 8 classification benchmarks",
  "Support examples transfer across different LMs",
  "Robust to permutation order of examples",
  "Progressive filtering reduces computational cost significantly"
], [
  "ICL example selection is fundamentally different from coreset selection for training",
  "Informativeness (model feedback) combined with diversity yields best ICL performance",
  "Complexity O(N log N) makes the approach practical for moderate-scale datasets",
]);

// ═══════════════════════════════════════════════════
// PAPER 6: PDS
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "PDS: Data Selection via Optimal Control",
  "ICLR 2025 | Tsinghua + Microsoft + Peking University",
  "Formulate data selection as an optimal control problem; solve via Pontryagin's Maximum Principle",
  [
    "Pre-training loss: L(theta, gamma) = sum gamma_n * l(x_n, theta)",
    "Formulate selection as: min_gamma sum J(theta_t) s.t. gradient descent dynamics",
    "Apply PMP: compute co-state vectors lambda via backward pass",
    "Update gamma based on gradient alignment with co-state vectors",
    "Train a data scorer model on proxy-computed quality scores",
    "Use Gumbel-Top-K sampling to select data for target model",
  ],
  "Optimal control theory provides principled framework connecting data weights to downstream task performance",
  "LIMA-scale models",
  "CommonCrawl, LIMA, DCLM"
);
addPaperResultsSlide("PDS", [
  "Conventional Pre-Training",
  "RHO-Loss",
  "DSIR",
  "IF-Score"
], [
  "Superior downstream task performance with selected subsets",
  "Effective data reduction while maintaining quality",
  "Scales across model sizes via proxy model approach",
  "PMP provides convergence guarantees for score optimization"
], [
  "Optimal control theory offers a mathematically principled approach to data selection",
  "Proxy model strategy makes the method practical for large-scale pre-training",
  "Co-state vectors capture long-horizon training dynamics beyond single-step influence",
]);

// ═══════════════════════════════════════════════════
// PAPER 7: QUAD
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "QUAD: Quality and Diversity for Data Selection",
  "ICLR 2025 | Beijing IT + SenseTime + Shanghai AI Lab",
  "K-means clustering for diversity, gradient similarity with validation set for quality via influence functions",
  [
    "Cluster candidate pool D_c into groups using K-means",
    "Compute influence function: I(D_r, z) = -grad L(theta, D_r) * (H+lambda I)^{-1} * grad L(theta, z)",
    "Multi-Armed Bandit (UCB) for cluster selection: CS_i = I_bar_i + alpha * sqrt(...)",
    "Select top-K clusters, sample data within each cluster",
    "Threshold-based selection: add data above influence threshold",
    "Kronecker product approximation for efficient Hessian computation",
  ],
  "UCB-based MAB framework naturally balances exploration (diversity) and exploitation (quality)",
  "GPT-4, LLaMA-3.1",
  "SlimPajama, FineWeb, LAMBADA, Openwebmath, FLAN"
);
addPaperResultsSlide("QUAD", [
  "Random sampling",
  "Qurating",
  "DSIR, PPL",
  "MATES"
], [
  "Outperforms baselines on multiple pre-training benchmarks",
  "UCB exploration improves over pure top-k cluster selection",
  "Efficient Hessian approximation via Kronecker product",
  "Robust across different clustering configurations"
], [
  "Influence functions adapted with Kronecker factorization scale to LLM pre-training",
  "MAB framework provides a principled way to balance quality and diversity without tuning",
  "Submodularity of the influence function ensures near-optimal greedy selection",
]);

// ═══════════════════════════════════════════════════
// PAPER 8: ROSE
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "ROSE: Reward-Oriented Data Selection",
  "ICLR 2025 | Under Review",
  "Estimate how much validation reward improves from a gradient step on each sample",
  [
    "Initialize model with 5% random subset of training data",
    "Transform validation set into preference pairs (winning/losing responses)",
    "Compute gradients on training data (Adam) and validation preferences (SGD)",
    "Estimate influence scores via gradient inner product",
    "Select top 5% of data points with highest influence scores",
    "Train final model on selected high-influence data",
  ],
  "Reward-based influence estimation aligns data selection with human preference optimization",
  "Llama-2-7B/13B, Llama-3.1-8B, Mistral-7B",
  "DOLLY, Open Assistant, FLAN V2, COT"
);
addPaperResultsSlide("ROSE", [
  "Random, BM25",
  "Representation-based data selection (RDS)",
  "DSIR, Influence Functions",
  "LESS, Shapley values"
], [
  "Outperforms all baselines on preference benchmarks",
  "Strong correlation between validation loss and test win rate",
  "Effective with only 5% of training data",
  "Transfers across model families and datasets"
], [
  "Preference-based objectives better align selection with actual deployment goals (human preferences)",
  "DPO-style loss provides a differentiable proxy for reward-based selection",
  "The method is surprisingly effective with very small selection budgets (5%)",
]);

// ═══════════════════════════════════════════════════
// PAPER 9: STAFF
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "STAFF: Speculative Coreset Selection",
  "ICLR 2025 | Xi'an Jiaotong + NTU + UMass",
  "Use a small proxy model to speculate importance scores, then verify on the target LLM",
  [
    "Fine-tune small model theta_s on full dataset D",
    "Compute speculative scores: S_d^s = ||grad L(theta_s(d))||_2",
    "Divide dataset into K regions based on speculative scores",
    "Verify each region on target LLM theta_t with a small budget",
    "Calculate verification ratio: V_i = sum(S_d^t) / sum(S_d^s) per region",
    "Allocate selection budget based on verification scores, compile final coreset",
  ],
  "Speculative execution paradigm: cheap proxy estimates guide expensive target model verification",
  "Gemma-7b, Llama-2-13b, Mistral-Nemo",
  "BioInstruct, DialogSum, WMT-19 (Kazakh-English)"
);
addPaperResultsSlide("STAFF", [
  "Random selection",
  "GraNd, EL2N",
  "CCS",
  "D2 Pruning"
], [
  "Outperforms baselines across pruning rates (20-80%)",
  "Effective on diverse tasks: QA, summarization, translation",
  "Significant speedup over full target-model scoring",
  "Robust to choice of small model architecture"
], [
  "Speculative execution from systems research transfers effectively to data selection",
  "Region-based verification amortizes the cost of target model evaluation",
  "The small-to-large model score correlation is strong enough for practical selection",
]);

// ═══════════════════════════════════════════════════
// PAPER 10: TrackStar
// ═══════════════════════════════════════════════════
addPaperMethodSlide(
  "TrackStar: Scalable Influence and Fact Tracing",
  "ICLR 2025 | Google DeepMind + UC San Diego",
  "Dot product of down-projected, Hessian-corrected gradients as influence/quality score",
  [
    "Compute loss gradients for training and query examples",
    "Apply optimizer state correction: divide by sqrt(V) (second moment)",
    "Project gradients to lower dimension via random projection matrix P_d",
    "Apply Hessian correction: R^{-1/2} with mixed train/eval Hessian approximation",
    "Unit-normalize corrected gradients to reduce outlier influence",
    "Influence score: I(z_m, z_q) = G_bar(z_m) . G_bar(z_q)",
  ],
  "Careful gradient preprocessing (correction + projection + normalization) enables scalable influence at 8B scale",
  "154M, 1B, 8B parameter LMs",
  "English C4, T-REx, KILT"
);
addPaperResultsSlide("TrackStar", [
  "BM25",
  "Gecko embeddings",
  "TRAK",
  "Ablated TrackStar variants"
], [
  "State-of-the-art fact tracing on T-REx and C4",
  "Scales to 8B parameters and 160B token corpora",
  "No pre-filtering or subsampling required",
  "Strong tail-patch influence metric results"
], [
  "Full gradient-based influence is feasible at LLM scale with proper engineering",
  "Each preprocessing step (correction, projection, normalization) contributes meaningfully",
  "Fact tracing has direct applications for data attribution and selection in pre-training",
]);

// ═══════════════════════════════════════════════════
// SLIDE 25: COMPARATIVE ANALYSIS TABLE
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("Comparative Analysis");
  const header = [
    { text: "Method", options: { fill: { color: C.navy }, color: C.white, bold: true, fontSize: 10, fontFace: FONTS.body, align: "center", valign: "middle" } },
    { text: "Stage", options: { fill: { color: C.navy }, color: C.white, bold: true, fontSize: 10, fontFace: FONTS.body, align: "center", valign: "middle" } },
    { text: "Quality Signal", options: { fill: { color: C.navy }, color: C.white, bold: true, fontSize: 10, fontFace: FONTS.body, align: "center", valign: "middle" } },
    { text: "Diversity Signal", options: { fill: { color: C.navy }, color: C.white, bold: true, fontSize: 10, fontFace: FONTS.body, align: "center", valign: "middle" } },
    { text: "Optimization", options: { fill: { color: C.navy }, color: C.white, bold: true, fontSize: 10, fontFace: FONTS.body, align: "center", valign: "middle" } },
  ];
  const cellOpts = (alt) => ({ fontSize: 9, fontFace: FONTS.body, color: C.darkGray, fill: { color: alt ? C.offWhite : C.white }, valign: "middle" });
  const rows = [
    ["ADAPT-INF", "Fine-tune", "Entropy scoring", "K-means clusters", "CCS sampling"],
    ["CRISP", "Pre-train", "Importance ratio", "Hierarchical clusters", "Importance sampling"],
    ["DELIFT", "Fine-tune", "Pairwise utility (ICL)", "Submodular coverage", "Greedy submodular"],
    ["KMQ", "Fine-tune", "Perplexity ratio", "K-means clusters", "Iterative resampling"],
    ["LENS", "ICL", "InfoScore (PPL gap)", "Cosine similarity", "Filter-then-search"],
    ["PDS", "Pre-train", "Gradient alignment", "Implicit (Gumbel)", "Optimal control (PMP)"],
    ["QUAD", "Pre-train", "Influence function", "K-means + MAB", "UCB exploration"],
    ["ROSE", "Fine-tune", "Reward influence", "Implicit", "Gradient inner prod."],
    ["STAFF", "Fine-tune", "Gradient norm", "Region-based", "Speculative execution"],
    ["TrackStar", "Pre-train", "Hessian-corrected grad", "Implicit", "Dot product ranking"],
  ];
  const tableData = [header, ...rows.map((r, i) => r.map(cell => ({ text: cell, options: cellOpts(i % 2 === 0) })))];
  s.addTable(tableData, { x: 0.3, y: 1.1, w: 9.4, colW: [1.3, 1.1, 2.2, 2.2, 2.6], border: { pt: 0.5, color: C.lightGray }, autoPage: false });
}

// ═══════════════════════════════════════════════════
// SLIDE 26: COMMON THEMES
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("Common Themes Across Methods");
  const themes = [
    { title: "Clustering for Diversity", items: "K-means (ADAPT-INF, KMQ, QUAD), hierarchical clustering (CRISP), region-based division (STAFF). Nearly all methods use some form of clustering." },
    { title: "Gradient/Influence-Based Quality", items: "Gradient norms (STAFF), gradient alignment (PDS, ROSE), influence functions (QUAD, TrackStar), perplexity-based scores (CRISP, KMQ, LENS)." },
    { title: "Submodular Optimization", items: "DELIFT explicitly uses submodular functions (FL, FLMI, FLCG). QUAD proves submodularity of influence. Greedy algorithms provide 1-1/e guarantees." },
    { title: "Proxy Models for Efficiency", items: "STAFF uses small model to speculate scores. PDS trains proxy for co-state computation. TrackStar uses random projection for dimensionality reduction." },
  ];
  themes.forEach((t, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const xx = 0.5 + col * 4.7;
    const yy = 1.1 + row * 2.15;
    addCard(s, xx, yy, 4.4, 2.0);
    s.addShape(pres.shapes.RECTANGLE, { x: xx, y: yy, w: 0.07, h: 2.0, fill: { color: C.accent } });
    s.addText(t.title, { x: xx + 0.2, y: yy + 0.1, w: 4, h: 0.35, fontSize: 14, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
    s.addText(t.items, { x: xx + 0.2, y: yy + 0.5, w: 4, h: 1.4, fontSize: 11, fontFace: FONTS.body, color: C.darkGray, valign: "top", margin: 0 });
  });
}

// ═══════════════════════════════════════════════════
// SLIDE 27: OPEN CHALLENGES
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("Open Challenges & Future Directions");
  const challenges = [
    { title: "Scalability", desc: "Most methods tested on millions of samples. Scaling to trillion-token corpora remains a major challenge. Proxy-based approaches (STAFF, PDS) show promise." },
    { title: "Transferability", desc: "Selection policies learned for one model may not transfer to another. Few works study cross-model or cross-task transferability systematically." },
    { title: "Theoretical Guarantees", desc: "Submodularity-based methods (DELIFT, QUAD) offer approximation guarantees. Optimal control (PDS) provides convergence results. Most methods lack formal bounds." },
    { title: "Online/Streaming Selection", desc: "Current methods are largely offline (batch selection). Real-time selection for streaming pre-training data is an open problem." },
    { title: "Multi-Objective Selection", desc: "Balancing quality, diversity, safety, and fairness simultaneously. Most methods optimize at most two objectives." },
    { title: "Selection for Alignment", desc: "ROSE pioneers reward-oriented selection. Extending selection to RLHF, constitutional AI, and safety alignment remains underexplored." },
  ];
  challenges.forEach((ch, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const xx = 0.5 + col * 3.1;
    const yy = 1.1 + row * 2.15;
    addCard(s, xx, yy, 2.85, 2.0);
    s.addShape(pres.shapes.RECTANGLE, { x: xx, y: yy, w: 2.85, h: 0.05, fill: { color: C.accent } });
    s.addText(ch.title, { x: xx + 0.15, y: yy + 0.15, w: 2.55, h: 0.35, fontSize: 13, fontFace: FONTS.heading, color: C.navy, bold: true, margin: 0 });
    s.addText(ch.desc, { x: xx + 0.15, y: yy + 0.5, w: 2.55, h: 1.35, fontSize: 10, fontFace: FONTS.body, color: C.darkGray, valign: "top", margin: 0 });
  });
}

// ═══════════════════════════════════════════════════
// SLIDE 28: CONCLUSION
// ═══════════════════════════════════════════════════
{
  const s = addDarkSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: C.darkNavy } });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 0.8, w: 2, h: 0.05, fill: { color: C.accent } });
  s.addText("Conclusion", { x: 1.5, y: 1.0, w: 7, h: 0.6, fontSize: 28, fontFace: FONTS.heading, color: C.white, bold: true, margin: 0 });

  const conclusions = [
    "Subset selection is a critical enabler for efficient LLM training at scale",
    "Quality and diversity are complementary objectives - the best methods address both",
    "Clustering (K-means, hierarchical) is the dominant approach for ensuring diversity",
    "Gradient-based signals (influence functions, gradient norms/alignment) are the leading quality indicators",
    "Proxy models and dimensionality reduction make selection scalable to large models",
    "Submodular optimization and optimal control provide strong theoretical foundations",
    "Future work should address streaming selection, multi-objective optimization, and alignment-aware selection",
  ];
  conclusions.forEach((c, i) => {
    s.addText(c, {
      x: 1.5, y: 1.8 + i * 0.48, w: 7, h: 0.4,
      fontSize: 13, fontFace: FONTS.body, color: C.ice, bullet: true, margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════
// SLIDE 29: REFERENCES
// ═══════════════════════════════════════════════════
{
  const s = addContentSlide("References");
  const refs = [
    "[1] ADAPT-INF: Scalable Continual Multimodal Instruction Tuning via Dynamic Data Selection. ICLR 2025.",
    "[2] CRISP: Task-Adaptive Pretrained Language Models via Clustered Importance Sampling. ICLR 2025.",
    "[3] DELIFT: Data Efficient Language Model Instruction Fine-Tuning. ICLR 2025.",
    "[4] KMQ: Diversify and Conquer - Diversity-Centric Data Selection with Iterative Refinement. ICLR 2025.",
    "[5] LENS: Finding Support Examples for In-Context Learning. Fudan University.",
    "[6] PDS: Data Selection via Optimal Control for Language Models. ICLR 2025.",
    "[7] QUAD: Harnessing Diversity for Important Data Selection in Pretraining LLMs. ICLR 2025.",
    "[8] ROSE: Reward-Oriented Data Selection for LLM Task-Specific Instruction Tuning. ICLR 2025.",
    "[9] STAFF: Speculative Coreset Selection for Task-Specific Fine-Tuning. ICLR 2025.",
    "[10] TrackStar: Scalable Influence and Fact Tracing for LLM Pretraining. ICLR 2025.",
  ];
  refs.forEach((r, i) => {
    s.addText(r, { x: 0.5, y: 1.1 + i * 0.42, w: 9, h: 0.38, fontSize: 11, fontFace: FONTS.body, color: C.darkGray, valign: "middle", margin: 0 });
  });
}

// ═══════════════════════════════════════════════════
// GENERATE FILE
// ═══════════════════════════════════════════════════
pres.writeFile({ fileName: "C:\\Users\\soura\\Desktop\\digest\\subset-selection-llm\\Subset_Selection_LLM_Review.pptx" })
  .then(() => console.log("Presentation created successfully!"))
  .catch(err => console.error("Error:", err));
