# Energy in Biological Systems

Living systems obey the same thermodynamic laws as any physical system, but they have evolved astonishingly efficient mechanisms for capturing and using energy. Two processes stand out: photosynthesis in autotrophs, and cellular respiration in heterotrophs.

Photosynthesis captures sunlight and stores it as chemical bonds in glucose, splitting water and releasing oxygen as a by-product. The global oxygen balance of the atmosphere is maintained largely by this process — plants, algae, and cyanobacteria together produce roughly 270 billion tons of oxygen per year.

Cellular respiration reverses the bookkeeping: it oxidizes glucose back to carbon dioxide and water, releasing energy captured as ATP. The mitochondrion, the eukaryotic cell's power plant, is the site of most of this. Aerobic respiration extracts about 30–32 ATP per glucose, compared with only 2 ATP from anaerobic glycolysis — a staggering efficiency advantage that explains why oxygen-breathing metabolism dominates complex life.

## Parallel with Machine Learning

It is tempting to draw a parallel between biological representation learning and the hierarchical representations learned by deep neural networks. Both systems extract useful features from noisy sensory input; both exploit layered architectures; both appear to learn via gradient-like credit assignment (in the brain this remains contested; in networks it is exactly backpropagation, modulo implementation).

The analogy is imperfect. Biological neurons fire with discrete spikes, have complex dendritic computations, are modulated by neurotransmitters, and learn with far less data than any artificial network requires. Still, the convergence of form — hierarchical, distributed, plastic — is striking.

## Concept Map

- Photosynthesis → glucose, O2
- Glucose → cellular respiration → ATP, CO2
- CO2 → (plants) → photosynthesis (cycle closes)
- ATP powers: biosynthesis, active transport, muscle contraction, neural signaling

The entire biosphere runs on this closed cycle, itself powered by the Sun. Remove the Sun and the biosphere, powered only by residual chemical gradients (hydrothermal vents, radiolysis), would collapse to a tiny fraction of its present mass within a few centuries.

## Linear Algebra Sidebar

Metabolic networks are, mathematically, signed bipartite graphs, and much of modern systems biology uses linear algebra to analyze them. Flux balance analysis treats reaction rates as a vector in a space constrained by stoichiometry (a matrix S where rows are metabolites and columns are reactions). Steady-state solutions satisfy Sv = 0; optimization against a biological objective (growth rate, ATP yield) determines which solution an organism plausibly uses. The tools — null spaces, linear programming — are exactly those used in neural network analysis, applied to very different systems.
